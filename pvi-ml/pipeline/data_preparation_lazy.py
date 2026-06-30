from src.packages import *
from src.utils.primitives import *
from src.utils import miscellaneous as misc
from src.utils import h5io
from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_extraction import PviRawDataset
from src.pipeline._data_preparation import PviConfiguredDataset

from src.pipeline.graph_partitioner import GraphBipartitePartitioner as gbp
from src.pipeline.batch_sampler import PviBatchSampler

from collections import OrderedDict
import gc

class PviLazyDataset(PviConfiguredDataset):

    @dataclass
    class DatasetMappings:
        files: list[PviRawDataset]  # Which file each sample comes from
        masks_global: list[tuple[int, int]]  # Global mask for each sample
        masks_local: list[tuple[int, int]]  # Local mask for each sample
        indices_global: list[int]  # Global index (might be redundant)
        indices_local: list[int]  # Local index within file
        partition: list[str] # Local partition within file

    @dataclass
    class DatasetGroupings:
        files: list[PviRawDataset]  # Unique files (no repetition)
        masks_global: list[list[tuple[int, int]]]  # Global masks grouped by file
        masks_local: list[list[tuple[int, int]]]  # Local masks grouped by file
        indices_global: list[list[int]]  # Global indices grouped by file
        indices_local: list[list[int]]  # Local indices grouped by file
        partition: list[list[str]]  # Local partition grouped by file
        dataset_bounds: list[tuple[int, int]]

    access_counter: int = 0

    def __init__(self,
                 ds_files: list[PviDataFile]|PviDatasetInventory,
                 input_mode: str|InputMode,
                 output_mode: str|OutputMode,
                 mask_key: str|SequenceMask,
                 name: str = 'dataset_lazy',
                 max_cache: int=5,
                 persistent_handle: bool=False,
                 ) -> None:

        super().__init__(input_mode=input_mode,
                         output_mode=output_mode,
                         mask_key=mask_key)

        self._validate_components(ds_files)

        self._alias = type(self).__name__  # temporary alias
        self.name = name
        self.path = 'N/A'

        self.inventory: list[PviDataFile]|PviDatasetInventory = ds_files
        self.raws: list[PviRawDataset] = []
        self.persistent_handle = persistent_handle

        self.period_length: int = 0
        self.num_periods: int = 0

        self.input_mode = InputMode(input_mode)
        self.output_mode = OutputMode(output_mode)
        self.mask_key = SequenceMask(mask_key)

        self.active_mask: list[tuple[int,int]] = []

        self._device = torch.device('cpu')
        self._dtype = torch.float32
        self._transfer_kwargs: dict = {}

        self.mappings: 'DatasetMappings' = None
        self.groupings: 'DatasetGroupings' = None

        self._split_params: dict = {}
        self._loader_params: dict = {}

        # these are unique to lazy dataset
        self.test_subgroups: list[str] = []
        self.train_subgroups: list[str] = []

        self.max_cache = max_cache
        self.cache = None

        self._alias = f"{self.name} (LazyLoader)"

    def _validate_components(self, ds_files) -> None:
        crit1 = isinstance(ds_files, PviDatasetInventory)
        crit2 = isinstance(ds_files, (tuple, list, set)) and all([isinstance(dsf, PviDataFile) for dsf in ds_files])
        if not (crit1 or crit2):
            raise TypeError(
                f"Invalid components type for class '{type(self)}'! Expect 'PviDatasetInventory' or an iterable of 'PviDataFile'.")

    def _survey_datasets(self,
                         raws: list[PviRawDataset]=None,
                         mask_key: SequenceMask=None) -> 'DatasetMappings':
        raws = self.raws if raws is None else raws
        mask_key = self.mask_key if mask_key is None else mask_key

        total = len(raws)

        # Mappings (sample-level, same length as total samples)
        mapping_files = []
        mapping_masks_global = []
        mapping_masks_local = []
        mapping_idx_global = []
        mapping_idx_local = []

        # Groupings (file-level, same length as number of files)
        grouping_files = []
        grouping_masks_global = []
        grouping_masks_local = []
        grouping_idx_global = []
        grouping_idx_local = []
        grouping_bounds = []

        offset_periods = 0
        offset_sequences = 0

        for _, ds_raw in enumerate(raws):
            masks_local = ds_raw.masks[mask_key.value]
            masks_global = [misc.offset_integers(m, offset_periods) for m in masks_local]
            idx_local = list(range(len(masks_local)))
            idx_global = [offset_sequences + i for i in idx_local]

            grouping_files.append(ds_raw)
            grouping_bounds.append((offset_periods, offset_periods + ds_raw.num_periods))
            grouping_masks_local.append(masks_local)
            grouping_masks_global.append(masks_global)
            grouping_idx_local.append(idx_local)
            grouping_idx_global.append(idx_global)

            mapping_files.extend([ds_raw]*len(masks_local))
            mapping_masks_global.extend(masks_global)
            mapping_masks_local.extend(masks_local)
            mapping_idx_global.extend(idx_global)
            mapping_idx_local.extend(idx_local)

            offset_periods += ds_raw.num_periods
            offset_sequences += len(masks_local)

        mappings = self.DatasetMappings(
                files=mapping_files,
                masks_global=mapping_masks_global,
                masks_local=mapping_masks_local,
                indices_global=mapping_idx_global,
                indices_local=mapping_idx_local,
                partition=[] # require partition state
        )

        groupings = self.DatasetGroupings(
                files=grouping_files,
                masks_global=grouping_masks_global,
                masks_local=grouping_masks_local,
                indices_global=grouping_idx_global,
                indices_local=grouping_idx_local,
                dataset_bounds=grouping_bounds,
                partition = [] # require partition state
        )

        print(f"{self._alias}: Finish mapping {total} local datasets to global masks.")

        return mappings, groupings

    @property
    def num_frames(self) -> int:
        return self.num_periods * self.period_length

    def build(self) -> 'PviLazyDataset':
        self.raws = [PviRawDataset(file,persistent=self.persistent_handle) for file in self.inventory]
        self.period_length = self.raws[0].period_length
        self.num_periods = sum([ds_raw.num_periods for ds_raw in self.raws])

        self.mappings, self.groupings = self._survey_datasets()
        self.active_mask = self.mappings.masks_global

        return self

    def print_info(self) -> None:
        pass

    def set_partition(self,
                      test_size: float=0.1,
                      shuffle: bool=False,
                      split_mode: str|SplitMode = 'local',
                      **kwargs) -> None:

        params = {'test_size': test_size,
                  'shuffle': shuffle,
                  'split_mode': SplitMode(split_mode).value,
                  **kwargs}

        self._split_params = params

    def _compute_samples_pooling(self) -> tuple:

        test_size = self._split_params['test_size']
        split_mode = SplitMode(self._split_params['split_mode'])

        all_subjects = [file.subject for file in self.groupings.files]
        all_mask_subgroups = self.groupings.masks_global

        # combining new subgroups by subjects
        # we need this because the subjects are scattered into multiple files/subgroups
        subgroups = {}
        for subject, mask in zip(all_subjects, all_mask_subgroups):
            if subject not in subgroups:
                subgroups[subject] = mask
            else:
                subgroups[subject].extend(mask)

        if split_mode == SplitMode.GLOBAL:
            return self.active_mask, test_size

        if split_mode == SplitMode.WITHIN: # local
            return list(subgroups.values()), [test_size]*len(subgroups)

        if split_mode == SplitMode.DISJOINT:
            unique_subjects = list(subgroups.keys())

            min_required = 5
            if len(unique_subjects) < min_required:
                raise RuntimeError(f"Cannot perform subject-wise partition! Require at least {min_required} unique subjects.")

            if self._split_params.get('shuffle'):
                rng = random.Random(self._split_params.get('random_state'))
                rng.shuffle(unique_subjects)

            mask_subgroups = [subgroups[s] for s in unique_subjects]

            test_size_subgroups = [0.]*len(mask_subgroups)
            test_keys = []
            num_test = 0
            for k, subject in enumerate(unique_subjects):
                candidate_count = len(subgroups[subject])
                projected_ratio = (num_test + candidate_count) / len(self.active_mask)

                if num_test > 0 and projected_ratio > 1.2 *test_size:
                    break

                test_keys.append(subject)
                test_size_subgroups[k] = 1.0
                num_test += candidate_count

            self.test_subgroups = test_keys
            self.train_subgroups = [subject for subject in unique_subjects if subject not in test_keys]

            return mask_subgroups, test_size_subgroups

        elif split_mode == SplitMode.MIXED:
            raise NotImplementedError("Mixed partition mode not yet implemented!")

        else:
            raise KeyError("Invalid partition mode!")

    def state_dict(self) -> dict:
        state_dict = super().state_dict()

        # quick fix
        for kw, value in state_dict.items():
            if kw in self._VALID_MASK_ATTRIBUTES:
                value.sort()

        state_dict['test_subgroups'] = sorted(self.test_subgroups)
        state_dict['train_subgroups'] = sorted(self.train_subgroups)

        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        # quick fix
        for kw, value in state_dict.items():
            if kw in self._VALID_MASK_ATTRIBUTES:
                value.sort()

        super().load_state_dict(state_dict)

        self.map_local_partition()

    def map_local_partition(self) -> None:

        train_map: set = set(self.train_mask)
        test_map: set = set(self.test_mask)

        self.mappings.partition = []
        self.groupings.partition = []

        for group in self.groupings.masks_global:
            new = []
            for gm in group:
                if gm in train_map:
                    new.append('train')
                elif gm in test_map:
                    new.append('test')
                else:
                    new.append('none')

            self.mappings.partition.extend(new)
            self.groupings.partition.append(new)

    def get_partition(self) -> dict[str, Subset | Dataset]:

        if not self.active_mask:
            raise AttributeError("Cannot split dataset! Active mask not set.")

        if not self._split_params:
            raise AttributeError("Cannot split dataset! Split params not set.")

        if not self.groupings.dataset_bounds:
            raise AttributeError("Cannot split dataset! Dataset bounds not given.")

        mask_subgroups, test_size_subgroups = self._compute_samples_pooling()

        gbp_params = {k: v for k, v in self._split_params.items()
                      if k not in ['test_size', 'split_mode', 'test_subgroups', 'train_subgroups']}

        splitter = gbp(intervals=mask_subgroups,
                       test_size=test_size_subgroups,
                       **gbp_params)

        self.train_mask, self.test_mask = splitter.split()

        self.subsets = self._get_subsets_from_split()

        self.map_local_partition()

        return self.subsets

    def _validate_build(self) -> None:
        pass

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self,
           device:torch.device|str=None,
           dtype: torch.dtype=None,
           **kwargs) -> 'PviLazyDataset':

        self._device = device
        self._dtype = dtype
        self._transfer_kwargs.update(kwargs)

        print(f"{self._alias} (WARNING): Lazy tensors will be transferred upon indexing.")

        if self.cache and len(self.cache):
            for key, data in self.cache.items():
                self.cache[key] = h5io.transfer(data, self._device, self._dtype, **self._transfer_kwargs)

        return self

    def _get_new_data(self, ds_raw) -> dict[str,torch.Tensor]:
        ds_raw.data = ds_raw.extract_tensors(input_mode=self.input_mode, idx=None)

        formatted_data = h5io.format_raw_tensors(raw_data=ds_raw.data,
                                                 input_mode=self.input_mode,
                                                 output_mode=self.output_mode,
                                                 period_length=self.period_length)

        formatted_data = h5io.transfer(formatted_data,
                                       device=self._device,
                                       dtype=self._dtype,
                                       **self._transfer_kwargs)

        return formatted_data

    def __getitem__(self, idx: int) -> dict[str,torch.Tensor]:
        self.access_counter += 1

        ds_raw = self.mappings.files[idx]
        local_mask = self.mappings.masks_local[idx]

        hit = bool(self.cache) and (ds_raw in self.cache)

        # print()
        # print(f"Access counter: {self.access_counter}")
        # print(f"\t cache_hit = {hit}")

        if hit:
            formatted_data = self.cache[ds_raw]
        else:
            formatted_data = self._get_new_data(ds_raw)
            self._update_cache(key=ds_raw, data=formatted_data)

        sequence = h5io.slice_sequences(data=formatted_data,
                                        bounds=local_mask,
                                        period_length=self.period_length)
        return sequence

    def _update_cache(self,
                      key: PviRawDataset,
                      data: dict[str,torch.Tensor],
                      cleanup: bool=True) -> None:

        if self.cache is None:
            self.cache: OrderedDict[PviRawDataset, dict[str, torch.Tensor]] = OrderedDict()

        self.cache[key] = data
        if len(self.cache) > self.max_cache:
            ds_raw_oldest, formatted_oldest = self.cache.popitem(last=False) # pop first item (FIFO)
            ds_raw_oldest.unload()
            del ds_raw_oldest
            del formatted_oldest

            if cleanup:
                gc.collect()

    def set_dataloaders(self,
                        batch_size: int=32,
                        shuffle: bool=True,
                        stratified: bool=False,
                        **kwargs) -> None:

        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  'stratified': stratified,
                  **kwargs}

        if stratified:
            print()
            print(f"{self._alias} (WARNING): Custom BatchSampler will be used. Proceed with caution!")
            time.sleep(0.5)

            params['cluster_size'] = max(5, self.max_cache - 2)

            if shuffle:
                print(f"{self._alias} (WARNING): 'shuffle' is ignored when using custom batch sampler!")

        params['persistent_handle'] = self.persistent_handle

        self._loader_params = params

    def get_dataloaders(self) -> dict[str, DataLoader]:

        if not self.subsets:
            raise AttributeError("Cannot get DataLoaders! Train and test subsets not available. Call get_partition() before proceed.")

        if not self._loader_params:
            raise AttributeError("Cannot get DataLoaders! Loader params not set.")

        if self._loader_params.get('stratified') == False:
            params = {k: v for k, v in self._loader_params.items()
                      if k not in ['stratified', 'cluster_size', 'persistent_handle']}

            self._loader_params = params
            self.loaders = super().get_dataloaders()

        else:
            train_sampler = PviBatchSampler(global_indices=self.subsets['train'].indices,
                                            grouping=self.groupings.indices_global,
                                            cluster_size=self._loader_params['cluster_size'],
                                            batch_size=self._loader_params['batch_size'],
                                            )

            test_sampler = PviBatchSampler(global_indices=self.subsets['test'].indices,
                                            grouping=self.groupings.indices_global,
                                            cluster_size=self._loader_params['cluster_size'],
                                            batch_size=self._loader_params['batch_size'],
                                            )

            worker_params = {k: v for k, v in self._loader_params.items()
                            if k in ['num_workers', 'prefetch_factor', 'pin_memory', 'persistent_workers']}

            train_params = {'batch_sampler': train_sampler}
            train_params.update(worker_params)

            # test_params = {'batch_size': self._loader_params['batch_size'], 'shuffle': False}
            test_params = {'batch_sampler': test_sampler}
            test_params.update(worker_params)

            self.loaders = {"train": DataLoader(self.subsets['train'], **train_params),
                            "test": DataLoader(self.subsets['test'], **test_params)}

            print(f"{self._alias}: Finish making DataLoaders with STRATIFIED RANDOM SAMPLER.")
            # print(f"{self._alias} (WARNING): Test loader was NOT stratified and instead uses default DataLoader!")

        return self.loaders

"""
TESTING AREA 
"""

def benchmark_loaders(ds_lazy:PviLazyDataset,
                      num_trials:int=50,
                      shuffle: bool=True) -> None:

    ds_lazy.set_partition(test_size=0.01,
                          shuffle=shuffle)

    ds_lazy.set_dataloaders(batch_size=32,
                            shuffle=True,
                            stratified=True,
                            pin_memory=True,
                            )

    _ = ds_lazy.get_partition()

    loader = ds_lazy.get_dataloaders()['train']
    samples = iter(loader)

    # batch_subgroups = loader.batch_sampler.batch_subgroups
    # idx_subgroups = iter(batch_subgroups)

    profile_lazy = []
    for trial_counter in range(num_trials):
        t1 = time.perf_counter()
        # indices = next(idx_subgroups)
        batch = next(samples)
        dt = time.perf_counter() - t1

        print()
        print(f"Batch trial #{trial_counter + 1}:")
        # print(f"\t Current indices: {indices}")
        print(f"\t Time: {dt:,.2f} s")
        print(f"\t Current cache: {len(ds_lazy.cache)}/{ds_lazy.max_cache}")

        profile_lazy.append(dt)

    print(f"Average time (lazy):"
          f"{sum(profile_lazy) / len(profile_lazy):,.2f} s, "
          f"max={max(profile_lazy):,.2f} s, "
          f"min={min(profile_lazy):,.2f} s")

def benchmark_wasserstein(ds_lazy:PviLazyDataset,
                      shuffle: bool=True) -> None:

    ds_lazy.set_partition(test_size=0.1,
                          shuffle=shuffle,
                          split_mode='disjoint')

    ds_lazy.set_dataloaders(batch_size=32,
                            shuffle=True,
                            stratified=True,
                            pin_memory=False,
                            )

    from src.pipeline._data_preparation import ensemble_distance
    D1 = ds_lazy.subsets['train']
    D2 = ds_lazy.subsets['test']

    wd = ensemble_distance(D1, D2, ds_lazy.input_mode)

if __name__ == '__main__':

    inventory = PviDatasetInventory(branch='main')

    dsp1 = PviLazyDataset(ds_files=inventory,
                          input_mode=InputMode.IMPEDANCE,
                          output_mode=OutputMode.WAVEFORM,
                          mask_key=SequenceMask.MASK05,
                          ).build()

    # benchmark_wasserstein(ds_lazy)

    inventory = PviDatasetInventory(branch='holdout')
    dsp2 = PviLazyDataset(ds_files=inventory,
                          input_mode=InputMode.IMPEDANCE,
                          output_mode=OutputMode.WAVEFORM,
                          mask_key=SequenceMask.MASK05,
                          ).build()

    inventory = PviDatasetInventory(branch='longitudinal')
    dsp3 = PviLazyDataset(ds_files=inventory,
                          input_mode=InputMode.IMPEDANCE,
                          output_mode=OutputMode.WAVEFORM,
                          mask_key=SequenceMask.MASK05,
                          ).build()

    p1 = dsp1.get_raw_statistics()
    p2 = dsp2.get_raw_statistics()
    p3 = dsp3.get_raw_statistics()

    pass