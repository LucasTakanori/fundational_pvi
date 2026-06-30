from src.packages import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset

from scipy import stats as stats
import math
import ot
import gc

from tqdm import tqdm
from src.utils.primitives import DefaultStringFormat as dfmt
from src.utils.primitives import *
from src.utils import miscellaneous as misc
from src.utils import h5io as h5io
from src.pipeline.data_extraction import PviRawDataset

class PviConfiguredDataset(Dataset, ABC):
    """
    A class to transform the raw h5 data to torch-compatible format
    for use in training loop
    """

    @dataclass
    class DatasetConfigurations:
        input_mode: InputMode = InputMode.IMAGE
        output_mode: OutputMode = OutputMode.WAVEFORM
        mask_key: SequenceMask = SequenceMask.MASK05

    # @dataclass
    # class DatasetMappings:
    #     files: list[PviRawDataset]  # Which file each sample comes from
    #     masks_global: list[tuple[int, int]]  # Global mask for each sample
    #     masks_local: list[tuple[int, int]]  # Local mask for each sample
    #     indices_global: list[int]  # Global index (might be redundant)
    #     indices_local: list[int]  # Local index within file
    #     partition: list[str]  # Local partition within file
    #
    # @dataclass
    # class DatasetGroupings:
    #     files: list[PviRawDataset]  # Unique files (no repetition)
    #     masks_global: list[list[tuple[int, int]]]  # Global masks grouped by file
    #     masks_local: list[list[tuple[int, int]]]  # Local masks grouped by file
    #     indices_global: list[list[int]]  # Global indices grouped by file
    #     indices_local: list[list[int]]  # Local indices grouped by file
    #     partition: list[list[str]]  # Local partition grouped by file
    #     dataset_bounds: list[tuple[int, int]]

    def __init__(self,
                 input_mode: str|InputMode,
                 output_mode: str|OutputMode,
                 mask_key: str|SequenceMask,
                 verbose: bool=True,
                 ) -> None:
        Dataset.__init__(self)

        # The following assignments give the required attributes for subclasses.
        # I don't know of a way to 'declare' abstract attributes (or properties?)
        # WITHOUT DIRECTLY ASSIGNING their values, but instead contract the subclasses
        # to implement/assign those attributes. In addition, it will give an error
        # if we forget to define those attributes (or misspelled) in the subclasses.
        # There is the recommended approach of chaining @property and @abstractmethod
        # (in that order), but then the properties ar defined like a method.
        # For a long list of attributes, this gets cumbersome, and we have to
        # implement the subclasses' attributes as @property method as well.

        self.input_mode = InputMode(input_mode)
        self.output_mode = OutputMode(output_mode)
        self.mask_key = SequenceMask(mask_key)
        self._verbose = verbose

        self.name: str = ...
        self._alias = f"{self.name} (Configured)" if self.name else type(self).__name__

        self.period_length: int = 0
        self.num_periods: int = 0

        self.raws: ... = None

        self.offset_period: int = 0

        self.active_mask: list[tuple[int]] = []
        self.train_mask: list[tuple[int]] = []
        self.test_mask: list[tuple[int]] = []

        self._split_params: dict = {}
        self._loader_params: dict = {}

        self.subsets: dict[str, Subset] = {}
        self.loaders: dict[str, DataLoader]= {}

        self.data: dict = {}

        # Define repetitive string keys to avoid mistake
        self._VALID_CONFIGS_KEYS = ['input_mode', 'output_mode', 'mask_key']

        self._VALID_MASK_ATTRIBUTES = ["active_mask", "train_mask", "test_mask"]

    def __len__(self) -> int:
        if not self.active_mask:
            raise ValueError("Cannot determine length for '{self._alias}'! Active mask not available.")

        return len(self.active_mask)

    def __getitem__(self, idx: int) -> dict:
        bounds = self.active_mask[idx]
        sample = h5io.slice_sequences(data=self.data,
                                      bounds=bounds,
                                      period_length=self.period_length)
        return sample

    # @abstractmethod
    def _validate_components(self, components) -> None:
        pass

    @abstractmethod
    def build(self) -> 'PviConfiguredDataset':
        pass

    def cleanup(self, attrs: list[str] | str, placeholder=None) -> None:
        misc.cleanup_attributes(self, attrs=attrs, placeholder=placeholder)

    def print_info(self) -> None:
        self._print_info_default()

    def _print_info_default(self) -> None:
        print("=" * 15 + f"[{type(self).__name__}]" + "=" * 15)
        try:
            sample = self.__getitem__(0)
            fmt_keys = sample.keys()
            A, B = self.train_mask, self.test_mask
            ratio = len(A) / (len(A) + len(B))
            fmt_ratio = lambda ratio: f"{round(ratio, 3)}|{round(1 - ratio, 3)}"

            print(f"Dataset name: '{self.name}'")
            print(f"Configured data keys: {list(fmt_keys)}")
            print(f"\t Input type (pvi): '{self.input_mode}' (shape={tuple(sample[PviChannelGroup.HP.value].shape)})")
            print(f"\t Output type (bp): '{self.output_mode}' (shape={tuple(sample['bp'].shape)})")
            print(f"\t Active mask: '{self.mask_key}' (total {len(self.active_mask):,} clean sequences)")
            print(f"\t Partition: (Train|Test)=({len(A):,}|{len(B):,}) sequences, ratio=({fmt_ratio(ratio)})")

        except BaseException:
            print("Cannot display info. Something is amiss!")
        print("=" * 15 + f"[{type(self).__name__}]" + "=" * 15)

    def _print_configs(self) -> None:
        print("Dataset configuration:")
        print(f"\t Name: '{self.name}'")
        print(f"\t Input type: '{self.input_mode}'")
        print(f"\t Output type: '{self.output_mode}'")
        print(f"\t Mask key: '{self.mask_key}'")

    def set_partition(self,
                      test_size: float = 0.1,
                      shuffle: bool = True,
                      **kwargs) -> None:

        if not isinstance (test_size, (int, float)):
            raise TypeError(f"Invalid type '{type(test_size)}' for 'test_size'! Expect a float or int.")

        params = {'test_size': test_size,
                  'shuffle': shuffle,
                  **kwargs}

        self._split_params = params

    @abstractmethod
    def get_partition(self) -> dict[str, Subset | Dataset]:
        pass

    def _get_subsets_from_split(self,
                                active_mask: list[tuple[int,...]]=None,
                                train_mask: list[tuple[int,...]]=None,
                                test_mask: list[tuple[int,...]]=None,
                                ) -> dict[str, Subset]:

        active_mask = self.active_mask if active_mask is None else active_mask
        train_mask = self.train_mask if train_mask is None else train_mask
        test_mask = self.test_mask if test_mask is None else test_mask

        # use dictionary for speed instead of list comprehension
        mask_lookup = {tp: idx for idx, tp in enumerate(active_mask)}
        train_idx = [mask_lookup[tp] for tp in train_mask]
        test_idx = [mask_lookup[tp] for tp in test_mask]

        # already shuffled when splitting train vs test samples.
        # This reorder helps with indexing, especially for lazy dataset
        # We don't have to shuffle again HERE. If more shuffling is required,
        # it will be handled by the DataLoader (when making the batches)
        train_idx.sort()
        test_idx.sort()

        # THE USE OF SUBSET CLASS AFFECTS HOW CUSTOM BATCH SAMPLER PARSES THE INDICES
        # IF WE CHANGE THIS,BE SURE TO ALSO INSPECT PVIBATCHSAMPLER FOR CONSISTENCY
        subsets = {'train': Subset(self, train_idx),
                   'test': Subset(self, test_idx)}

        for k in subsets.keys():
            for kw in self._VALID_CONFIGS_KEYS:
                setattr(subsets[k], kw, getattr(self,kw))

        return subsets

    def get_params_shallow(self) -> dict:
        return self._get_params_default()

    def _get_params_default(self) -> dict:

        counts = {'num_periods': self.num_periods,
                  'num_sequences': len(self.active_mask),
                  'num_train': len(self.train_mask),
                  'num_test': len(self.test_mask)
                  }

        split_state_meta = {k:v for k,v in self.state_dict().items()
                            if k not in self._VALID_MASK_ATTRIBUTES}

        dict_out = {'class': type(self).__name__,
                    'name': self.name,
                    'constituents': [ds.name for ds in self.raws],
                    'configs': self.configs,
                    'counts': counts,
                    **split_state_meta,
                    'split_params': self._split_params,
                    'batch_params':self._loader_params,
                    'shapes': self.shapes,
                    'raw_stats': self.get_raw_statistics(),
                    }

        return dict_out

    def get_raw_statistics(self) -> dict[str, float | int]:
        raws = [self.raws] if isinstance(self.raws, PviRawDataset) else self.raws

        d = {'sqi': self.sqi,
             'num_periods': sum([dsr.num_periods for dsr in raws]),
             }
        for mk in SequenceMask.keys():
            kw = mk.replace('mask','num_seq')
            d[kw] = 0
            for dsr in raws:
                d[kw] += len(dsr.masks[mk]) if mk in dsr.masks else 0

        return d

    def state_dict(self) -> dict:
        state = {kw: getattr(self, kw)
                 for kw in self._VALID_MASK_ATTRIBUTES + ["offset_period"]} # only relevant attributes
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        for kw, value in state_dict.items():
            setattr(self, kw, value)

        self.subsets = self._get_subsets_from_split()

    def set_dataloaders(self,
                        batch_size: int=32,
                        shuffle: bool=True,
                        **kwargs) -> None:

        params = {'batch_size': batch_size,
                  'shuffle': shuffle,
                  **kwargs}

        self._loader_params = params

    def get_dataloaders(self) -> dict[str,DataLoader]:

        if not self.subsets:
            raise AttributeError("Cannot get DataLoaders! Train and test subsets not available.")

        if not self._loader_params:
            raise AttributeError("Cannot get DataLoaders! Loader params not set.")

        train_params = self._loader_params

        test_params = copy.copy(train_params)
        test_params.update({'shuffle': False})

        self.loaders = {"train": DataLoader(self.subsets['train'], **train_params),
                        "test": DataLoader(self.subsets['test'], **test_params)}

        if self._verbose:
            print(f"{self._alias}: Finish making DataLoaders.")

        if self._loader_params.get('shuffle'):
            print(f"{self._alias} (WARNING): By default, test loader was NOT shuffled!")

        return self.loaders

    def remove_offset(self) -> 'PviConfiguredDataset':
        new = self.add_offset(-self.offset_period)
        return new

    def add_offset(self, offset: int=0) -> 'PviConfiguredDataset':
        # Used when stacking multiple datasets
        if offset==0:
            return self

        new = copy.copy(self)  # Shallow copy for metadata
        for kw in self._VALID_MASK_ATTRIBUTES: # active_mask, train_mask, test_mask
            if hasattr(new, kw) and (getattr(new, kw) is not None):
                current_mask = getattr(self, kw)
                new_mask = [misc.offset_integers(m, offset) for m in current_mask]
                setattr(new, kw, new_mask)
        new.offset_period = self.offset_period + offset
        print(f"{self._alias}: Mask offset {self.offset_period:,} -> {new.offset_period:,}.")

        return new

    def to(self, device: str|torch.device=None, dtype: torch.dtype=None, **kwargs) -> 'PviConfiguredDataset':
        self.data = h5io.transfer(self.data, device=device, dtype=dtype, **kwargs)
        return self

    @property
    def sqi(self) -> float: # signal quality index with wilson correction
        raws = [self.raws] if isinstance(self.raws, PviRawDataset) else self.raws
        num_raw = sum([dsr.num_periods for dsr in raws])
        key = SequenceMask.MASK01.value
        num_clean = sum([len(dsr.masks[key]) for dsr in raws])

        p = num_clean/num_raw

        conf = 0.95 # confidence interval
        z = stats.norm.ppf(1 - (1 - conf)/2).item()

        t = z**2/num_raw

        margin = float(math.sqrt(p*(1 - p)*t + t**2/4)/(1 + t))
        center = float((p + t / 2) / (1 + t))
        low = center - margin
        high = center + margin

        return low

    @property
    def unit_mask(self) -> tuple[int,...]:
        return misc.shift_integers_to_zero(self.active_mask[0])

    @property
    def num_frames(self) -> int:
        return self.num_periods * self.period_length

    @property
    def dtype(self) -> torch.dtype:
        sample = self.__getitem__(0)
        return next(iter(sample.values())).dtype

    @property
    def device(self) -> torch.device:
        sample = self.__getitem__(0)
        return next(iter(sample.values())).device

    @property
    def shapes(self) -> dict:
        sample = self.__getitem__(0)
        shapes = {'input': tuple(sample[PviChannelGroup.HP.value].shape),
                  'output': tuple(sample['bp'].shape),
                  'stats': tuple(sample['stats'].shape)}

        return shapes

    @property
    def configs(self) -> dict:
        if (not self.input_mode) or (not self.output_mode) or (not self.mask_key):
            raise ValueError(f"Cannot access configs for '{self._alias}'! Missing attributes.")

        configs = {kw: getattr(self, kw).value for kw in self._VALID_CONFIGS_KEYS}
        return configs

    def _validate_build(self) -> None:
        if self._verbose:
            print(f"{self._alias}: Validating build...")

        self.__len__()
        self.__getitem__(0)
        self.__getitem__(-1)

        h5io.validate_format_bp(self.data['bp'],
                                self.output_mode,
                                self.num_periods,
                                self.period_length)

        h5io.validate_format_pvi(self.data[PviChannelGroup.HP.value],
                                 self.input_mode,
                                 self.num_frames)

        h5io.validate_format_stats(self.data['stats'],
                                   num_stats=2,
                                   num_periods=self.num_periods)

        h5io.validate_data_contiguous(self.data)

        if self._verbose:
            print(f"\t No exception found!")

    @staticmethod
    def _compute_group_offset(ds_list: list['PviRawDataset']) -> list[int]:
        cs = [0]  # cumulative sum
        for ds in ds_list:
            cs.append(cs[-1] + ds.num_periods)
        return cs[:-1]

    # def _survey_datasets(self,
    #                      raws: list[PviRawDataset]=None,
    #                      mask_key: SequenceMask=None) -> 'DatasetMappings':
    #     raws = self.raws if raws is None else raws
    #     mask_key = self.mask_key if mask_key is None else mask_key
    #
    #     total = len(raws)
    #
    #     # Mappings (sample-level, same length as total samples)
    #     mapping_files = []
    #     mapping_masks_global = []
    #     mapping_masks_local = []
    #     mapping_idx_global = []
    #     mapping_idx_local = []
    #
    #     # Groupings (file-level, same length as number of files)
    #     grouping_files = []
    #     grouping_masks_global = []
    #     grouping_masks_local = []
    #     grouping_idx_global = []
    #     grouping_idx_local = []
    #     grouping_bounds = []
    #
    #     offset_periods = 0
    #     offset_sequences = 0
    #
    #     for _, ds_raw in enumerate(raws):
    #         masks_local = ds_raw.masks[mask_key.value]
    #         masks_global = [misc.offset_integers(m, offset_periods) for m in masks_local]
    #         idx_local = list(range(len(masks_local)))
    #         idx_global = [offset_sequences + i for i in idx_local]
    #
    #         grouping_files.append(ds_raw)
    #         grouping_bounds.append((offset_periods, offset_periods + ds_raw.num_periods))
    #         grouping_masks_local.append(masks_local)
    #         grouping_masks_global.append(masks_global)
    #         grouping_idx_local.append(idx_local)
    #         grouping_idx_global.append(idx_global)
    #
    #         mapping_files.extend([ds_raw]*len(masks_local))
    #         mapping_masks_global.extend(masks_global)
    #         mapping_masks_local.extend(masks_local)
    #         mapping_idx_global.extend(idx_global)
    #         mapping_idx_local.extend(idx_local)
    #
    #         offset_periods += ds_raw.num_periods
    #         offset_sequences += len(masks_local)
    #
    #     mappings = self.DatasetMappings(
    #             files=mapping_files,
    #             masks_global=mapping_masks_global,
    #             masks_local=mapping_masks_local,
    #             indices_global=mapping_idx_global,
    #             indices_local=mapping_idx_local,
    #             partition=[] # require partition state
    #     )
    #
    #     groupings = self.DatasetGroupings(
    #             files=grouping_files,
    #             masks_global=grouping_masks_global,
    #             masks_local=grouping_masks_local,
    #             indices_global=grouping_idx_global,
    #             indices_local=grouping_idx_local,
    #             dataset_bounds=grouping_bounds,
    #             partition = [] # require partition state
    #     )
    #
    #     print(f"{self._alias}: Finish mapping {total} local datasets to global masks.")
    #
    #     return mappings, groupings

    def _compute_local_masks(self) -> dict:
        group_offsets = self._compute_group_offset(self.raws)
        group_bounds = [(offset, offset + ds.num_periods) for offset, ds in zip(group_offsets, self.raws)]

        train_set = set(self.train_mask)
        test_set = set(self.test_mask)

        mappings = {'gm_start': [tp[0] for tp in self.active_mask],
                    'gm_end': [tp[-1] for tp in self.active_mask],
                    'source_name': [],
                    'lm_start': [],
                    'lm_end': [],
                    'partition_subset': []}

        for gm in self.active_mask:
            for k, bd in enumerate(group_bounds):
                if not (bd[0] <= gm[0] and gm[-1] <= bd[-1]):
                    continue

                else:
                    lm = misc.offset_integers(gm, -group_offsets[k])

                    mappings['source_name'].append(self.raws[k].name)
                    mappings['lm_start'].append(lm[0])
                    mappings['lm_end'].append(lm[-1])

                    if gm in train_set:
                        mappings['partition_subset'].append('train')
                    elif gm in test_set:
                        mappings['partition_subset'].append('test')
                    else:
                        mappings['partition_subset'].append('none')
                    break

        return mappings

### SUPPORT FUNCTIONS. ORIGINALLY PARTS OF THE CLASS, BUT EXTRACTED FOR EXTERNAL USE
def ensemble_distance(D1: 'PviConfiguredDataset',
                      D2: 'PviConfiguredDataset',
                      input_mode: str|InputMode,
                      ) -> tuple[float, float]:

    def _reduce_dims(sample: dict[str, torch.Tensor],
                     input_mode: str|InputMode):
        pvi_batch = []
        for kw in PviChannelGroup.keys():
            tensor = sample[kw].unsqueeze(0)  # shape: (1, C, H, W, T)
            tensor = tensor.transpose(1, -1)  # shape: (1, T, H, W, C)
            if InputMode(input_mode) == InputMode.IMPEDANCE:
                C = tensor.shape[-1]  # num_channels
                RT = tensor[..., :C // 2].mean(dim=-1)  # resistance
                XT = tensor[..., C // 2:].mean(dim=-1)  # reactance
                tensor = torch.concat((RT, XT), dim=-1)  # shape: (1, 2*T)
            else:
                tensor = tensor.flatten(start_dim=2)  # shape: (1, T, C*H*W)
                tensor = tensor.nanmean(dim=-1)  # shape: (1, T)
            pvi_batch.append(tensor.contiguous())

        pvi_batch = torch.concat(pvi_batch, dim=-1)  # shape: (1, (2*)2*T)
        bp_batch = sample['bp'].unsqueeze(0)  # shape: (1, 50)

        return pvi_batch, bp_batch

    if len(D1) > len(D2):
        D1, D2 = D2, D1

    if len(D1) == 0 or len(D2) == 0:
        raise AttributeError("Cannot get ensemble distance! Train or test subset is empty!")

    # allocate D1 (smaller dataset)
    x, y = _reduce_dims(D1[0], input_mode=input_mode)
    device = x.device
    dtype = x.dtype

    X1 = torch.zeros(len(D1), x.shape[-1], device=device, dtype=dtype)
    Y1 = torch.zeros(len(D1), y.shape[-1], device=device, dtype=dtype)

    # allocate cost matrices
    CX = torch.zeros(len(D1), len(D2), device=device, dtype=dtype)
    CY = torch.zeros(len(D1), len(D2), device=device, dtype=dtype)

    pbar = tqdm(iterable=D1, desc="\t Stacking dataset", unit='batch', bar_format=dfmt.tqdm)
    for row, sample in enumerate(pbar):
        x, y = _reduce_dims(sample, input_mode=input_mode)
        X1[row, ...] = x.squeeze()
        Y1[row, ...] = y.squeeze()

    pbar = tqdm(iterable=D2, desc="\t Assembling cost", unit='batch', bar_format=dfmt.tqdm)
    for col, sample in enumerate(pbar):
        X2, Y2 = _reduce_dims(sample, input_mode=input_mode)
        tmpX = ot.dist(X1, X2, metric='euclidean')
        tmpY = ot.dist(Y1, Y2, metric='euclidean')

        CX[..., col] = tmpX.squeeze()
        CY[..., col] = tmpY.squeeze()

    print("Solving linear program for input (w1_domain)...")
    w1X = ot.solve(CX).value

    print("Solving linear program for output (w1_label)...")
    w1Y = ot.solve(CY).value

    del CX, CY
    gc.collect()

    return float(w1X), float(w1Y)