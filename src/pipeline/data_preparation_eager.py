import time

from src.packages import *

from torch.utils.data import ConcatDataset

from src.utils.primitives import *
from src.utils import miscellaneous as misc
from src.utils import h5io
from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_extraction import PviRawDataset
from src.pipeline.graph_partitioner import GraphBipartitePartitioner as gbp

from src.pipeline._data_preparation import PviConfiguredDataset

class PviSingleDataset(PviConfiguredDataset):
    def __init__(self,
                 raw: PviRawDataset,
                 input_mode: str|InputMode,
                 output_mode: str|OutputMode,
                 mask_key: str|SequenceMask,
                 verbose: bool=True,
                 ) -> None:

        super().__init__(input_mode=input_mode,
                         output_mode=output_mode,
                         mask_key=mask_key,
                         verbose=verbose)

        self._validate_components(raw)

        self._alias = type(self).__name__ # temporary alias
        self.name = raw.name
        self.path = raw.path

        self.period_length = raw.period_length
        self.num_periods = raw.num_periods

        self.active_mask = raw.masks[self.mask_key.value]

        self.offset_period = 0 # modifiable

        self._alias = f"{self.name} (Single)"

        self.raws = raw

        if self._verbose:
            print(f"{self._alias}: Ready.")

    def _validate_components(self, raw) -> None:
        # crit1 = isinstance(raw, PviRawDataset)
        if not isinstance(raw, PviRawDataset):
            raise TypeError(
                f"Invalid components type for class '{type(self)}'! Expect 'PviRawDataset'.")

    def build(self, cleanup: bool=True) -> 'PviSingleDataset':
        if self._verbose:
            print(f"{self._alias}: Creating configured dataset...")
            self._print_configs()

        # Dataset transform pipeline: format → slice → cleanup
        t1 = time.perf_counter()
        # print(f"{self._alias}: Formatting raw data tensors...")
        self.data = h5io.format_raw_tensors(self.raws.data,
                                            input_mode=self.input_mode,
                                            output_mode=self.output_mode,
                                            period_length=self.period_length)

        if self._verbose:
            self.raws.unload()

        self._validate_build()
        dt = time.perf_counter() - t1

        if self._verbose:
            print(f"{self._alias}: Finish configuring data. ({dt:.2f} seconds)")

        return self

    def get_partition(self) -> dict[str, Subset | Dataset]:
        if not self._split_params:
            raise AttributeError("Cannot split dataset! Split params not set.")

        splitter = gbp(intervals=self.active_mask,
                       caller_id=f"{self.name} (Splitter)",
                       **self._split_params,
                       )

        self.train_mask, self.test_mask = splitter.split()

        t1 = time.perf_counter()
        self.subsets = self._get_subsets_from_split()

        dt = time.perf_counter() - t1

        print(f"{self._alias}: Finish collating subsets. ({dt:.2f} seconds)")
        print(f"\t Final partition: (Train|Test) = ({len(self.subsets['train'])}|{len(self.subsets['test'])})")

        return self.subsets

class PviCompositeDataset(PviConfiguredDataset):
    def __init__(self,
                 ds_raws: list[PviRawDataset],
                 input_mode: str | InputMode,
                 output_mode: str | OutputMode,
                 mask_key: str | SequenceMask,
                 name: str='dataset_composite',
                 verbose: bool=True) -> None:

        super().__init__(input_mode=input_mode,
                         output_mode=output_mode,
                         mask_key=mask_key,
                         verbose=verbose)

        self._validate_components(ds_raws)

        self.name = name
        self.path = 'N/A'

        self._alias = f"{self.name} (Composite)" if self.name else type(self).__name__

        self._check_group_unique(ds_raws)
        self._check_group_attributes(ds_raws, "period_length")

        self.period_length = ds_raws[0].period_length
        self.num_periods = sum([dsr.num_periods for dsr in ds_raws])

        self.offset_period = 0 # modifiable

        self.input_mode = InputMode(input_mode)
        self.output_mode = OutputMode(output_mode)
        self.mask_key = SequenceMask(mask_key)

        self.raws: list[PviRawDataset] = ds_raws
        self.singles: list[PviSingleDataset] = []

        if self._verbose:
            print(f"{self._alias}: Ready.")

    def _validate_components(self, ds_raws) -> None:
        crit1 = isinstance(ds_raws, (tuple, list, set)) and all([isinstance(dsr, PviRawDataset) for dsr in ds_raws])
        if not crit1:
            raise TypeError(
                f"Invalid components type for class '{type(self)}'! Expect an iterable of 'PviRawDataset'.")

    def build(self, cleanup: bool=True) -> 'PviCompositeDataset':
        if (not hasattr(self, "raws")) or (not self.raws):
            msg = [f"({type(self).__name__}) Missing 'raws' attribute!",
                   "Either dataset is already composed and cannot be re-composed,",
                   "or dataset was not instantiated properly."]
            msg = ' '.join(msg)
            raise ValueError(msg)

        if self._verbose:
            print(f"{self._alias}: Composing {len(self.raws)} datasets...")

        t1 = time.perf_counter()

        ds_singles = []
        for dsr in self.raws:
            ds = PviSingleDataset(dsr,
                                  input_mode=InputMode(self.input_mode),
                                  output_mode=OutputMode(self.output_mode),
                                  mask_key=SequenceMask(self.mask_key),
                                  verbose=False,
                                  ).build(cleanup=cleanup)
            ds_singles.append(ds)

        # mapping local to global masks
        self._append_masks(ds_singles)

        # self.singles = ds_singles

        # appending tensors (not stacking yet)
        self.data = self._append_data(ds_singles)

        # Remap to contiguous memory
        self.singles = self._remap_data(ds_singles,
                                        self.data,
                                        self.period_length,
                                        cleanup=cleanup)

        if cleanup:
            for dsr in self.raws:
                dsr.unload()
            # self.cleanup("raws")

        self._validate_build()
        dt = time.perf_counter() - t1

        if self._verbose:
            print(f"{self._alias}: Finish composing data. ({dt:.2f} seconds)")

        return self

    def _append_data(self, ds_singles) -> dict:
        """
        The returned data is not a contiguous block on memory.
        As a consequences, the tensors are duplicated!
        We need to remap the data, then clean up.
        """
        t1 = time.perf_counter()

        data = {}
        for kw in (['bp'] + PviChannelGroup.keys()):
            data_tensors = []
            for ds in ds_singles:
                data_tensors.append(ds.data[kw])

            data[kw] = data_tensors

        data['bp'] = torch.vstack(data['bp'])

        for kw in PviChannelGroup.keys():
            data[kw] = torch.cat(data[kw], dim=-1)

        stats_tensors = []
        for ds in ds_singles:
            stats_tensors.append(ds.data['stats'])

        data['stats'] = torch.cat(stats_tensors, dim=-1)

        dt = time.perf_counter() - t1
        # if self._verbose:
        #     print(f"{self._alias}: Finish appending data. ({self.num_periods:,} samples in {dt:.2f} seconds)")

        return data

    def _remap_data(self,
                    ds_singles,
                    data_block: dict,
                    period_length: int,
                    cleanup: bool=True) -> list[dict]:
        """
        Map constituent datasets to slices of combined block to avoid memory duplication
        """

        t1 = time.perf_counter()

        group_offsets = self._compute_group_offset(ds_singles)
        group_lengths = [ds.num_periods for ds in ds_singles]
        group_bounds = [(L0, L0 + dL) for (L0, dL) in zip(group_offsets, group_lengths)]

        group_new = []
        for ds, bounds in zip(ds_singles, group_bounds):
            ds_new = copy.copy(ds)

            sl_period, sl_point = h5io.compute_tensor_slice(bounds, period_length)

            # this block is very similar to slicing sequences (but NOT THE SAME)
            data = {'bp': data_block['bp'][sl_period],
                    'stats': data_block['stats'][..., sl_period]}

            data.update({kw: data_block[kw][...,sl_point] for kw in PviChannelGroup.keys()})

            ds_new.data = data
            group_new.append(ds_new)

            if cleanup:
                ds.cleanup("data")

        dt = time.perf_counter() - t1
        # if self._verbose:
        #     print(f"{self._alias}: Finish remapping. ({dt:.2f} seconds)")

        return group_new

    def _append_masks(self, ds_singles) -> None:

        t1 = time.perf_counter()

        group_offsets = self._compute_group_offset(ds_singles)

        masks = {kw: [] for kw in self._VALID_MASK_ATTRIBUTES} # active_mask, train_mask, test_mask
        for ds, offset in zip(ds_singles, group_offsets):
            new = ds.remove_offset().add_offset(offset)
            for kw in self._VALID_MASK_ATTRIBUTES:
                if hasattr(new, kw) and (getattr(new, kw) is not None):
                    masks[kw].extend(getattr(new, kw))

        dt = time.perf_counter() - t1

        # if self._verbose:
        #     print(f"{self._alias}: Finish appending data masks. ({dt:.2f} seconds)")

        for kw in self._VALID_MASK_ATTRIBUTES:
            setattr(self, kw, masks[kw])

    def get_partition(self) -> dict[str, Subset|Dataset]:

        t1 = time.perf_counter()
        new_group = []
        for ds in self.singles:
            ds = ds.remove_offset()
            ds.set_partition(**self._split_params)
            _ = ds.get_partition() # delegate to each constituent datasets
            new_group.append(ds)

        t2 = time.perf_counter()
        if self._verbose:
            print(f"{self._alias}: Finish splitting all constituent datasets. ({(t2 - t1):.2f} seconds)")
        # assign global masks

        self._append_masks(new_group)
        self.singles = new_group

        self.subsets = self._get_subsets_from_split()

        t3 = time.perf_counter()

        if self._verbose:
            print(f"{self._alias}: Finish collating subsets. ({(t3 - t2):.2f} seconds)")
            print(f"\t Final partition: (Train|Test) = ({len(self.subsets['train']):,}|{len(self.subsets['test']):,})")

        return self.subsets

    def _check_group_attributes(self, ds_singles, *args) -> None:
        # make sure all params are homogeneous
        if not args:
            args = ["period_length"]
        elif args == "configs":
            args = self._VALID_CONFIGS_KEYS
        else:
            pass

        for kw in args:
            attributes = set() # is this a reserved keyword?
            for ds in ds_singles:
                attributes.add(str(getattr(ds,kw)))

            if len(attributes) > 1:
                phrase = repr(list(attributes))
                if kw in self._VALID_CONFIGS_KEYS:
                    msg = f"Expect all datasets in group to have the same configurations. Got {phrase} for '{kw}'."
                else:
                    msg = f"Expect all datasets in group to have the same {kw}. Got {phrase}."

                raise RuntimeError(msg)

    @staticmethod
    def _check_group_unique(ds_singles) -> None:
        # make sure there is no duplication
        names = set()
        for ds in ds_singles:
            names.add(ds.name)

        if len(names) < len(ds_singles):
            raise RuntimeError(f"Found duplicated datasets in group.")

#### MAIN TESTING FUNCTIONS ####
def test_single() -> None:
    inventory = PviDatasetInventory()

    ds_raw = PviRawDataset(ds_file=inventory[0])
    ds_raw = ds_raw.load().to('cuda')

    ds = PviSingleDataset(ds_raw,
                          input_mode='image',
                          output_mode='waveform',
                          mask_key='mask10').build(cleanup=True).to('cpu')

    ds.set_partition()
    ds.set_dataloaders()

    subsets = ds.get_partition()
    loaders = ds.get_dataloaders()

    batch = next(iter(loaders['train']))

def test_composite() -> None:

    ds_fallback = Path(r'C:\localdata_SRL4\pvi_datasets_clone\datasets')
    inventory = PviDatasetInventory(ds_root=ds_fallback, branch='main')

    raws = []
    for file in inventory[:5]:
        dsr = PviRawDataset(ds_file=file).load()
        raws.append(dsr)

    dsp = PviCompositeDataset(raws,
                              input_mode='img',
                              output_mode='waveform',
                              mask_key='mask05',
                              verbose=False,
                              )

    dsp = dsp.build(cleanup=True)

    dsp.set_partition()
    dsp.set_dataloaders()

    _ = dsp.get_partition()
    _ = dsp.get_dataloaders()

    D1 = dsp.subsets['train']
    D2 = dsp.subsets['test']
    from src.pipeline._data_preparation import ensemble_distance
    _ = ensemble_distance(D1, D2, dsp.input_mode)

    # batch = next(iter(dsp.loaders['train']))

    mappings = dsp._compute_local_masks()

if __name__ == "__main__":
    print("tmp...")

    # test_single()
    test_composite()