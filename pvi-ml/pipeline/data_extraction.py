from src.packages import *

from src.utils.primitives import *
from src.utils import miscellaneous as misc
from src.utils import h5io
from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory

from contextlib import contextmanager
from scipy import stats as stats
import math

class PviRawDataset:
    def __init__(self,
                 ds_file: PviDataFile,
                 persistent: bool=False,
                 verbose: bool=True) -> None:

        self.path: Path = ds_file.path
        self.name: str = ds_file.name
        self._alias = f"{self.name} (Raw)" if self.name else type(self).__name__

        # attributes describing the tensors
        self.data: dict = {}

        self.persistent: bool = persistent
        self.handle = None

        # self.build_info = None
        self.build_info = self.extract_build_info()
        self.meta = self.extract_metadata()
        self.masks = self.extract_masks()
        self._verbose = verbose

        print(f"{self._alias}: Ready. (persistent={self.persistent})")
        # self._print_build_info()

        # InputMode is argument passed to ConfiguredDataset, can have variation and synonyms, meant to be easily understood
        # PviMode must match the name of the subgroup in the hdf5 data. There are no variants
        self._keyword_map = {InputMode.IMAGE: PviSignalGroup.IMAGE.value,
                             InputMode.SIGNAL: PviSignalGroup.SIGNAL.value,
                             InputMode.RESISTANCE: PviSignalGroup.RESISTANCE.value,
                             InputMode.REACTANCE: PviSignalGroup.REACTANCE.value,
                             InputMode.IMPEDANCE: (PviSignalGroup.RESISTANCE.value, PviSignalGroup.REACTANCE.value)}

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        else:
            return self.path.resolve() == other.path.resolve()

    def __hash__(self) -> int:
         # to be used as dictionary keys
        return hash(self.path.resolve())

    @contextmanager
    def _get_handle(self,
                    mode:str='r'):
        if self.persistent:
            if not self.handle:
                self.handle = h5py.File(self.path, mode)
            yield self.handle

        else:
            h5f = h5py.File(self.path, mode)
            try:
                yield h5f
            finally:
                h5f.close()

    @property
    def subject(self) -> str:
        return self.meta['subject']

    @property
    def session(self) -> str:
        return self.meta['session']

    @property
    def num_periods(self) -> int:
        return self.meta['num_periods']

    @property
    def period_length(self) -> int:
        return self.meta['period_length']

    @property
    def num_frames(self) -> int:
        return self.num_periods * self.period_length

    def _print_build_info(self) -> None:
        if self.build_info is None:
            self.build_info = self.extract_build_info()

        print("Build info:")
        for kw, content in self.build_info.items():
            print(f"\t {kw.capitalize()}: {content}")

    def _print_metadata(self) -> None:
        print("Dataset metadata:")
        print(f"\t {self.num_periods:,} total periods ")
        print(f"\t {self.num_frames:,} total frames ({self.period_length} frames per period)")
        print("Available sequence masks:")
        for k, v in self.masks.items():
            print(f"\t {k}: {len(v):,} clean sequences")

    def _print_tensors_info(self) -> None:
        if not self.data:
            raise RuntimeError("Raw dataset not yet loaded!")

        print("Raw tensor shape:")
        shapes = self.shapes
        for sn in self.data.keys():
            for fn in shapes[sn].keys():
                tmp = '.'.join([sn, fn])
                shape = shapes[sn][fn]
                print(f"\t {tmp}: {shape}")

    def print_info(self) -> None:

        print("="*15 + f"[{self._alias}]" + "="*15)
        try:
            print(f"Dataset name: '{self.name}'")
            print(f"Dataset path: '{self.path}'")
            self._print_build_info()
            self._print_metadata()
            self._print_tensors_info()
        except Exception:
            print("Cannot display info. Something is amiss!")

        print("="*15 + f"[{self._alias}]" + "="*15)

    def unload(self) -> None:
        if self.persistent and self.handle:
            self.handle.close()
            misc.cleanup_attributes(self, attrs='handle')

        misc.cleanup_attributes(self, attrs='data')
        setattr(self, 'data', {})

    def load(self) -> 'PviRawDataset':
        t1 = time.perf_counter()

        if self._verbose:
            print(f"{self._alias}: Loading raw tensors...")

        new = copy.copy(self)  # Shallow copy for metadata
        new.data = new.extract_tensors(idx=None) # extract full tensor
        new._validate_raw_shapes()

        dt = time.perf_counter() - t1
        # print(f"{self._alias}: Finish loading raw dataset '{self.name}'. ({dt:.2f} seconds)")
        return new

    def extract_build_info(self) -> dict[str, str]:
        t1 = time.perf_counter()

        with self._get_handle() as h5f:
            info = h5io.read_group_from_h5(h5f['build'])

        dt = time.perf_counter() - t1
        # print(f"{self._alias}: Finish extracting build info. ({dt:.2f} seconds)")
        return info

    def extract_metadata(self) -> dict:
        t1 = time.perf_counter()

        with self._get_handle() as h5f:
            meta = h5io.read_group_from_h5(h5f['metadata'])

        dt = time.perf_counter() - t1
        # print(f"{self._alias}: Finish extracting metadata. ({dt:.2f} seconds)")

        return meta

    def extract_shapes(self) -> dict:
        t1 = time.perf_counter()

        with self._get_handle() as h5f:
            shapes_from_h5 = h5io.read_group_from_h5(h5f['shapes'])

        shapes = {}
        for sn, subdict in shapes_from_h5.items():
            shapes[sn] = {}
            for fn, content in subdict.items():
                shapes[sn][fn] = tuple(content)

        dt = time.perf_counter() - t1

        # print(f"{self._alias}: Finish extracting data shapes. ({dt:.2f} seconds)")

        return shapes

    def extract_masks(self) -> dict:
        t1 = time.perf_counter()

        with self._get_handle() as h5f:
            h5_masks = h5io.read_group_from_h5(h5f['masks'])

        masks = {kw: [] for kw in SequenceMask.keys()}

        for kw, content in h5_masks.items():
            array = np.asarray(content)
            if np.ndim(array) > 2:
                raise RuntimeError("Improper shape for mask data!")
            else:
                if array.size == 2 and array.ndim == 1: # single mask
                    array = array[np.newaxis, :]
                else:
                    pass

            array[:, 0] = array[:, 0] - 1 # 0-base indexing, while keeping [:, 1] as is for slicing
            masks[kw] = [tuple(x) for x in array.tolist()]

        dt = time.perf_counter() - t1
        # print(f"{self._alias}: Finish extracting data masks. ({dt:.2f} seconds)")

        return masks

    def extract_tensors(self,
                        idx: int|tuple[int,...]=None,
                        input_mode: str|InputMode=None) -> dict:
        """
        Loading tensors from h5 file, and maintaining the original hierarchical structure
        This is NOT the final state of the dataset
        """

        if idx is not None:
            if isinstance(idx, int):
                bounds = (idx, idx+1)
            elif isinstance(idx, tuple):
                bounds = idx
            else:
                raise TypeError("Invalid type for slice/index")
            sl_periods, sl_frames = h5io.compute_tensor_slice(bounds, self.period_length)

        else:
            sl_periods, sl_frames = None, None

        with self._get_handle() as h5f:
            h5data = h5f['data']

            if input_mode is None:  # read all groups
                raw = h5io.read_tensors_from_h5(h5data, idx=sl_frames, dim=-1)

            else:  # selectively read the input tensor groups
                raw = {'bp': h5io.read_tensors_from_h5(h5data['bp'])}

                input_mode = InputMode(input_mode)
                for sn in PviChannelGroup.keys():
                    raw[sn] = {}
                    h5pvi = h5data[sn]

                    if input_mode != InputMode.IMPEDANCE:
                        fn = self._keyword_map[input_mode]
                        raw[sn][fn] = h5io.read_tensors_from_h5(h5pvi[fn], idx=sl_frames, dim=-1)
                    else:
                        fnames = self._keyword_map[input_mode]
                        for fn in fnames:
                            raw[sn][fn] = h5io.read_tensors_from_h5(h5pvi[fn], idx=sl_frames, dim=-1)

            h5stats = h5f['stats'][PviChannelGroup.HP.value]
            stats = h5io.read_tensors_from_h5(h5stats, idx=sl_periods, dim=-1)
            raw['stats'] = stats

        return raw

    def _validate_raw_shapes(self) -> None:
        bp_tensor = self.data['bp']['signal']
        crit1 = bp_tensor.shape[0] == 1
        crit2 = bp_tensor.shape[-1] == self.num_frames

        if not (crit1 and crit2):
            raise RuntimeError("Unexpected tensor ordering!")

    def to(self, device: str|torch.device=None, dtype: torch.dtype=None, **kwargs) -> 'PviRawDataset':
        new = copy.copy(self)  # Shallow copy for metadata
        new.data = h5io.transfer(self.data, device=device, dtype=dtype, **kwargs)
        return new

    @property
    def sqi(self) -> float:  # signal quality index with wilson correction
        # this only concerns the raw dataset and is different than the configured dataset
        num_raw = self.num_periods
        num_clean = len(self.masks['mask01'])

        p = num_clean / num_raw

        conf = 0.95  # confidence interval
        z = stats.norm.ppf(1 - (1 - conf) / 2).item()

        t = z ** 2 / num_raw

        margin = float(math.sqrt(p * (1 - p) * t + t ** 2 / 4) / (1 + t))
        center = float((p + t / 2) / (1 + t))
        low = center - margin

        return low

    @property
    def dtype(self) -> torch.dtype:
        if not self.data:
            raise ValueError(f"Cannot determine dtype for '{self._alias}'! Raw data not available.")
        return self.data['bp']['signal'].dtype

    @property
    def device(self) -> torch.device:
        if not self.data:
            raise ValueError(f"Cannot determine device for '{self._alias}'! Raw data not available.")
        return self.data['bp']['signal'].device

    @property
    def shapes(self) -> dict:
        if self.data:
            shapes = {}
            for sn, content in self.data.items():
                shapes[sn] = {}
                for fn, tensor in content.items():
                    shapes[sn][fn] = tuple(tensor.shape)
        else:
            print(f"{self._alias} (WARNING): Raw dataset NOT available. Default to shapes information provided in h5 group!")
            shapes = self.extract_shapes()

        return shapes

if __name__ == "__main__":
    pm = ProjectPathManager()
    inventory = PviDatasetInventory()

    ds_raw = PviRawDataset(inventory[-45],persistent=True)
    ds_raw.extract_tensors(input_mode="image")
    ds_raw = ds_raw.load().to('cpu')
    ds_raw.print_info()