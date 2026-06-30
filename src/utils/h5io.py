from src.packages import *
from src.utils.primitives import *
import numbers

"""
Some useful functions io and dataset transformation
"""

def read_tensors_from_h5(node: h5py.Group|h5py.Dataset,
                         idx: [int|slice|None]=None,
                         dim: [int|None] = None,
                         ) -> dict|torch.Tensor:

    def _read_dataset(node: h5py.Dataset, idx:[int|slice|None]=None, dim:[int|None]=None) -> torch.Tensor:
        if idx is None:
            array = node[()]  # native numpy array
        else:
            if not isinstance(idx, (int, slice)):
                raise TypeError("Invalid type for slice/index")
            if dim is None:
                raise ValueError("Dimension must be specified for slicing/indexing")
            if dim == 0:
                array = node[idx, ...]
            elif dim == -1:
                array = node[..., idx]
            else:
                raise NotImplementedError("Only accept indexing for dimensions 0 and -1")

        array = array.astype(np.float32)
        return torch.from_numpy(array)
        # return torch.from_numpy(array).contiguous()

    if isinstance(node, h5py.Dataset):
        return _read_dataset(node, idx, dim)

    dict_out = {}
    for kw, content in node.items():
        if isinstance(content, h5py.Group):
            dict_out[kw] = read_tensors_from_h5(node=content, idx=idx, dim=dim)
        elif isinstance(content, h5py.Dataset):
            dict_out[kw] = _read_dataset(node=content, idx=idx, dim=dim)
        else:
            raise NotImplementedError(f"Cannot parse node '{content.name}' with type '{type(content)}'")

    return dict_out

def read_group_from_h5(node: h5py.Group) -> dict:
    """
    recursively read a nested group.
    The leaf dataset is assumed to contain either a byte data, or numeric,
    or a homogeneous array of either bytes or numeric
    """

    def _whole_to_int(data: numbers.Number|list[numbers.Number]
                      ) -> numbers.Number|list[numbers.Number]:
        if isinstance(data, (list, tuple)):
            return [_whole_to_int(x) for x in data]
        else:
            if isinstance(data, float) and data.is_integer():
                return int(data)
            else:
                return data

    dict_out = {}
    for kw, content in node.items():
        if isinstance(content, h5py.Group):
            dict_out[kw] = read_group_from_h5(node=content)
        else:
            content = content[()].squeeze()
            if np.ndim(content) == 0:
                content = content.item()
                if isinstance(content, bytes):
                    dict_out[kw] = content.decode()
                else:
                    dict_out[kw] = _whole_to_int(content)
            else:
                content = content.tolist()
                if content and isinstance(content[0], bytes):
                    dict_out[kw] = [x.decode() for x in content]
                else:
                    dict_out[kw] = _whole_to_int(content)
    return dict_out

def write_tensors(dict_in: dict,
                  group: h5py.Group = '/',
                  compression: str = "gzip",
                  chunks: bool = True,
                  ) -> None:
    """
    Inputs at leaf dictionaries (final level) must be np arrays or
    any type of numerics that can be converted to np array,
    such as list of float/int, or nested lists of ints, etc.
    """
    for kw, content in dict_in.items():
        if not content:
            continue

        if isinstance(content, dict):
            print(f"\t {group.name}/{kw}...")
            subgroup = group.create_group(name=kw)
            write_tensors(dict_in=content,
                          group=subgroup,
                          compression=compression,
                          chunks=chunks)
        else:
            if  isinstance(content, (list, tuple)):
                content = np.array(content)
            elif isinstance(content, torch.Tensor):
                content = content.numpy()
            elif isinstance(content, np.ndarray) or np.isscalar(content):
                pass
            else:
                raise TypeError(f"Warning: Unsupported datatype {type(content)} for {kw}")

            print(f"\t {group.name}/{kw} ({content.dtype} : {content.shape})")
            group.create_dataset(name=kw,
                                 data=content,
                                 dtype=content.dtype,
                                 compression=compression,
                                 chunks=chunks)

def write_nested_group(dict_in: dict, group: h5py.Group = '/') -> None:

    def _is_all_similar_type(obj, types: tuple[type,...]):
        crit1 = isinstance(obj, types)
        crit2 = isinstance(obj, (list, tuple)) and all(isinstance(_, types) for _ in obj)
        return crit1 or crit2

    for kw, content in dict_in.items():
        if not content:
            continue
        if isinstance(content, dict):
            print(f"\t {group.name}/{kw}...")
            subgroup = group.create_group(name=kw)
            write_nested_group(dict_in=content, group=subgroup)
        else:
            if _is_all_similar_type(content, (str, bytes)):
                dtype = h5py.string_dtype(encoding='utf-8')
            elif _is_all_similar_type(content, (int, list, tuple)):
                content = np.asarray(content)
                dtype = content.dtype
            else:
                raise TypeError(f"Unsupported datatype {type(content)} for {kw}")

            print(f"\t {group.name}/{kw} ({dtype})")
            group.create_dataset(name=kw, data=content, dtype=dtype)

def preallocate_tensors(shapes_dict: dict,
                        tensor_group: h5py.Group,
                        tensor_dtype: np.dtype = np.float32,
                        compression: str = "gzip",
                        chunks: bool = True,
                        ) -> None:

    for kw, content in shapes_dict.items():
        if tensor_group.name == '/':
            raise RuntimeError("Cannot pre-allocate tensors to root group!")

        if isinstance(content, dict):
            print(f"\t {tensor_group.name}/{kw}...")
            subgroup = tensor_group.create_group(name=kw)
            preallocate_tensors(shapes_dict=content,
                                tensor_group=subgroup,
                                tensor_dtype=tensor_dtype,
                                compression=compression,
                                chunks=chunks)
        else:
            print(f"\t {tensor_group.name}/{kw} ({tensor_dtype} : {content})")
            tensor_group.create_dataset(name=kw,
                                        shape=content,
                                        dtype=tensor_dtype,
                                        compression=compression,
                                        chunks=True)

def format_raw_tensors(raw_data: dict[str,dict[str,torch.Tensor]],
                       output_mode: OutputMode,
                       input_mode: InputMode,
                       period_length) -> dict[str, torch.Tensor]:
    # convert to Enum
    input_mode = InputMode(input_mode)
    output_mode = OutputMode(output_mode)

    t1 = time.perf_counter()
    data = {'bp': format_tensors_bp(bp_data=raw_data['bp'],
                                    output_mode=output_mode,
                                    period_length=period_length),}

    for kw in [_.value for _ in PviChannelGroup]:
        data[kw] = format_tensors_pvi(pvi_data=raw_data[kw],
                                      input_mode=input_mode)

    data['stats'] = format_tensors_stats(stats_dict=raw_data['stats'])

    return data

def format_tensors_bp(bp_data: dict[str, torch.Tensor],
                      output_mode: OutputMode,
                      period_length: int,
                      ) -> torch.Tensor:

    bp = bp_data['signal'] # shape: (N, T) or (1, T*N)
    bp = bp.reshape(-1, period_length) # shape: (N, T)

    if output_mode == OutputMode.WAVEFORM:
        bp_tensor = bp.squeeze()
    else:
        sbp = bp.max(dim=-1)[0].unsqueeze(dim=-1)  # shape: (N, 1)
        dbp = bp.min(dim=-1)[0].unsqueeze(dim=-1)  # shape: (N, 1)

        if output_mode == OutputMode.SYSTOLIC:
            bp_tensor = sbp # shape: (N, 1)
        elif output_mode == OutputMode.DIASTOLIC:
            bp_tensor = dbp # shape: (N, 1)
        else:  # 'minmax' case
            bp_tensor = torch.hstack([dbp, sbp])  # shape: (N, 2)

    return bp_tensor

def format_tensors_pvi(pvi_data: dict[str,torch.Tensor],
                       input_mode: InputMode) -> torch.Tensor:

    if input_mode == InputMode.SIGNAL:
        pvi_tensor = pvi_data[PviSignalGroup.SIGNAL.value]  # shape: (1, T*N)

    elif input_mode == InputMode.IMAGE:
        tensor = pvi_data[PviSignalGroup.IMAGE.value]  # shape: (H, W, T*N)
        pvi_tensor = tensor.unsqueeze(dim=0)  # shape: (1, H, W, N*T) (add extra dim for 'channel')

    else:
        R = pvi_data[PviSignalGroup.REACTANCE.value] # shape: (C, N*T)
        X = pvi_data[PviSignalGroup.RESISTANCE.value] # shape: (C, N*T)

        if input_mode == InputMode.RESISTANCE:
            pvi_tensor = R
        elif input_mode == InputMode.REACTANCE:
            pvi_tensor = X
        else:
            pvi_tensor = torch.vstack([R, X])  # shape: (Cr + Cx,N*T)

    return pvi_tensor

def format_tensors_stats(stats_dict: dict) -> torch.Tensor:
    stats = []
    for kw in ['duration', 'tMax']: # must be in this order
        stats.append(stats_dict[kw].squeeze())

    stats_tensor = torch.vstack(stats)

    return stats_tensor

def slice_sequences(data: dict[str,torch.Tensor],
                    bounds: tuple[int,...],
                    period_length: int) -> dict[str, torch.Tensor]:
    sl_period, sl_point = compute_tensor_slice(bounds, period_length)

    sequence = {'bp': data['bp'][sl_period][-1],
                'stats': data['stats'][..., sl_period]}

    sequence.update({kw: data[kw][..., sl_point] for kw in PviChannelGroup.keys()})

    return sequence

def validate_format_bp(bp_data: dict|torch.Tensor,
                       output_mode: OutputMode,
                       num_periods: int,
                       period_length: int) -> None:

    def _validate_bp_tensor_format(bp_tensor, mode, nperiods, plength):
        expected_first_dim = nperiods
        if mode == OutputMode.WAVEFORM:
            expected_last_dim = plength
        elif mode == OutputMode.FIDUCIALS:
            expected_last_dim = 2
        else:
            expected_last_dim = 1
        criteria = [expected_first_dim == bp_tensor.shape[0],
                    expected_last_dim == bp_tensor.shape[-1]]
        if not all(criteria):
            raise RuntimeError(f"Invalid BP tensor shape: {tuple(bp_tensor.shape)}")

    if isinstance(bp_data, torch.Tensor):
        _validate_bp_tensor_format(bp_data, output_mode, num_periods, period_length)
        return

    if not isinstance(bp_data, dict):
        raise TypeError(f"Invalid type {type(bp_data)}. Expected bp_data to be a nested dict of torch.Tensors.")

    for kw, content in bp_data.items():

        if isinstance(content, dict):
            validate_format_bp(bp_data, output_mode, num_periods, period_length)

        elif isinstance(content, torch.Tensor):
            _validate_bp_tensor_format(content, output_mode, num_periods, period_length)
        else:
            raise TypeError(f"Invalid type {type(content)}. Expected bp_data to be a nested dict of torch.Tensors.")

def validate_format_pvi(pvi_data: dict|torch.Tensor,
                        input_mode: InputMode,
                        num_frames: int) -> None:

    def _validate_pvi_tensor_format(pvi_tensor, mode, nframes):
        expected_last_dim = nframes
        if mode == InputMode.SIGNAL:
            expected_ndim = 2
            expected_first_dim = 1
        elif mode == InputMode.IMAGE:
            expected_ndim = 4
            expected_first_dim = 1
        elif mode == InputMode.IMPEDANCE:
            expected_ndim = 2
            expected_first_dim = None # not specified yet
        else:
            expected_ndim = 2
            expected_first_dim = None  # not specified yet

        criteria = [expected_ndim == pvi_tensor.ndim,
                    expected_last_dim == pvi_tensor.shape[-1]]

        if expected_first_dim is not None:
            criteria.append(expected_first_dim == pvi_tensor.shape[0])

        if not all(criteria):
            raise RuntimeError(f"Invalid {kw} tensor shape: {tuple(pvi_tensor.shape)}")

    if isinstance(pvi_data, torch.Tensor):
        _validate_pvi_tensor_format(pvi_data, input_mode, num_frames)
        return

    if not isinstance(pvi_data, dict):
        raise TypeError(f"Invalid type {type(pvi_data)}. Expected pvi_data to be a nested dict of torch.Tensors.")

    for kw, content in pvi_data.items():


        if isinstance(content, dict):
            validate_format_pvi(pvi_data, input_mode, num_frames)

        elif isinstance(content, torch.Tensor):
            _validate_pvi_tensor_format(content, input_mode, num_frames)

        else:
            raise TypeError(f"Invalid type {type(content)}. Expected pvi_data to be a nested dict of torch.Tensors.")


def validate_format_stats(stats_data: dict|torch.Tensor,
                          num_stats: int,
                          num_periods: int) -> None:
    def _validate_stats_tensor_format(stats_tensor, nstats, nperiods):
        expected_shape = (nstats, nperiods)
        if expected_shape != tuple(stats_tensor.shape):
            raise RuntimeError(f"Invalid stats tensor shape: {tuple(stats_tensor.shape)}")

    if isinstance(stats_data, torch.Tensor):
        _validate_stats_tensor_format(stats_data, num_stats, num_periods)
        return

    if not isinstance(stats_data, dict):
        raise TypeError(f"Invalid type {type(stats_data)}. Expected stats_data to be a nested dict of torch.Tensors.")

    for kw, content in stats_data.items():


        if isinstance(content, dict):
            validate_format_stats(content, num_stats, num_periods)
        elif isinstance(content, torch.Tensor):
            _validate_stats_tensor_format(content, num_stats, num_periods)
        else:
            raise TypeError(f"Invalid type {type(content)}. Expected stats_data to be a nested dict of torch.Tensors.")

def reverse_tensor_dims(tensor: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(tensor.ndim))
    new_dims = dims[::-1]
    new_tensor = torch.permute(tensor, new_dims)

    return new_tensor

def compute_tensor_slice(period_bounds: tuple[int,...],
                         period_length: int) -> tuple[slice,...]:
    sl_period = slice(*period_bounds)
    sl_point = slice(*tuple(_*period_length for _ in period_bounds))

    return sl_period, sl_point

def check_tensors_mismatch(tensor1, tensor2) -> bool:
    tensor1 = tensor1.nan_to_num(nan=0.0)
    tensor2 = tensor2.nan_to_num(nan=0.0)

    return torch.allclose(tensor1, tensor2)

def check_tensors_duplicate(tensor1, tensor2) -> bool:
    ptr1 = tensor1.data_ptr()
    ptr2 = tensor2.data_ptr()

    return ptr1 == ptr2

def validate_data_contiguous(data: dict|torch.Tensor) -> None:

    def _is_contiguous(tensor):
        if not tensor.is_contiguous():
            raise MemoryError(f"Tensors are NOT contiguous!")

    if isinstance(data, torch.Tensor):
        _is_contiguous(data)
        return

    if not isinstance(data, dict):
        raise TypeError("Invalid type for contiguous memory check. Expect torch.Tensor.")

    for kw, content in data.items():


        if isinstance(content, dict):
            validate_data_contiguous(content)
        elif isinstance(content, torch.Tensor):
            _is_contiguous(content)
        else:
            raise TypeError("Invalid type for contiguous memory check. Expect torch.Tensor.")


def transfer(data: dict|torch.Tensor,
             device: str|torch.device=None,
             dtype: torch.dtype=None,
             **kwargs) -> dict|torch.Tensor:

    def _transfer_tensor(tensor, device=None, dtype=None, **kwargs):
        device = tensor.device if device is None else device
        dtype = tensor.dtype if dtype is None else dtype
        return tensor.to(device=device, dtype=dtype, **kwargs)

    if isinstance(data, torch.Tensor):
        return _transfer_tensor(tensor=data, device=device, dt=dtype, **kwargs)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict or torch.Tensor, got {type(data)}")

    result = {}
    for kw, content in data.items():

        if isinstance(content, dict):
            result[kw] = transfer(data=content, device=device, dtype=dtype, **kwargs)
        elif isinstance(content, torch.Tensor):
            result[kw] = _transfer_tensor(tensor=content, device=device, dtype=dtype, **kwargs)
        else:
            raise TypeError(f"Expected dict or torch.Tensor for key '{kw}', got {type(content)}")

    return result