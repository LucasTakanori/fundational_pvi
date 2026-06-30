from src.packages import *

from src.utils.primitives import *
from src.utils import miscellaneous as misc

from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_extraction import PviRawDataset
from src.pipeline.data_preparation_eager import PviSingleDataset, PviCompositeDataset

# Loading pre-trained checkpoints and run inference
def main(ds_list: list[PviDataFile],
         subject_id: str) -> None:


    # data = stack_images_3d(ds_list)
    # cmap = bkr()
    # fg_dir = ds_list[0].path.parent / '_frames' / f'{subject_id}'
    # fg_dir.mkdir(parents=True, exist_ok=True)
    # from PIL import Image
    # for k, tensor in enumerate(data,start=1):
    #     print(f'{subject_id} ({k}/{len(data)})')
    #     fg_name = f'{subject_id}_s{k:04d}'
    #     fg_path = fg_dir / (fg_name + '.png')
    #     rgb = (cmap((tensor + 1) / 2) * 255).astype(np.uint8)[:, :, :3] # drop alpha
    #     Image.fromarray(rgb).save(fg_path)

    data = stack_interps_1d(ds_list)
    csv_dir = ds_list[0].path.parent / '_interps' / 'subjects'
    csv_dir.mkdir(parents=True, exist_ok=True)
    # for sn, tensor in data.items():
    #     df = pd.DataFrame(tensor)
    #     csv_name = f'{subject_id}_{sn}.csv'
    #     csv_path = csv_dir / csv_name
    #     df.to_csv(csv_path, header=False, index=False)

    # data = extract_raw_stats(ds_list)
    # df = pd.DataFrame(data)
    # csv_dir = ds_list[0].path.parent / '_stats' / 'subjects'
    # csv_dir.mkdir(parents=True, exist_ok=True)
    # csv_name = f'{subject_id}_stats.csv'
    # csv_path = csv_dir / csv_name
    # df.to_csv(csv_path, header=True, index=False)
    #
    # gc.collect()

    pass

def stack_images_3d(ds_list: list[PviDataFile]) -> list[torch.Tensor]:

    def stack_grid(tensor: torch.Tensor,
                   num_rows:int=5,
                   num_cols:int=10,
                   ) -> torch.Tensor:

        _, height, width = tensor.shape

        grid = tensor.reshape(num_rows, num_cols, height, width)
        grid = grid.permute(0, 2, 1, 3)
        grid = grid.reshape(num_rows * height, num_cols * width)
        return grid

    def rescale(tensor: torch.Tensor) -> torch.Tensor:
        # take difference
        tensor = tensor - tensor[0]

        # rescale
        vmin = np.nanmin(tensor)
        vmax = np.nanmax(tensor)
        vlim = max(abs(vmin), abs(vmax))
        tensor = tensor / vlim
        return tensor

    raws = [PviRawDataset(ds_file=file).load() for file in ds_list]
    ds = PviCompositeDataset(
            ds_raws=raws,
            input_mode=InputMode.IMAGE,
            output_mode=OutputMode.WAVEFORM,
            name=raws[0].subject,
            mask_key=SequenceMask.MASK01).build(cleanup=True)

    data = []
    for sample in ds:
        tensor = None
        for sn in PviChannelGroup.keys():
            tmp = sample[sn].detach().cpu() # shape: (1, H, W, T)
            tmp = tmp.squeeze() # shape: (H, W, T)
            if tensor is None:
                tensor = tmp
            else:
                tensor += tmp
        tensor = torch.permute(tensor, (2,0,1)).contiguous() # shape: (T, H, W)
        tensor = rescale(tensor)
        tensor = stack_grid(tensor)
        data.append(tensor)

    return data

def stack_interps_1d(ds_list: list[PviDataFile]) -> dict[str, torch.Tensor]:

    raws = [PviRawDataset(ds_file=file).load() for file in ds_list]
    ds = PviCompositeDataset(
            ds_raws=raws,
            input_mode=InputMode.IMPEDANCE,
            output_mode=OutputMode.WAVEFORM,
            name=raws[0].subject,
            mask_key=SequenceMask.MASK01).build(cleanup=True)

    if ds.input_mode != InputMode.IMPEDANCE:
        raise AttributeError("Expected IMPEDANCE input mode.")

    signal_names = ['bp'] + PviChannelGroup.keys()
    data = {sn: None for sn in signal_names}
    for sn in signal_names:
        for sample in ds:
            tensor = sample[sn].detach().cpu() # shape: (2C, T)
            row = tensor.flatten() # shape: (1, T*2C)
            if data[sn] is None:
                data[sn] = row
            else:
                data[sn] = torch.vstack([data[sn], row]).contiguous()

    return data

def extract_raw_stats(ds_list: list[PviDataFile]) -> dict[str, torch.Tensor]:
    raws = [PviRawDataset(ds_file=file).load() for file in ds_list]

    keys = ['sbp', 'dbp', 'hr', 'zmax', 'zmin', 'cmax', 'cmin']
    data = {k: [] for k in keys}

    ds_img = PviCompositeDataset(
            ds_raws=raws,
            input_mode=InputMode.IMAGE,
            output_mode=OutputMode.FIDUCIALS,
            name=raws[0].subject,
            mask_key=SequenceMask.MASK01).build(cleanup=False)

    ds_bioz = PviCompositeDataset(
            ds_raws=raws,
            input_mode=InputMode.IMPEDANCE,
            output_mode=OutputMode.FIDUCIALS,
            name=raws[0].subject,
            mask_key=SequenceMask.MASK01).build(cleanup=True)

    for sample in ds_img:
        bp = sample['bp']
        sbp = bp.max().item()
        dbp = bp.min().item()
        hr = 60/sample['stats'][0].item()

        data['sbp'].append(sbp)
        data['dbp'].append(dbp)
        data['hr'].append(hr)

    for sample in ds_img:
        rHP = sample['pviHP']
        # rLP = sample['pviLP']
        img = rHP
        img = img.nanmean(dim=(1, 2)).squeeze()
        cmax = img.max().item()
        cmin = img.min().item()
        data['cmax'].append(cmax)
        data['cmin'].append(cmin)

    for sample in ds_bioz:
        rHP = sample['pviHP'][:32]
        # rLP = sample['pviLP'][:32]
        resistance = rHP
        zmax = resistance.max().item()
        zmin = resistance.min().item()
        data['zmax'].append(zmax)
        data['zmin'].append(zmin)

    data = {k: torch.tensor(v) for k, v in data.items()}

    return data

from matplotlib.colors import LinearSegmentedColormap
def bkr(N=256):
    # Blue (most negative) -> Black (zero) -> Red (most positive)
    cdict = {
        'red': [(0.0, 0.0, 0.0),
                (0.5, 0.0, 0.0),
                (1.0, 1.0, 1.0)],

        'green': [(0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0)],

        'blue': [(0.0, 1.0, 1.0),
                 (0.5, 0.0, 0.0),
                 (1.0, 0.0, 0.0)]
    }

    cmap = LinearSegmentedColormap('BKR', cdict, N=N)
    cmap.set_bad(color='white')  # NaN renders as white

    return cmap

if __name__ == '__main__':

    artifact_repo = Path(r"D:\PviProject\artifacts\_final_ss")

    inventory = PviDatasetInventory(branch='long')
    for subID in inventory.subjects:
        ds_list = inventory.filter(subID)
        # if not ds_list:
        #     continue  # Skip subjects with no data

        main(ds_list=ds_list,
             subject_id=subID)
    pass