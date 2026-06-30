from src.packages import *

from src.utils.primitives import *
from src.models._model_mapper import *

from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_extraction import PviRawDataset
from src.pipeline.data_preparation_eager import PviCompositeDataset
from src.pipeline.data_preparation_lazy import PviLazyDataset

from src.models.mlp_models import PviLinearRegression

from src.models.loss_functions import MorphologyLoss
from src.models.trainer_v3 import ModelEvaluator
from src.pipeline._data_preparation import ensemble_distance
from src.models import perf_metrics

# =============================================================================
if __name__ == '__main__':
    inventory_main = PviDatasetInventory(branch='main')
    dsp_main = PviLazyDataset(
            ds_files=inventory_main,
            input_mode=InputMode.IMPEDANCE,
            output_mode=OutputMode.FIDUCIALS,
            mask_key=SequenceMask.MASK05,
    ).build()

    inventory_holdout = PviDatasetInventory(branch='holdout')
    dsp_holdout = PviLazyDataset(
            ds_files=inventory_holdout,
            input_mode=InputMode.IMPEDANCE,
            output_mode=OutputMode.FIDUCIALS,
            mask_key=SequenceMask.MASK05,
    ).build()

    dsp_main.set_partition(test_size=0.1, shuffle=True, split_mode='within')

    checkpoint_path = r"D:\PviProject\artifacts\_final_pw\pw15-crt-bioz-to-waveform\main\checkpoints\dataset_lazy_checkpoints.pth"
    checkpoint = torch.load(checkpoint_path)

    dsp_main.load_state_dict(checkpoint['dataset'])

    dsp_main.set_dataloaders(batch_size=100, shuffle=False)
    dsp_main.get_dataloaders()

    bp = []
    for batch in dsp_main.loaders['test']:
        bp.append(batch['bp'])

    export_dir = Path(r"D:\PviProject\artifacts\_final_pw\pw15-crt-bioz-to-waveform\main")

    bp = torch.vstack(bp).numpy()
    np.savetxt(export_dir / "bp_test.csv", bp, delimiter=",")

    pass