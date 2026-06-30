from src.packages import *

from src.utils.primitives import *
from src.utils import h5io

from src.pipeline.data_discovery import ProjectPathManager
from src.pipeline.data_preparation_eager import PviConfiguredDataset
from src.utils import miscellaneous as misc

from scipy.stats import pearsonr

import geomloss as gl

class QualityEvaluator:
    def __init__(self,
                 dataset: PviConfiguredDataset,
                 model: nn.Module,
                 loss_func: nn.Module,
                 checkpoint_path: str | Path = None,
                 ) -> None:
        self.dataset = dataset
        self.model = model
        self.loss_func = loss_func

        self.checkpoint = None

        # TO-DO: Get following dataset statistics
        # num_samples
        # train_size
        # test_size
        # num_epochs

        # Following must be computed after inference
        # sqi
        # W1Loss(Xtrain, Xtest)
        # W1Loss(Ytrain, Ytest)
        # W1Loss(Ytest, Ytrue)

        # amae
        # armse

        # sbp_r2
        # sbp_pv
        # sbp_cc
        # sbp_mae
        # sbp_sd
        # sbp_tol05
        # sbp_tol10
        # sbp_tol15
        # dbp_r2
        # dbp_pv
        # dbp_cc
        # dbp_mae
        # dbp_sd
        # dbp_tol05
        # dbp_tol10
        # dbp_tol15

        if checkpoint_path:
            print(f"{self._alias}: Checkpoint provided. Loading checkpoint...")
            self.checkpoint = torch.load(checkpoint_path, weights_only=True)

    def unpack_from_checkpoint(self, kw: str) -> None:
        if self.checkpoint is None:
            raise ValueError("Checkpoint not available!")

        if (not hasattr(self, kw)) or (getattr(self, kw) is None):
            raise NotImplementedError(f"Component '{kw}' not available for class {type(self).__name__}")

        component = getattr(self, kw)
        component.load_state_dict(self.checkpoint[kw])
        setattr(self, kw, component)

        if kw.lower() == 'dataset':
            self.loaders = self.dataset.get_dataloaders()

            print(f"{self._alias} (WARNING): Dataset from checkpoint will differ from holdout dataset!",
                  "If holdout dataset is being used for inference, it must be assigned to the evaluator object.")

    def assign_dataset(self, dataset: PviConfiguredDataset) -> None:
        self.dataset = dataset
        _ = self.dataset.get_partition()

        self.loaders = self.dataset.get_dataloaders()

    def infer_results(self):
        pass

    def compute_rmse(self):
        pass

    def compute_amae(self):
        pass

    def compute_wdloss(self):
        pass

    def compute_metrics(self) -> None:
        pass