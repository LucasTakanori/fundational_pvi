from src.packages import *

from tqdm import tqdm

from src.utils import h5io
from src.utils.primitives import DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE, ArtifactType
from src.utils.primitives import DefaultStringFormat as dfmt

from src.pipeline.data_discovery import ProjectPathManager
from src.pipeline._data_preparation import ensemble_distance
from src.pipeline.data_preparation_eager import PviConfiguredDataset
from src.pipeline.data_preparation_lazy import PviLazyDataset

from src.models.base_model import BasePviLearner
from src.models import perf_metrics

class BaseModelHandler(ABC):
    def __init__(self,
                 dataset: PviConfiguredDataset,
                 model: nn.Module|BasePviLearner,
                 loss_func: nn.Module|None,
                 ) -> None:

        self._alias = type(self).__name__
        self.dataset = dataset
        self.model = model
        self.loss_func = loss_func

        self.eval_results: dict[str, tuple[torch.Tensor, torch.Tensor]]  = {}

        self.device = None
        self.dtype = None

    def to(self, device:torch.device|str=None, dtype: torch.dtype=None) -> 'BaseModelHandler':
        self.model = self.model.to(device=device, dtype=dtype)
        # Lazy loaders yield CPU tensors; train_epoch/evaluate_epoch move each batch.
        # Calling dataset.to(cuda) breaks pin_memory in DataLoader.
        if not isinstance(self.dataset, PviLazyDataset):
            self.dataset = self.dataset.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

        return self

    def evaluate_epoch(self, kw:str='test') -> tuple[torch.Tensor, torch.Tensor]:

        kw = kw.lower()
        if kw not in ['train', 'test']:
            raise RuntimeError(f"Invalid loader key '{kw}'. Expected 'train' or 'test'")

        loader = self.dataset.loaders[kw]

        if not len(loader):
            raise RuntimeError("Empty loader! Cannot perform inference")

        pbar = tqdm(iterable=loader, desc=f"\t Inferring", unit='batch', bar_format=dfmt.tqdm)

        all_predictions = []
        all_targets = []

        self.model.eval()
        with torch.no_grad():
            for batch in pbar:
                batch = h5io.transfer(batch, device=self.device, dtype=self.dtype)

                input_sequences, input_stats, batch_targets = self.model.process_batch(batch)
                batch_predictions = self.model(input_sequences, input_stats)

                all_predictions.append(batch_predictions.detach().cpu())
                all_targets.append(batch_targets.detach().cpu())

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        self.eval_results[kw] = (predictions, targets)

        return predictions, targets

    def compute_tracking_metrics(self,
                                 predictions: torch.Tensor,
                                 targets: torch.Tensor) -> dict[str, float]:

        loss = self.loss_func(predictions, targets).detach().cpu()
        accuracy = perf_metrics.bp_accuracy(predictions, targets)

        return {'loss': float(loss), 'bp_accuracy': float(accuracy)}

    @staticmethod
    def format_inference(predictions: torch.Tensor | np.ndarray,
                         targets: torch.Tensor | np.ndarray,
                         ) -> pd.DataFrame:
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        stacked_data = np.hstack((predictions, targets))

        num_cols = predictions.shape[1]
        pred_cols = [f"pred_{i + 1}" for i in range(num_cols)]
        target_cols = [f"target_{i + 1}" for i in range(num_cols)]
        columns = pred_cols + target_cols

        df = pd.DataFrame(stacked_data, columns=columns)

        return df

    def export_results(self,
                       results: tuple[torch.Tensor, torch.Tensor],
                       save_path: Path) -> None:

        extension = save_path.suffix.lower()
        valid = '.csv'
        if extension != valid:
            raise ValueError(f"Inappropriate extension '{extension}' for logging. Expect '{valid}'.")

        results_df = self.format_inference(*results)
        results_df.to_csv(save_path, index=False)

        print(f"\t Exported '{save_path.name}'.")

class ModelTrainer(BaseModelHandler):
    def __init__(self,
                 dataset: PviConfiguredDataset,
                 model: nn.Module|BasePviLearner,
                 loss_func: nn.Module,
                 optimizer: optim.Optimizer,
                 clip_grad_norm: float=1.0,
                 ) -> None:

        super().__init__(dataset, model, loss_func)

        if not self.loss_func:
            raise AttributeError(f"Missing loss_func for object of class '{self._alias}'")

        self.optimizer = optimizer

        if clip_grad_norm is None:
            self.clip_grad_norm = None
        elif isinstance(clip_grad_norm, (float, int)):
            self.clip_grad_norm = float(clip_grad_norm)
        else:
            raise ValueError("clip_grad_norm must be a float or int")

        self.use_amp = False
        self._scaler = None

    def set_amp(self, enabled: bool = True) -> None:
        cuda = self.device is not None and getattr(self.device, "type", str(self.device)) == "cuda"
        self.use_amp = bool(enabled and cuda)
        self._scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def train_epoch(self) -> dict[str, float]:
        """
        NOTE:
        Ideally, the training metrics (i.e. loss and bp_accuracy for the training dataset) should be computed in the
        dedicated method "run_inference", to be more robust and also comparable with the test metrics. But that would
        require running the forward passes twice (once during train method, here, and then in the inference method),
        for a single epoch. For large datasets + lazy loading + cache limitation + HPC time limitation, this becomes
        prohibitively expensive (2x training time for 400GB dataset).
        """
        kw = 'train'
        loader = self.dataset.loaders[kw]

        if not len(loader):
            raise RuntimeError("Empty loader! Cannot perform forward pass!")

        pbar = tqdm(iterable=loader, desc=f"\t Training", unit='batch', bar_format=dfmt.tqdm)

        running_metrics = {} # Keys will be added later
        num_samples = 0

        for batch in pbar:
            batch = h5io.transfer(batch,
                                device=self.device,
                                dtype=self.dtype,
                                non_blocking=self.use_amp)

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda",
                                enabled=self.use_amp,
                                dtype=torch.float16):
                input_sequences, input_stats, batch_targets = self.model.process_batch(batch)
                batch_predictions = self.model(input_sequences, input_stats)
                batch_loss = self.loss_func(batch_predictions, batch_targets)

            if self._scaler is not None and self.use_amp:
                self._scaler.scale(batch_loss).backward()
                if self.clip_grad_norm is not None:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                             max_norm=self.clip_grad_norm,
                                             norm_type=2.0)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                batch_loss.backward()
                if self.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                             max_norm=self.clip_grad_norm,
                                             norm_type=2.0)
                self.optimizer.step()

            pbar.set_postfix_str(s=f"loss={batch_loss:.2f}")

            batch_metrics = self.compute_tracking_metrics(batch_predictions, batch_targets)

            current_batch_size = batch_predictions.shape[0]
            for key, value in batch_metrics.items():
                if key not in running_metrics:
                    running_metrics[key] = 0.0  # initialize dict, only active in first batch
                else:
                    running_metrics[key] += value * current_batch_size

            num_samples += current_batch_size

        train_metrics = {key: value / num_samples for key, value in running_metrics.items()}

        return train_metrics

    def transfer_optimizer(self,
                           device:torch.device|str=None,
                           dtype: torch.dtype=None):

        for param in self.optimizer.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device=device,dtype=dtype)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device=device,dtype=dtype)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device=device,dtype=dtype)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device=device,dtype=dtype)

class ModelEvaluator(BaseModelHandler):
    def __init__(self,
                 model: nn.Module,
                 dataset: PviConfiguredDataset=None,
                 checkpoint_path: str|Path=None,
                 ) -> None:
        super().__init__(dataset=dataset, model=model, loss_func=None)

        self.checkpoint = None

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
            print(f"{self._alias} (WARNING): If holdout dataset is being used for inference, use assign_dataset!")
            try:
                self.get_partition()
                self.get_loaders()
            except:
                pass

    def assign_dataset(self, dataset: PviConfiguredDataset) -> None:
        self.dataset = dataset

        if self.device:
            self.dataset = self.dataset.to(device=self.device)
        if self.dtype:
            self.dataset = self.dataset.to(dtype=self.dtype)

        try:
            self.get_partition()
            self.get_loaders()
        except:
            pass

    def set_partition(self, *args, **kwargs) -> None:
        self.dataset.set_partition(*args, **kwargs)

    def get_partition(self) -> None:
        self.dataset.subsets = self.dataset.get_partition()

    def set_loaders(self, *args, **kwargs) -> None:
        self.dataset.set_dataloaders(*args, **kwargs)

    def get_loaders(self) -> None:
        self.dataset.loaders = self.dataset.get_dataloaders()

    def get_stats(self, compute_wd: bool=True) -> dict[str, float | int]:

        ds_params = self.dataset.get_params_shallow()
        summary = {
            'num_train': ds_params['counts']['num_train'],
            'num_test': ds_params['counts']['num_test'],
        }

        dict_out = {}
        dict_out |= ds_params['raw_stats']
        dict_out |= summary

        # results = self.evaluate_epoch()
        results = self.eval_results['test']

        if compute_wd:
            D1 = self.dataset.subsets['train']
            D2 = self.dataset.subsets['test']

            wdX, wdY = ensemble_distance(D1, D2, D1.input_mode)
            wdGen = perf_metrics.metrics_ensemble(*results)

            del D1, D2
            gc.collect()
        else:
            wdX, wdY, wdGen = None, None, None

        wd_dict = {'w1_domain': wdX,
                   'w1_label': wdY,
                   'w1_gen': wdGen}

        dict_out |= wd_dict

        # inference metrics
        dict_out |= perf_metrics.metrics_waveform(*results)
        dict_out |= perf_metrics.metrics_fiducial(*results)

        return dict_out

    @staticmethod
    def export_stats(stats: dict[str, float|int],
                     save_path: Path) -> None:

        extension = save_path.suffix.lower()
        valid = '.json'
        if extension != valid:
            raise ValueError(f"Inappropriate extension '{extension}' for logging. Expect '{valid}'.")

        with open(save_path, 'w') as json_file:
            json.dump(stats,
                      json_file,
                      indent=4,
                      default=str)

        print(f"\t Exported '{save_path.name}'.")