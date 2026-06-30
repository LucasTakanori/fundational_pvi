from src.packages import *
from src.packages import get_environment

from src.utils.primitives import ArtifactType
from src.utils.primitives import DefaultStringFormat as dfmt
from src.pipeline.data_extraction import ProjectPathManager
from src.pipeline.data_preparation_eager import PviConfiguredDataset

from src.models.early_stopper import EarlyStopper

from collections import OrderedDict

class MetricsTracker:
    def __init__(self) -> None:
        self.history = {'epoch': [],
                        'train_loss': [],
                        'test_loss': [],
                        'train_accuracy': [],
                        'test_accuracy': [],
                        'lr': []
                        }

    def add_epoch(self,
                  num: int,
                  new: dict[str,float]) -> None:
        self.history['epoch'].append(num)
        for kw, value in new.items():
            if kw in self.history:
                self.history[kw].append(float(value))

    def state_dict(self) -> dict[str, list]:
        return self.history

    def load_state_dict(self, state_dict: dict[str, list]):
        self.history = state_dict

    def to_dataframe(self) -> pd.DataFrame:
        for kw, value in self.history.items():
            if kw !='epoch':
                self.history[kw] = [float(i) for i in value]

        return pd.DataFrame(self.history)

    def export(self, save_path) -> None:
        extension = save_path.suffix.lower()
        valid = '.csv'
        if extension != valid:
            raise ValueError(f"Inappropriate extension '{extension}' for logging. Expect '{valid}'.")

        df = self.to_dataframe()
        df.to_csv(save_path, index=False)

        print(f"\t Exported '{save_path.name}'.")

class TrainingLogger:
    def __init__(self,
                 dataset: PviConfiguredDataset,
                 model: nn.Module,
                 loss_func: nn.Module,
                 tracker: MetricsTracker,
                 optimizer: optim.Optimizer = None,
                 scheduler: optim.lr_scheduler.LRScheduler = None,
                 stopper: EarlyStopper = None,
                 ) -> None:

        self._alias = type(self).__name__
        self._alias = "Logger"

        self.dataset = dataset
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stopper = stopper
        self.tracker = tracker

        self._current_record: dict[str,dict] = {}
        self._current_epoch: int = -1
        self._dataset_name = dataset.name
        self._status: str= 'initial'

    def get_dataset(self) -> dict|str|None:
        if self.dataset is None:
            print(f"{self._alias}: Dataset not available for compilation.")
            return None

        try:
            dict_out = self.dataset.get_params_shallow()
            # print(f"{self._alias}: Compiled {self.dataset._alias}.")
            return dict_out

        except Exception as e:
            print(f"{self._alias}: Error compiling params for {type(self.dataset).__name__}.")
            return f'Cannot compile {type(self.dataset).__name__}: {str(e)}'

    def get_model(self) -> dict|str|None:
        if self.model is None:
            print(f"{self._alias}: Model not available for compilation.")
            return None

        try:
            dict_out = self.model.get_params_shallow()
            # print(f"{self._alias}: Compiled {self.model._alias}.")
            return dict_out

        except Exception as e:
            print(f"{self._alias}: Error compiling params for {type(self.model).__name__}.")
            return f'Cannot compile {type(self.model).__name__}: {str(e)}'

    def get_loss_func(self) -> dict|str|None:
        if self.loss_func is None:
            print(f"{self._alias}: Loss function not available for compilation.")
            return None

        try:
            dict_out = self.loss_func.get_params_shallow()
            # print(f"{self._alias}: Compiled {self.loss_func._alias}.")
            return dict_out

        except Exception as e:
            print(f"{self._alias}: Error compiling params for {type(self.loss_func).__name__}.")
            return f'Cannot compile {type(self.loss_func).__name__}: {str(e)}'

    def get_optimizer(self) -> dict|str|None:
        if self.optimizer is None:
            print(f"{self._alias}: Optimizer not available for compilation.")
            return None

        try:
            dict_out = {'name': type(self.optimizer).__name__,
                        'param_groups': {},
                        }
            for idx, pg in enumerate(self.optimizer.param_groups):
                p = {key: value for key, value in pg.items() if key not in ["params", "param_names"]}
                p = {idx: p}
                dict_out['param_groups'].update(p)
            # print(f"{self._alias}: Compiled {type(self.optimizer).__name__}.")
            return dict_out

        except Exception as e:
            print(f"{self._alias}: Error compiling params for {type(self.optimizer).__name__}.")
            return f'Cannot compile {type(self.optimizer).__name__}: {str(e)}'

    def get_scheduler(self) -> dict|str|None:
        if self.scheduler is None:
            print(f"{self._alias}: Scheduler not available for compilation.")
            return None

        try:
            dict_out = {}
            dict_out['name'] = type(self.scheduler).__name__
            dict_out.update(self.scheduler.state_dict())
            # print(f"{self._alias}: Compiled {type(self.scheduler).__name__}.")
            return dict_out

        except Exception as e:
            print(f"{self._alias}: Error compiling params for {type(self.scheduler).__name__}.")
            return f'Cannot compile {type(self.scheduler).__name__}: {str(e)}'

    def get_stopper(self) -> dict|str|None:
        if self.stopper is None:
            print(f"{self._alias}: Stopper not available for compilation.")
            return None

        try:
            dict1 = self.stopper.get_params_shallow()
            dict2 = {k: getattr(self.stopper,k)
                     for k in ['is_active', 'trigger_stop', 'counter', 'found_best', 'best_epoch', 'best_score']}

            dict_out = dict1 | dict2

            # print(f"{self._alias}: Compiled {type(self.stopper).__name__}.")
            return dict_out

        except Exception as e:
            print(f"{self._alias}: Error compiling params for {type(self.stopper).__name__}.")
            return f'Cannot compile {type(self.stopper).__name__}: {str(e)}'

    def get_summary(self) -> dict | str | None:
        if self.tracker is None:
            print(f"{self._alias}: History and metrics not available for compilation.")
            return None

        try:
            history_dict = self.tracker.state_dict()
            epoch = history_dict['epoch']
            if (not epoch) or epoch[-1] < 0:
                self._current_epoch = -1
                return {'status': self._status}

            else:
                self._current_epoch = epoch[-1]

                exit_metrics = {kw: content[-1] for kw, content in history_dict.items()}
                dict_out = {'status': self._status,
                            'total_epoch': self._current_epoch,
                            'exit_metrics': exit_metrics}

                # print(f"{self._alias}: Compiled {type(self.tracker).__name__}.")
                return dict_out

        except Exception as e:
            print(f"{self._alias}: Error compiling params for {type(self.tracker).__name__}.")
            return f'Cannot compile {type(self.tracker).__name__}: {str(e)}'

    def update(self, status: str='interval') -> None:

        self._status = status

        record = {'datetime': datetime.datetime.now().strftime(dfmt.datetime),
                  'summary': self.get_summary(),
                  'dataset': self.get_dataset(),
                  'model': self.get_model(),
                  'loss_func': self.get_loss_func(),
                  'optimizer': self.get_optimizer(),
                  'scheduler': self.get_scheduler(),
                  'stopper': self.get_stopper(),
                  'environment': get_environment()}

        self._current_record = record

    def export(self, save_path) -> None:
        extension = save_path.suffix.lower()
        valid = '.json'
        if extension != valid:
            raise ValueError(f"Inappropriate extension '{extension}' for logging. Expect '{valid}'.")

        with open(save_path, 'w') as json_file:
            json.dump(self._current_record,
                      json_file,
                      indent=4,
                      default=str)

        print(f"\t Exported '{save_path.name}'.")

class TrainingCheckpoint:
    def __init__(self,
                 path_manager: ProjectPathManager,
                 dataset: PviConfiguredDataset,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 tracker: MetricsTracker,
                 loss_func: nn.Module = None,
                 scheduler: optim.lr_scheduler.LRScheduler = None,
                 stopper: EarlyStopper = None,
                 ) -> None:

        self._alias = type(self).__name__
        self._alias = "Checkpoint"

        self.dataset = dataset
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stopper = stopper
        self.tracker = tracker

        self.path_manager = path_manager

        self._current_checkpoint: dict[str,...] = {}
        self._current_epoch: int = -1
        self._dataset_name: str = dataset.name

        self.components = self._get_components()
        self._validate_components()

    def _get_components(self) -> dict:
        # list of dynamic components that can be tracked
        required = ['dataset', 'model', 'optimizer', 'tracker']
        extra = ['loss_func', 'scheduler', 'stopper']
        trackable = required + extra
        available = {kw: getattr(self, kw) for kw in trackable if getattr(self, kw) is not None}

        return available

    def _validate_components(self) -> None:
        for kw, component in self.components.items():
            if (not hasattr(component, 'state_dict')) or (not hasattr(component, 'load_state_dict')):
                raise AttributeError(f"Checkpoint component '{kw}' must implement state_dict() and load_state_dict()")
            try:
                tmp = component.state_dict()
                component.load_state_dict(tmp)
            except:
                raise AttributeError(
                    f"Cannot call state_dict() of checkpoint component '{kw}'")

    def create(self, name:str='interval') -> dict:
        now = datetime.datetime.now().strftime(dfmt.datetime)

        if not self.tracker.history['epoch']:
            epoch = -1
        else:
            epoch = self.tracker.history['epoch'][-1]

        checkpoint = {'datetime': now, 'name': name, 'epoch': epoch}
        t1 = time.perf_counter()

        for kw, component in self.components.items():
            checkpoint[kw] = component.state_dict()

        self._current_checkpoint = checkpoint
        self._current_epoch = checkpoint['epoch']
        self._alias = f"Checkpoint (@{self._current_epoch:,})"

        dt = time.perf_counter() - t1
        print(f"{self._alias}: Checkpoint generated. ({dt:.2f} seconds)")

        return checkpoint

    def save(self, checkpoint: dict=None, suffix: str=None) -> None:
        if checkpoint is None:
            checkpoint = self._current_checkpoint

        save_path = self.path_manager.generate_artifact_path(core_name=self.dataset.name,
                                                             artifact_name=ArtifactType.CHECKPOINTS.value,
                                                             suffix=suffix,
                                                             extension='pth')

        print(f"{self._alias}: Saving '{save_path.name}'...")
        t1 = time.perf_counter()

        torch.save(checkpoint, save_path)

        dt = time.perf_counter() - t1

    def load(self, suffix: str=None) -> dict:
        save_path = self.path_manager.generate_artifact_path(core_name=self.dataset.name,
                                                             artifact_name=ArtifactType.CHECKPOINTS.value,
                                                             suffix=suffix,
                                                             extension='pth')
        print(f"{self._alias}: Loading '{save_path.name}'")
        t1 = time.perf_counter()

        checkpoint = torch.load(save_path, weights_only=True)

        dt = time.perf_counter() - t1

        return checkpoint

    def unpack(self, checkpoint: dict=None) -> None:
        if checkpoint is None:
            checkpoint = self._current_checkpoint

        t1 = time.perf_counter()

        for kw, component in self.components.items():
            if kw in checkpoint:
                if kw == 'model': # load model but respect current device
                    device = component.device # component is a nn.Module model
                    dtype = next(iter(component.state_dict().values())).dtype

                    model_dict = checkpoint['model'] # state dict from checkpoint
                    new_tensors = [tensor.to(device=device, dtype=dtype) for tensor in model_dict.values()]
                    new_dict = OrderedDict(zip(model_dict.keys(), new_tensors))

                    component.load_state_dict(new_dict)

                else:
                    component.load_state_dict(checkpoint[kw])
            else:
                print(f"{self._alias} (WARNING): Component '{kw}' not found in checkpoint.")

        dt = time.perf_counter() - t1

        self._current_checkpoint = checkpoint
        self._current_epoch = checkpoint['epoch']
        self._alias = f"Checkpoint (@{self._current_epoch:,})"

        print(f"{self._alias}: Checkpoint unpacked. ({dt:.2f} seconds)")