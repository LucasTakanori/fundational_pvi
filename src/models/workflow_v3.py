from src.packages import *

from src.utils.primitives import DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE, ArtifactType
from src.utils.primitives import DefaultStringFormat as dfmt
from src.pipeline.data_extraction import ProjectPathManager
from src.pipeline._data_preparation import PviConfiguredDataset

from src.models.early_stopper import EarlyStopper
from src.models.tracking import MetricsTracker, TrainingCheckpoint, TrainingLogger
from src.models.trainer_v3 import ModelTrainer, ModelEvaluator

class TrainingWorkflow:
    def __init__(self,
                 path_manager: ProjectPathManager,
                 dataset: PviConfiguredDataset,
                 model: nn.Module,
                 loss_func: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler.LRScheduler = None,
                 stopper: EarlyStopper = None,
                 device: str | torch.device = None,
                 dtype: torch.dtype = None,
                 ) -> None:

        # self._alias = type(self).__name__
        self._alias = "Workflow"
        self.path_manager = path_manager

        self.dataset = dataset
        self.model = model
        self.device = device
        self.dtype = dtype

        self.trainer = ModelTrainer(dataset=dataset,
                                    model=model,
                                    optimizer=optimizer,
                                    loss_func=loss_func)

        # optional
        self.scheduler = scheduler
        self.stopper = stopper

        # tracking purpose
        self.status = 'initial'
        self.epoch = -1
        self.tracker = MetricsTracker()

        self.checkpoint = TrainingCheckpoint(dataset=self.dataset,
                                             model=self.model,
                                             optimizer=self.trainer.optimizer,
                                             loss_func=self.trainer.loss_func,
                                             scheduler=scheduler,
                                             stopper=self.stopper,
                                             tracker=self.tracker,
                                             path_manager=self.path_manager,)

        self.logger = TrainingLogger(dataset=self.dataset,
                                     model=self.model,
                                     optimizer=self.trainer.optimizer,
                                     loss_func=self.trainer.loss_func,
                                     scheduler=scheduler,
                                     stopper=self.stopper,
                                     tracker=self.tracker,
                                     )

        self._last_checkpoint_epoch: int = -1
        self._last_checkpoint_time: float = time.time()

        self.min_epochs: int = 0
        self.max_epochs: int = 0

        self._checkpoint_interval: dict = {}
        self.set_checkpoint_interval(minutes=180, epochs=50)

        self._paths: dict[ArtifactType, Path] = {}
        self._generate_save_paths()

    def _format_header(self, epoch) -> str:

        parts = [
            [f"Epoch: {self.epoch:,}/{self.max_epochs:,} (session @ {epoch:,})",
             f"Model: {self.model._alias}",
             f"Device: {self.model.device}",
             ],
            [f"Dataset: {self.dataset.name}",
             f"Input Mode: {self.dataset.input_mode.value}",
             f"Output Mode: {self.dataset.output_mode.value}",
             ]
            ]
        header = "\n".join([" | ".join(p) for p in parts])

        return header

    def run(self,
            min_epochs: int = 0,
            max_epochs: int = 100,
            ) -> nn.Module:

        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

        for epoch in range(max_epochs):
            if self.stopper and self.stopper.trigger_stop:
                print(f"\nEarly stopping triggered! (epoch={self.epoch})")
                # self.status='best'
                break

            if self.epoch >= max_epochs:
                print(f"\nMax global epochs reached! (epoch={self.epoch})")
                # self.status = 'terminal'
                break

            self.epoch += 1  # global epoch
            header = self._format_header(epoch)

            print("-"*15)
            print(header)

            # compute train metrics in 'train mode'
            train_metrics = self.trainer.train_epoch()

            train_loss = train_metrics['loss']
            train_accuracy = train_metrics['bp_accuracy']

            # compute test metrics in 'test mode'
            results = self.trainer.evaluate_epoch()

            test_metrics = self.trainer.compute_tracking_metrics(*results)
            test_loss = test_metrics['loss']
            test_accuracy = test_metrics['bp_accuracy']

            print(f"\t Epoch Loss (Train|Test): {train_loss:.4f}|{test_loss:.4f}")
            print(f"\t Epoch Accuracy (Train|Test): {train_accuracy:.4f}|{test_accuracy:.4f}")

            if self.scheduler and self.epoch >= min_epochs:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(test_loss)
                else:
                    self.scheduler.step()

                current_lr = self.scheduler.get_last_lr()[0]

            else:
                current_lr = self.trainer.optimizer.param_groups[0]['lr']

            tracking_metrics = {'train_loss': train_loss,
                                'test_loss': test_loss,
                                'train_accuracy': train_accuracy,
                                'test_accuracy': test_accuracy,
                                'lr': current_lr}

            self.tracker.add_epoch(self.epoch, tracking_metrics)

            if self.stopper and self.epoch >= min_epochs:
                self.stopper.step(self.epoch, test_accuracy)

            self._checkpoint_if_needed()

            # only valid for lazy dataset
            self.dataset.cleanup(attrs="cache", placeholder=None)
            gc.collect()

        self.terminate_training()

        return self.model

    def _checkpoint_if_needed(self) -> None:
        # THIS IS TO BE USED IN THE TRAINING LOOP.
        # NOT IN THE INITIATION AND TERMINATION
        now = time.time()
        crit1 = (now - self._last_checkpoint_time) / 60 >= self._checkpoint_interval['minutes']
        crit2 = (self.epoch > 0) and (self.epoch % self._checkpoint_interval['epochs'] == 0)
        crit3 = self.stopper and self.stopper.found_best

        if crit1 or crit2:
            self._last_checkpoint_time = now
            self._last_checkpoint_epoch = self.epoch
            checkpoint_type = 'interval'
            suffix = None

        elif crit3:
            checkpoint_type = 'best'
            suffix = 'best'

        else:
            checkpoint_type = None
            suffix = None

        if checkpoint_type:
            self.status = checkpoint_type
            new = self.checkpoint.create(name=checkpoint_type)
            self.checkpoint.save(new, suffix=suffix)
            self.export_artifacts(status=self.status)

    def set_checkpoint_interval(self,
                                minutes: int = 180,
                                epochs: int = 10) -> None:

        self._checkpoint_interval = {'minutes': minutes, 'epochs': epochs}

    def initiate_training(self,
                          use_checkpoint: bool=False,
                          device: torch.device=None,
                          dtype: torch.dtype=None) -> None:

        if self.path_manager.logdir is None:
            raise RuntimeError(f"{self._alias}: Logging dir not set! Cannot start training.")

        print()
        print(f"{self._alias}: Initiating training loop...")

        if use_checkpoint:
            train_from_scratch = False
            try:
                old = self.checkpoint.load()
                self.checkpoint.unpack(old)

                self.epoch = old['epoch']
                self.status = old['name'] # initial/interval/terminal/best

                if self.epoch > 0:
                    train_loss = old['tracker']['train_loss']
                    test_loss = old['tracker']['test_loss']
                    train_accuracy = old['tracker']['train_accuracy']
                    test_accuracy = old['tracker']['test_accuracy']

                    print(f"{self._alias}: Previous metrics @epoch={self.epoch:,}:")
                    print(f"\t Model Loss (Train|Test): {train_loss[-1]:.4f}|{test_loss[-1]:.4f}")
                    print(f"\t Model Accuracy (Train|Test): {train_accuracy[-1]:.4f}|{test_accuracy[-1]:.4f}")

                    print(f"{self._alias}: Resume training...")

            except Exception as e:
                print(f"{self._alias} (WARNING): Cannot load or unpack checkpoint. Will resume training from scratch!")
                train_from_scratch = True

        else:
            train_from_scratch = True

        if train_from_scratch:
            if len(self.dataset.train_mask + self.dataset.test_mask)==0:
                _ = self.dataset.get_partition()

        _ = self.dataset.get_dataloaders()

        # final sanity check before running
        if len(self.dataset.subsets['test']) ==0 or len(self.dataset.subsets['train']) == 0:
            raise RuntimeError("Subsets not available!")

        if len(self.dataset.loaders['test']) ==0 or len(self.dataset.loaders['train']) == 0:
            raise RuntimeError("Loaders not available!")

        self.device = self.device if device is None else device
        self.dtype = self.dtype if dtype is None else dtype
        self.trainer = self.trainer.to(device=self.device, dtype=self.dtype)
        self.trainer.transfer_optimizer(device=self.device, dtype=self.dtype)

        if self.status=='initial':
            new = self.checkpoint.create(name=self.status)
            self.checkpoint.save(new, suffix=None)
            self.export_artifacts(status=self.status)

    def terminate_training(self) -> None:
        if self.epoch < 0:
            raise RuntimeError("Invalid state!")

        print()
        print(f"{self._alias}: Terminating training loop...")

        new = self.checkpoint.create(name='terminal')
        try:
            old = self.checkpoint.load(suffix='best')
            print(f"{self._alias}: Best checkpoint found at epoch {old['epoch']}.")
            best_accuracy = old['tracker']['test_accuracy'][-1]
            current_accuracy = new['tracker']['test_accuracy'][-1]

            if best_accuracy >= current_accuracy:
                self.checkpoint.unpack(old)
                self.epoch = old['epoch']
                new = old
                print(f"{self._alias}: Best checkpoint restored as final.")

            else:
                print(f"{self._alias}: Final checkpoint saved as best.")

        except:
            print(f"{self._alias}: Cannot load BEST checkpoint. Use current checkpoint as final.")

        print(f"{self._alias}: Final model saved.")

        train_loss = new['tracker']['train_loss']
        test_loss = new['tracker']['test_loss']
        train_accuracy = new['tracker']['train_accuracy']
        test_accuracy = new['tracker']['test_accuracy']

        print()
        print(f"{self._alias}: Exit metrics @epoch={self.epoch:,}:")
        print(f"\t Model Loss (Train|Test): {train_loss[-1]:.4f}|{test_loss[-1]:.4f}")
        print(f"\t Model Accuracy (Train|Test): {train_accuracy[-1]:.4f}|{test_accuracy[-1]:.4f}")

        self.status = 'terminal'

    def export_artifacts(self, status: str=None) -> None:

        status = self.status if status is None else status

        print(f"{self._alias}: Exporting artifacts to:")
        print(f"\t {self.path_manager.export_root}")
        print(f"\t Logdir: '{self.path_manager.target}'")
        print(f"\t Branch: '{self.path_manager.branch.value}'")

        # always export logger
        self.logger.update(status=status)
        self.logger.export(save_path=self._paths[ArtifactType.CONFIGS])

        # only export history and results if available
        if status != 'initial':
            self.tracker.export(save_path=self._paths[ArtifactType.HISTORY])

            if 'test' in self.trainer.eval_results:
                results = self.trainer.eval_results['test']
            else:
                results = self.trainer.evaluate_epoch(kw='test')

            self.trainer.export_results(results=results, save_path=self._paths[ArtifactType.RESULTS])

        # only export final status at terminal
        if status == 'terminal':
            evaluator = ModelEvaluator(
                    dataset=self.trainer.dataset,
                    model=self.trainer.model)

            if 'test' in evaluator.eval_results:
                _ = evaluator.eval_results['test']
            else:
                _ = evaluator.evaluate_epoch(kw='test')

            stats = evaluator.get_stats()
            evaluator.export_stats(stats=stats, save_path=self._paths[ArtifactType.STATISTICS])

    def _generate_save_paths(self) -> None:
        pm = self.path_manager

        logger_path = pm.generate_artifact_path(
                core_name=self.dataset.name,
                artifact_name=ArtifactType.CONFIGS.value,
                extension="json")

        history_path = pm.generate_artifact_path(
                core_name=self.dataset.name,
                artifact_name=ArtifactType.HISTORY.value,
                extension='csv')

        results_path = pm.generate_artifact_path(
                core_name=self.dataset.name,
                artifact_name=ArtifactType.RESULTS.value,
                extension='csv')

        stats_path = pm.generate_artifact_path(
                core_name=self.dataset.name,
                artifact_name=ArtifactType.STATISTICS.value,
                extension='json')

        self._paths = {
            ArtifactType.CONFIGS: logger_path,
            ArtifactType.HISTORY: history_path,
            ArtifactType.RESULTS: results_path,
            ArtifactType.STATISTICS: stats_path,
        }