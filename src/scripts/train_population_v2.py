from src.packages import *

from src.utils.primitives import *
from src.models._model_mapper import *

from src.pipeline.data_extraction import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_preparation_lazy import PviLazyDataset

from src.models.base_model import BasePviLearner

from src.models.loss_functions import MorphologyLoss
from src.models.early_stopper import EarlyStopper
from src.models.workflow_v3 import TrainingWorkflow

# =============================================================================

def main(target_model: type(BasePviLearner),
         input_mode: str|InputMode,
         output_mode: str|OutputMode,
         logdir: str|Path,
         ds_root: str|Path=None,
         ) -> None:

    pm = ProjectPathManager(branch='main', target=logdir)

    # Dataset location resolves from (in order): the `ds_root` argument, the
    # PVIPROJECT_ROOT env var (via ProjectRoot), or a repo-relative ./data/datasets.
    ds_root = Path(ds_root) if ds_root is not None else ProjectRoot()() / "datasets"
    ds_list = PviDatasetInventory(ds_root=ds_root, branch='main')

    ds_lazy = PviLazyDataset(ds_files=ds_list,
                             input_mode=InputMode(input_mode),
                             output_mode=OutputMode(output_mode),
                             mask_key=SequenceMask.MASK05,
                             max_cache=50,
                             persistent_handle=True)

    ds_lazy = ds_lazy.build()

    ds_lazy.set_partition(test_size=0.1,
                          shuffle=True,
                          split_mode='disjoint')

    ds_lazy.set_dataloaders(batch_size=32,
                            shuffle=False,
                            stratified=True,  # this is the money-maker
                            )

    model = target_model(ds_lazy.shapes)

    mse_weight = 1.0 if isinstance(model, PviLinearRegression) else 0.2

    loss_fn = MorphologyLoss(base_loss=nn.MSELoss(), base_weight=mse_weight)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=50,
                                                     mode='min',
                                                     factor=0.8)
    early_stopper = EarlyStopper(patience=50,
                                 delta=1e-4,
                                 mode='max',
                                 threshold=0.5,
                                 verbose=True)

    wf = TrainingWorkflow(path_manager=pm,
                          dataset=ds_lazy,
                          model=model,
                          loss_func=loss_fn,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          stopper=early_stopper,
                          )

    wf.set_checkpoint_interval(minutes=120, epochs=10)

    wf.initiate_training(use_checkpoint=True,
                         device=DEFAULT_TRAIN_DEVICE,
                         dtype=DEFAULT_TRAIN_DTYPE)

    ds_lazy._print_info_default()

    # wf.stopper._reset()
    wf.run(min_epochs=1,
           max_epochs=500)

if __name__ == '__main__':
    from src.models.mlp_models import PviLinearRegression, PviMLP
    from src.models.cnn_models import PviCNN
    from src.models.attn_models import PviCNNTransformer
    from src.models.s4_models import PviSamba

    # main(target_model=PviSamba,
    #      input_mode="img",
    #      output_mode="waveform",
    #      logdir="ps17-samba-img-to-waveform")

    # main(target_model=PviSamba,
    #      input_mode="img",
    #      output_mode="waveform",
    #      logdir="ps17-samba-img-to-waveform")

    main(target_model=PviSamba,
         input_mode="img",
         output_mode="fiducials",
         logdir="ps18-samba-img-to-fiducials")

    pass # in case we comment everything and the script is empty