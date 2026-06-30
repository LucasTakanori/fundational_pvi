from src.packages import *

from src.utils.primitives import *
# from src.utils.miscellaneous import *

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
         ) -> None:

    pm = ProjectPathManager()
    pm.set_data_root(kw="alternative")
    pm.configure(logdir)

    inventory = PviDatasetInventory(ds_root=pm.dataset_root)

    ds_lazy = PviLazyDataset(ds_files=inventory,
                             input_mode=InputMode(input_mode),
                             output_mode=OutputMode(output_mode),
                             mask_key=SequenceMask.MASK05,
                             max_cache=20,
                             persistent_handle=True)

    ds_lazy = ds_lazy.build().to(device='cpu', dtype=DEFAULT_TRAIN_DTYPE)

    ds_lazy.set_partition(test_size=0.1,
                          shuffle=True,
                          split_mode='subjects') # subject-wise partition

    ds_lazy.set_dataloaders(batch_size=32,
                            shuffle=False,
                            stratified=True,  # this is the money-maker
                            # num_workers=2,
                            # prefetch_factor=20,
                            # pin_memory=True,
                            )

    model = target_model(ds_lazy.shapes).to(device=DEFAULT_TRAIN_DEVICE, dtype=DEFAULT_TRAIN_DTYPE)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=50,
                                                     mode='min',
                                                     factor=0.8)
    early_stopper = EarlyStopper(patience=75,
                                 delta=1e-4,
                                 mode='max',
                                 threshold=0.5,
                                 verbose=True)

    loss_fn = MorphologyLoss(base_loss=nn.MSELoss(),
                             base_weight=0.2)

    wf = TrainingWorkflow(path_manager=pm,
                          dataset=ds_lazy,
                          model=model,
                          loss_func=loss_fn,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          stopper=early_stopper,)

    wf.set_checkpoint_interval(minutes=120, epochs=10)

    wf.run(min_epochs=50,
           max_epochs=500,
           use_checkpoint=True)

if __name__ == '__main__':
    # from src.models.mlp_models import PviLinearRegression, PviMLP
    # from src.models.cnn_models import PviCNN
    from src.models.attn_models import PviCNNTransformer
    # from src.models.s4_models import PviSamba

    main(target_model=PviCNNTransformer,
         input_mode="image",
         output_mode="waveform",
         logdir="r13-crt-image-to-waveform")