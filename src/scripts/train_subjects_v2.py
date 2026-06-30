from src.packages import *

from src.utils.primitives import *
from src.utils import miscellaneous as misc

from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_extraction import PviRawDataset
from src.pipeline.data_preparation_eager import PviCompositeDataset

from src.models.base_model import BasePviLearner

from src.models.loss_functions import MorphologyLoss
from src.models.early_stopper import EarlyStopper
from src.models.workflow_v3 import TrainingWorkflow

# =============================================================================

def main(ds_list: list[PviDataFile],
         target_model: type(BasePviLearner),
         input_mode: str|InputMode,
         output_mode: str|OutputMode,
         pm:ProjectPathManager,
         ) -> None:

    raws = [PviRawDataset(ds_file=file).load() for file in ds_list]

    ds_subject = PviCompositeDataset(ds_raws=raws,
                                     input_mode=input_mode,
                                     output_mode=output_mode,
                                     mask_key="mask05",
                                     name=subID,
                                     ).build(cleanup=True)

    ds_subject.set_partition(test_size=0.1, shuffle=True)
    ds_subject.set_dataloaders(batch_size=32, shuffle=True)

    model = target_model(ds_subject.shapes)

    mse_weight = 1.0 if isinstance(model, PviLinearRegression) else 0.2

    loss_fn = MorphologyLoss(base_loss=nn.MSELoss(),
                             base_weight=mse_weight)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=100,
                                                     mode='min',
                                                     factor=0.9)

    stopper = EarlyStopper(patience=200,
                           delta=1e-4,
                           mode='max',
                           threshold=0.5,
                           verbose=True)

    wf = TrainingWorkflow(path_manager=pm,
                          dataset=ds_subject,
                          model=model,
                          loss_func=loss_fn,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          stopper=stopper, )

    wf.set_checkpoint_interval(epochs=100)

    wf.initiate_training(use_checkpoint=True,
                         device=DEFAULT_TRAIN_DEVICE,
                         dtype=DEFAULT_TRAIN_DTYPE)

    print()
    wf.dataset.print_info()
    print()
    wf.model.print_info()

    wf.run(min_epochs=1, max_epochs=5000)

    wf.checkpoint.save()
    wf.export_artifacts()

    if 'raws' in locals():
        for dsr in raws:
            dsr.unload()
            del dsr

        del raws

    if 'ds_subject' in locals():
        misc.cleanup_attributes(ds_subject, attrs=["data", "sequences"])
        del ds_subject

    gc.collect()

if __name__ == '__main__':
    from src.models.mlp_models import PviLinearRegression, PviMLP
    # from src.models.cnn_models import PviCNN
    from src.models.attn_models import PviCNNTransformer
    # from src.models.s4_models import PviSamba

    pm = ProjectPathManager(branch='main',target='_test')
    inventory = PviDatasetInventory(branch='main')

    for subID in inventory.subjects:
        ds_list = inventory.filter(subID)

        main(ds_list=ds_list,
             target_model=PviLinearRegression,
             input_mode="bioz",
             output_mode="waveform",
             pm=pm)

    pass
