from src.packages import *
from src.utils.primitives import *
from src.models._model_mapper import *

import src.utils.miscellaneous as misc
from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_extraction import PviRawDataset
from src.pipeline.data_preparation_eager import PviCompositeDataset
from src.pipeline.data_preparation_lazy import PviLazyDataset

from src.models.base_model import BasePviLearner

from src.models.loss_functions import MorphologyLoss
from src.models.early_stopper import EarlyStopper
from src.models.workflow_v3 import TrainingWorkflow
from src.models.trainer_v3 import ModelEvaluator

# =============================================================================

def run(dataset, model, pm) -> None:

    mse_weight = 1.0 if isinstance(model, PviLinearRegression) else 0.2

    loss_fn = MorphologyLoss(base_loss=nn.MSELoss(), base_weight=mse_weight)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     patience=50,
                                                     mode='min',
                                                     factor=0.85)

    early_stopper = EarlyStopper(patience=100,
                                 delta=1e-6,
                                 mode='max',
                                 threshold=0.5,
                                 verbose=True)

    wf = TrainingWorkflow(path_manager=pm,
                          dataset=dataset,
                          model=model,
                          loss_func=loss_fn,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          stopper=early_stopper,)

    wf.set_checkpoint_interval(minutes=180, epochs=100)

    wf.initiate_training(use_checkpoint=True,
                         device=DEFAULT_TRAIN_DEVICE,
                         dtype=DEFAULT_TRAIN_DTYPE)

    wf.run(min_epochs=1, max_epochs=1000)

    wf.checkpoint.save()
    wf.export_artifacts()

if __name__ == '__main__':
    inventory_main = PviDatasetInventory(branch='main')
    inventory_long = PviDatasetInventory(branch='long')

    # MUST ADJUST THESE THREE VALUES BEFORE RUNNING
    dtags_old = [f'd{k:02d}' for k in range(0,5)] # 'd00', 'd01', ... 'd04'
    dtags_new = [f'd{k:02d}' for k in range(1,6)] # 'd01', 'd02', ... 'd05'
    pop_ranges = [1, 2, 3, 4, 5] # number of longitudinal ds to include

    model_tags = ['ss15', 'ss17'] # change to ['ss03', 'ss07']

    for subID in inventory_long.subjects[4:]: # loop through subjects
        repo = Path(r"D:\PviProject\artifacts\_long") / subID

        for mtag in model_tags[:1]: # loop through models

            for otag, ntag, pops in zip(dtags_old, dtags_new, pop_ranges): # loop through days
                dirs = [f for f in repo.iterdir() if f.is_dir()]
                mdirs = [f for f in dirs if mtag in f.name]

                odir = [f for f in mdirs if otag in f.name][0]
                ndir = odir.parent / odir.name.replace(otag, ntag)

                target_model, input_mode, output_mode = ml_session_mapper(ndir.name)

                pm_old = ProjectPathManager(branch='long', target=odir)
                pm_new = ProjectPathManager(branch='long', target=ndir)

                ds_long = inventory_long.filter(subID)
                ds_train = inventory_main.filter(subID) # this is correct, don't fix

                for _ in range(pops):
                    ds_train.append(ds_long.pop(0)) # append longitudinal dataset to main

                ds_train = [PviRawDataset(ds_file=file).load() for file in ds_train]
                ds_train = PviCompositeDataset(
                        ds_raws=ds_train,
                        input_mode=input_mode,
                        output_mode=output_mode,
                        mask_key="mask05",
                        name=subID).build(cleanup=True)

                checkpoint_old = pm_old.logdirs['checkpoints'] / f'{subID}_checkpoints.pth'
                checkpoint_old = torch.load(checkpoint_old)

                ds_train.load_state_dict(checkpoint_old['dataset'])  # load original partition

                # only partition the final one, because everything before is already stored
                ds_last = ds_train.singles[-1]
                ds_last.set_partition(test_size=0.5, shuffle=False)
                _ = ds_last.get_partition()

                offset = sum([ds.num_periods for ds in ds_train.singles[:-1]])
                ds_last = ds_last.remove_offset().add_offset(offset)

                for kw in ['active_mask', 'train_mask', 'test_mask']:
                    new = getattr(ds_train, kw) + getattr(ds_last, kw)
                    setattr(ds_train, kw, new)

                ds_train.subsets = ds_train._get_subsets_from_split()

                ds_train.set_dataloaders(batch_size=32, shuffle=True)

                model = target_model(ds_train.shapes)
                model.load_state_dict(checkpoint_old['model'])

                run(model=model,
                    dataset=ds_train,
                    pm=pm_new)

                if 'ds_train' in locals():
                    misc.cleanup_attributes(ds_train, attrs=["data", "sequences"])
                    del ds_train

                gc.collect()