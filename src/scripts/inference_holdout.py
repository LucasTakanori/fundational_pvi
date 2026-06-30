from src.packages import *

from src.utils.primitives import *
from src.models._model_mapper import *

from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_extraction import PviRawDataset
from src.pipeline.data_preparation_eager import PviCompositeDataset
from src.pipeline.data_preparation_lazy import PviLazyDataset

from src.models.trainer_v3 import ModelEvaluator
from src.models import perf_metrics
from src.pipeline._data_preparation import ensemble_distance

# =============================================================================
def main(pm: ProjectPathManager,
         logdir: Path|str,
         dslist_main: list[PviDataFile],
         dslist_holdout: list[PviRawDataset]|list[PviDataFile]) -> None:

    pm.configure(logdir)
    target_model, input_mode, output_mode = ml_session_mapper(logdir.name)

    ds_main = PviLazyDataset(ds_files=dslist_main,
                             input_mode=InputMode(input_mode),
                             output_mode=OutputMode(output_mode),
                             mask_key=SequenceMask.MASK05,
                             max_cache=2,
                             persistent_handle=True).build()

    raws = []
    for file in dslist_holdout:
        dsr = PviRawDataset(ds_file=file).load()
        raws.append(dsr)

    ds_holdout = PviCompositeDataset(ds_raws=raws,
                                     input_mode=InputMode(input_mode),
                                     output_mode=OutputMode(output_mode),
                                     mask_key=SequenceMask.MASK05,
                                     name="dataset_holdout",
                                     ).build(cleanup=True)

    ds_holdout.set_partition(test_size=1, shuffle=False)
    ds_holdout.set_dataloaders(batch_size=32, shuffle=False)

    model = target_model(ds_holdout.shapes)

    checkpoint_path = pm.logdirs["checkpoints"] / "dataset_lazy_checkpoints.pth"
    evaluator = ModelEvaluator(dataset=ds_holdout, model=model, checkpoint_path=checkpoint_path)
    evaluator = evaluator.to(device=DEFAULT_TRAIN_DEVICE, dtype=DEFAULT_TRAIN_DTYPE)

    evaluator.unpack_from_checkpoint('model')
    evaluator.assign_dataset(dataset=ds_holdout)

    pm.print_info()
    results = evaluator.evaluate_epoch()
    wdGen = perf_metrics.metrics_ensemble(*results)
    results_df = evaluator.format_inference(*results)

    export_dir = pm.logdir / "_holdout" / "results"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = export_dir / f'{ds_holdout.name}_results.csv'
    results_df.to_csv(export_path, index=False)

    stats = evaluator.get_stats(compute_wd=False)

    ds_checkpoint = evaluator.checkpoint['dataset']
    ds_main.load_state_dict(ds_checkpoint)
    ds_main = ds_main.to(device='cpu', dtype=DEFAULT_TRAIN_DTYPE)

    D1 = ds_main.subsets['train']
    D2 = ds_holdout.subsets['test']

    wdX, wdY = ensemble_distance(D1, D2, ds_holdout.input_mode)
    wd_dict = {'w1_domain': wdX,
               'w1_label': wdY,
               'w1_gen': wdGen}

    stats.update(wd_dict)
    save_path = pm.logdirs["statistics"] / "dataset_lazy_statistics.json"
    evaluator.export_stats(stats, save_path)

    pass

if __name__ == '__main__':

    artifact_repo = Path(r"D:\PviProject\artifacts\_final_pw")

    pm = ProjectPathManager()
    pm.artifacts_dir = artifact_repo

    ds_fallback = Path(r'C:\localdata_SRL4\pvi_datasets_clone\datasets')

    inventory_holdout = PviDatasetInventory(ds_root=ds_fallback, branch='holdout')
    inventory_main = PviDatasetInventory(ds_root=ds_fallback, branch='main')

    subdirs = [f for f in artifact_repo.iterdir() if f.is_dir()]

    for sd in subdirs[13:]:
        main(pm=pm,
             logdir=sd,
             dslist_main=list(inventory_main),
             dslist_holdout=list(inventory_holdout))