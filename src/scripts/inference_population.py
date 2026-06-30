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
def main(pm_main: ProjectPathManager,
         inventory_main: PviDatasetInventory | list[PviDataFile],
         pm_infer: ProjectPathManager,
         inventory_infer: PviDatasetInventory | list[PviDataFile],
         ) -> None:

    target_model, input_mode, output_mode = ml_session_mapper(pm_main.target)
    device = DEFAULT_TRAIN_DEVICE
    # device = 'cpu'

    ds_main = PviLazyDataset(
            ds_files=inventory_main,
            input_mode=InputMode(input_mode),
            output_mode=OutputMode(output_mode),
            mask_key=SequenceMask.MASK05,
            max_cache=10,
            persistent_handle=False).build().to(device=device, dtype=DEFAULT_TRAIN_DTYPE)

    config_path = pm_main.logdirs["configs"] / "dataset_lazy_configs.json"
    model_kwargs = get_samba_kwargs(config_path) if config_path.exists() else {}
    model = target_model(ds_main.shapes, **model_kwargs).to(device=device, dtype=DEFAULT_TRAIN_DTYPE)

    checkpoint_path = pm_main.logdirs["checkpoints"] / "dataset_lazy_checkpoints.pth"

    if (pm_main == pm_infer) and (inventory_main == inventory_infer):
        ds_infer = ds_main
        evaluator = ModelEvaluator(dataset=ds_main, model=model, checkpoint_path=checkpoint_path)
        evaluator.unpack_from_checkpoint('dataset')

    else:
        ds_infer = PviLazyDataset(
                ds_files=inventory_infer,
                input_mode=InputMode(input_mode),
                output_mode=OutputMode(output_mode),
                mask_key=SequenceMask.MASK05,
                max_cache=10,
                persistent_handle=False).build().to(device=device, dtype=DEFAULT_TRAIN_DTYPE)

        evaluator = ModelEvaluator(dataset=ds_infer, model=model, checkpoint_path=checkpoint_path)
        evaluator.set_partition(test_size=1.0, shuffle=False)
        evaluator.get_partition()

        ds_main.load_state_dict(evaluator.checkpoint['dataset']) # load train subset for W1 distance

    
    evaluator.set_loaders(batch_size=32, shuffle=False, num_workers=0)
    evaluator.get_loaders()

    model_state = evaluator.checkpoint['model']
    if any(k.startswith('_orig_mod.') for k in model_state):
        model_state = {k.replace('_orig_mod.', '', 1): v for k, v in model_state.items()}
        evaluator.checkpoint['model'] = model_state
    if 'fc_out.weight' in model_state:
        model_state['mlp.0.weight'] = model_state.pop('fc_out.weight')
        model_state['mlp.0.bias'] = model_state.pop('fc_out.bias')

    evaluator.unpack_from_checkpoint('model')
    M1, M2 = evaluator.evaluate_epoch()

    save_path = pm_infer.logdirs["results"] / "dataset_lazy_results.csv"
    evaluator.export_results((M1,M2),save_path) # override previous results to make sure we sorted

    # M = pd.read_csv(save_path).to_numpy()
    # M1 = M[:,:50]
    # M2 = M[:, 50:]

    gm = ds_infer.mappings.masks_global
    subjects = [f.subject for f in ds_infer.mappings.files]

    mapper = {m: s for m, s in zip(gm, subjects)}
    subTest = [mapper[m] for m in sorted(ds_infer.test_mask)]
    idx = []
    subjects = sorted(list(set(subTest)))
    for subID in subjects:
        idx.append(subTest.index(subID))

    idx.append(len(ds_infer.test_mask))
    bounds = [(i1, i2) for (i1, i2) in zip(idx, idx[1:] + [None])]

    for b, subID in zip(bounds, subjects):
        sl = slice(*b)
        results = M1[sl], M2[sl]
        save_path = pm_infer.logdirs["results"] / f"{subID}_results.csv"
        evaluator.export_results(results, save_path)

    # D1 = ds_main.subsets['train']
    # D2 = ds_infer.subsets['test']
    #
    # wdX, wdY = ensemble_distance(D1, D2, ds_main.input_mode)
    # wdGen = perf_metrics.metrics_ensemble(M1, M2)
    # wd_dict = {'w1_domain': wdX,
    #            'w1_label': wdY,
    #            'w1_gen': wdGen}
    #
    # stats = evaluator.get_stats(compute_wd=False)
    # stats.update(wd_dict)
    # save_path = pm_infer.logdirs["statistics"] / "dataset_lazy_statistics.json"
    # evaluator.export_stats(stats, save_path)

    pass

if __name__ == '__main__':

    artifact_repo = Path(r"/mnt/d/artifacts/_final_ablations")
    subdirs = [f for f in artifact_repo.iterdir() if f.is_dir()]

    ds_fallback = Path(r'/mnt/d/datasets/')

    for sd in subdirs:
        pm_main = ProjectPathManager(branch='main', target=sd)
        pm_main.logdir = sd
        pm_main.logdirs = {kw: sd / kw for kw in ArtifactType.keys()}
        inventory_main = PviDatasetInventory(branch='main', ds_root=ds_fallback)

        # pm_infer = ProjectPathManager(branch='main', target=sd)
        # inventory_infer = PviDatasetInventory(branch='main')

        main(pm_main=pm_main,
             pm_infer=pm_main,
             inventory_main=inventory_main,
             inventory_infer=inventory_main)