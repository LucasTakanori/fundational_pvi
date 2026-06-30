import src.models._model_mapper
from src.packages import *

from src.utils.primitives import *
from src.utils import miscellaneous as misc

from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_extraction import PviRawDataset
from src.pipeline.data_preparation_eager import PviSingleDataset, PviCompositeDataset

from src.models.trainer_v3 import ModelEvaluator

# Loading pre-trained checkpoints and run inference
def main(pm: ProjectPathManager,
         ds_list: list[PviDataFile],
         subject_id: str) -> None:

    checkpoint_path = pm.logdirs["checkpoints"] / (subject_id + "_checkpoints.pth")
    if not checkpoint_path.exists():
        raise RuntimeError(f"Datasets for '{subject_id}' exists but checkpoint not found!")

    target_model, input_mode, output_mode = src.models._model_mapper.ml_session_mapper(pm.logdir.name)
    raws = [PviRawDataset(ds_file=file).load() for file in ds_list]

    ds_subject = PviCompositeDataset(ds_raws=raws,
                                     input_mode=input_mode,
                                     output_mode=output_mode,
                                     mask_key=SequenceMask.MASK05,
                                     name=subject_id,
                                     ).build(cleanup=True)

    ds_subject = ds_subject.to(device=DEFAULT_TRAIN_DEVICE, dtype=DEFAULT_TRAIN_DTYPE)

    checkpoint = torch.load(checkpoint_path)
    ds_subject.load_state_dict(checkpoint['dataset'])

    config_path = pm.logdirs["configs"] / f'{ds_subject.name}_configs.json'

    print(f"\t Processing: '{config_path.name}'")
    with open(config_path, 'r') as file:
        record = json.load(file)
        record['dataset'] = ds_subject.get_params_shallow()

    with open(config_path, 'w') as file:
        json.dump(record,
                  file,
                  indent=4,
                  default=str)

    # mappings = ds_subject._compute_local_masks()
    # mappings_df = pd.DataFrame(mappings)
    #
    # export_dir = pm.logdirs["configs"] / "_partition"
    # export_dir.mkdir(parents=True, exist_ok=True)
    # export_path = export_dir / f'{ds_subject.name}_partition_mappings.csv'
    # mappings_df.to_csv(export_path, index=False)

    # ds_subject.set_dataloaders(batch_size=32, shuffle=False)
    #
    # model = target_model(ds_subject.shapes).to(device=DEFAULT_TRAIN_DEVICE, dtype=DEFAULT_TRAIN_DTYPE)
    #
    # evaluator = ModelEvaluator(dataset=ds_subject,
    #                            model=model,
    #                            checkpoint_path=checkpoint_path)

    # evaluator.unpack_from_checkpoint('model')
    # evaluator.unpack_from_checkpoint('dataset')
    #
    # dict_out = evaluator.get_stats()
    #
    # export_dir = pm.logdirs["statistics"]
    # export_dir.mkdir(parents=True, exist_ok=True)
    # export_path = export_dir / f'{ds_subject.name}_statistics.json'
    # with open(export_path, 'w') as json_file:
    #     json.dump(dict_out, json_file, indent=4, default=str)

    if 'raws' in locals():
        for dsr in raws:
            dsr.unload()
            del dsr

        del raws

    if 'ds_subject' in locals():
        misc.cleanup_attributes(ds_subject, attrs=["data", "sequences"])
        del ds_subject

    gc.collect()

    pass

if __name__ == '__main__':

    artifact_repo = Path(r"D:\PviProject\artifacts\_final_ss")
    pm = ProjectPathManager()

    inventory = PviDatasetInventory(branch='main')

    subdirs = [f for f in artifact_repo.iterdir() if f.is_dir()]

    # subdirs = subdirs[2:3]

    for sd in subdirs:
        pm.configure(sd)

        for subID in inventory.subjects:
            ds_list = inventory.filter(subID)
            # if not ds_list:
            #     continue  # Skip subjects with no data

            main(pm=pm,
                 ds_list=ds_list,
                 subject_id=subID)
    pass