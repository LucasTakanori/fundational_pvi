from src.packages import *

from src.utils.primitives import *

from src.models._model_mapper import ml_session_mapper

from src.pipeline.data_discovery import ProjectPathManager

from collections import OrderedDict

class ModelAverager:
    def __init__(self, checkpoint_dir: str | Path) -> None:
        self._alias = type(self).__name__

        self.names: list[str] = []
        self.coeffs: list[int] = []
        self.model_dict = None

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_paths = list(self.checkpoint_dir.glob("*_checkpoints.pth"))

        print(f"{self._alias}: Checkpoint directory set to: \n\t {str(self.checkpoint_dir)}...")
        print(f"\t Found checkpoints for {len(self.checkpoint_paths)} submodels!")


    def compute_model_checkpoint(self) -> dict[str, OrderedDict]:
        scale_factor = 1 / sum(self.coeffs)
        state_dict = OrderedDict([(k, tensor * scale_factor) for (k, tensor) in self.model_dict.items()])

        return {'model': state_dict}

    def _update_model_dict(self,
                           new_dict: dict[str, torch.Tensor],
                           coefficient: int=1) -> None:

        for key, tensor in new_dict.items():
            new_dict[key] = coefficient*tensor.to(device='cpu', dtype=DEFAULT_TRAIN_DTYPE)

        if self.model_dict is None:
            self.model_dict = new_dict
        else:
            for key, tensor in new_dict.items():
                self.model_dict[key] += tensor

    def load_checkpoints(self) -> None:
        num_checkpoints = len(self.checkpoint_paths)

        print(f"{self._alias}: Processing {num_checkpoints} submodels:")
        for k, pth in enumerate(self.checkpoint_paths, start=1):
            print(f"\t ({k}/{num_checkpoints}): Loading '{str(pth.name)}'...")
            checkpoint = torch.load(pth)

            state_dict = checkpoint['model']
            # coefficient = len(checkpoint['dataset']['test_mask'])
            coefficient = checkpoint['tracker']['test_accuracy'][-1]

            self._update_model_dict(new_dict=state_dict,
                                    coefficient=coefficient)

            self.names.append(pth.name)
            self.coeffs.append(coefficient)

            del checkpoint

def main(logdir: str|Path) -> None:

    # target_model, input_mode, output_mode = ml_session_mapper(logdir.name)

    pm = ProjectPathManager()
    pm.configure(logdir)

    ma = ModelAverager(checkpoint_dir=pm.logdirs['checkpoints'])
    ma.load_checkpoints()

    checkpoint = ma.compute_model_checkpoint()

    export_dir = pm.logdirs["checkpoints"] / "_holdout"
    export_dir.mkdir(parents=True, exist_ok=True)

    export_path = export_dir / 'average_checkpoints.pth'
    torch.save(checkpoint, export_path)


if __name__ == '__main__':

    artifact_repo = Path(r"D:\PviProject\artifacts\_final_ss")
    subdirs = [f for f in artifact_repo.iterdir() if f.is_dir()]

    for sd in subdirs[::-1]:
        main(logdir=sd)