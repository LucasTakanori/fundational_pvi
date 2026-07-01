from src.utils.primitives import *
from src.utils.miscellaneous import *
from src.utils.h5io import *
from typing import Iterator

class ProjectPathManager:
    def __init__(self,
                 branch: str | TrainingBranch,
                 target: str | Path,
                 export_root: str | Path = None) -> None:

        if export_root is None:
            self.export_root = ProjectRoot().root / "artifacts"
        else:
            self.export_root = Path(export_root)

        self.target = target
        self.branch = TrainingBranch(branch)

        self.logdir = None
        self.logdirs: dict[str,Path] = {}

        self.configure()

    def configure(self, clear_if_temp: bool=True) -> None:

        if isinstance(self.target, str):
            self.logdir = self.export_root / self.target / self.branch.value

            if self.target in ['_temp', '_tmp', 'temp', 'tmp'] and self.logdir.exists() and clear_if_temp:
                import shutil
                print(f"{type(self).__name__} (WARNING): Logging dir '{self.target}' WILL BE OVERWRITTEN!")
                shutil.rmtree(self.logdir)

        elif isinstance(self.target, Path) and self.target.is_absolute():
            self.export_root = self.target.parent
            self.target = self.target.name

            self.logdir = self.export_root / self.target / self.branch.value

        else:
            raise ValueError("Cannot set logging dir")

        self.logdir.mkdir(parents=True, exist_ok=True)

        print(f"{type(self).__name__}:\n\t Logging dir set to: '{str(self.logdir)}'")
        self.logdirs: dict[str, Path] = {}
        for kw in ArtifactType.keys():
            self.logdirs[kw] = self.logdir / kw
            self.logdirs[kw].mkdir(parents=True, exist_ok=True)

    def generate_artifact_path(self,
                               core_name: str,
                               artifact_name: str,
                               extension: str,
                               prefix: str = None,
                               suffix: str = None,
                               save_dir: Path|str=None,
                               ) -> Path:

        if save_dir is None:
            if (not self.logdirs) or (self.logdir is None):
                raise ValueError("Logging dir not set, cannot generate save path.")


        save_dir = self.logdirs[artifact_name]

        save_name = f"{core_name}_{artifact_name}"
        if prefix is not None:
            save_name = f"{prefix}_{save_name}"
        if suffix is not None:
            save_name = f"{save_name}_{suffix}"

        save_path = save_dir / (save_name + f".{extension.lower()}")
        return save_path

    def print_info(self) -> None:
        print("=" * 15 + f"[{type(self).__name__}]" + "=" * 15)

        try:
            print(f"Project root directory:\n\t {self.root}")
            if self.logdir:
                subdirs = [s for s in self.logdir.rglob('*') if s.is_dir()]
                print(f"Logging directory:\n\t {self.logdir}")
                for s in subdirs:
                    print(f"\t {s}")
            else:
                print("Logging directory not set")
        except:
            print("Cannot display info. Something is amiss!")

        print("=" * 15 + f"[{type(self).__name__}]" + "=" * 15)

class PviDatasetInventory:
    def __init__(self,
                 branch: str | TrainingBranch = TrainingBranch.MAIN,
                 ds_root: str | Path = None) -> None:
        self._alias = type(self).__name__

        self.ds_root = resolve_data_root(ds_root)

        # list of all possible sessions and subjects. Not necessarily what available in the target directory
        self._range_sessions = SessionName.keys()
        self._range_subjects = SubjectName.keys()

        # available sessions and subjects in target directory
        self.sessions: list[str] = []
        self.subjects: list[str] = []
        self.datasets: list[PviDataFile] = []

        self.branch = TrainingBranch(branch)
        self.target_dir = self.ds_root / self.branch.value
        print(f"{type(self).__name__}: Dataset directory set to:\n\t '{self.target_dir}'")

        self.datasets: list[PviDataFile] = self.find_all_datasets()

        print(f"{self._alias}: Ready.")

    def __iter__(self) -> Iterator[PviDataFile]:
        return iter(self.datasets)

    def __getitem__(self, idx: int) -> PviDataFile:
        return self.datasets[idx]

    def __len__(self) -> int:
        return len(self.datasets)

    def find_all_datasets(self) -> list[PviDataFile]:
        ds_list = []
        subjects_avail = set()
        sessions_avail = set()

        print(f"{type(self).__name__}: Searching dataset directory...")
        for subject in SubjectName.keys():
            for session in SessionName.keys():
                ds_name = '_'.join([subject, session])
                file_path = self.target_dir / (ds_name + "_masked.h5")
                if self._validate_dataset_path(file_path):
                    file = PviDataFile(name=ds_name,
                                       session=session,
                                       subject=subject,
                                       path=file_path)
                    ds_list.append(file)
                    subjects_avail.add(subject)
                    sessions_avail.add(session)

        print(f"\t Found {len(ds_list)} available datasets!")
        ds_list = self._sort_datasets(ds_list)
        self.subjects = list(subjects_avail)
        self.sessions = list(sessions_avail)

        self.subjects.sort()
        self.sessions.sort(key=lambda item: SessionName.keys().index(item))

        return ds_list

    def filter(self,
               keywords: str | list[str],
               ds_target: list[PviDataFile]=None,
               ) -> list[PviDataFile]:
        sessions_valid = set()
        subjects_valid = set()
        keywords_invalid = set()

        if isinstance(keywords, str):
            keywords = {keywords}

        print(f"{type(self).__name__}: Searching for datasets with keywords {keywords}...")

        for kw in keywords:
            if not isinstance(kw, str):
                raise TypeError(f"Expect all keywords to be of type string. Got type '{type(kw)}' for input '{kw}'. ")
            if kw in self.sessions:
                sessions_valid.add(kw)
            elif kw in self.subjects:
                subjects_valid.add(kw)
            else:
                keywords_invalid.add(kw)

        if keywords_invalid:
            excluded = '\n\t '.join(sorted(keywords_invalid))
            print(f"{type(self).__name__} (WARNING): Following invalid keywords excluded from search space:\n\t {excluded}")

        if (not subjects_valid) and (not sessions_valid):
            print(f"{type(self).__name__} (WARNING): Found 0 matching datasets.")
            return []

        if not sessions_valid:
            sessions_valid = self.sessions

        if not subjects_valid:
            subjects_valid = self.subjects

        if ds_target is None:
            ds_target = self.datasets

        condition = lambda file: (file.subject in subjects_valid) and (file.session in sessions_valid)
        ds_match = list(filter(condition, ds_target))

        if len(ds_match):
            ds_match = self._sort_datasets(ds_match)
            print(f"{type(self).__name__}: Found {len(ds_match)} matching datasets:")

            separator = '\n\t '
            print(f"\t {separator.join([f.name for f in ds_match])}")
        else:
            print(f"{type(self).__name__} (WARNING): Found 0 matching datasets.")

        return ds_match

    @staticmethod
    def _sort_datasets(file_list: list) -> list:
        predicate = lambda file: (file.subject, SessionName.keys().index(file.session))
        new_list = sorted(file_list, key=predicate)

        return new_list

    def _validate_dataset_path(self,
                               file_path: Path,
                               exhaustive: bool=False) -> bool:
        if not file_path.is_file():
            # print(f"{type(self).__name__} (WARNING): Dataset '{file_path.name}' NOT available!")
            return False
        else:
            if not exhaustive:
                return True
            else:
                try:
                    with h5py.File(file_path, 'r') as _:
                        return True
                except:
                    print(f"{type(self).__name__} (WARNING): Dataset '{file_path.name}' available but NOT accessible!")
                    return False

if __name__ == '__main__':
    pm = ProjectPathManager(branch='main',target='_tmp')
    inventory_main = PviDatasetInventory(branch='main')
    inventory_long = PviDatasetInventory(branch='long')
    inventory_holdout = PviDatasetInventory(branch='holdout')

    pass