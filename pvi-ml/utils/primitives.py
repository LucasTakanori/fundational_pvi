import os
from pathlib import Path
from dataclasses import dataclass
from enum import  Enum

import torch

DEFAULT_TRAIN_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_TRAIN_DTYPE = torch.float32

# class DatasetRoot:
#     def __init__(self, rootdir: str|Path=None) -> None:
#         if rootdir is None:
#             if os.name == "nt":
#                 self.root = Path(r"D:\PviProject\datasets")
#             elif os.name == "posix":
#                 if "PVIPROJECT_ROOT" in os.environ:
#                     self.root = Path(os.environ["PVIPROJECT_ROOT"]) / "datasets"
#                 else:
#                     self.root = Path(r"/uufs/chpc.utah.edu/common/home/sanchez-terrones-group1/PviProject/datasets")
#             else:
#                 raise NotImplementedError("Unrecognizable OS!")
#         else:
#             self.root = Path(rootdir)

class ProjectRoot:
    def __init__(self, rootdir: str|Path=None) -> None:
        if rootdir is None:
            if os.name == "nt":
                self.root = Path(r"D:\PviProject")
            elif os.name == "posix":
                if "PVIPROJECT_ROOT" in os.environ:
                    self.root = Path(os.environ["PVIPROJECT_ROOT"])
                else:
                    self.root = Path(r"/uufs/chpc.utah.edu/common/home/sanchez-terrones-group1/PviProject")
            else:
                raise NotImplementedError("Unrecognizable OS!")
        else:
            self.root = Path(rootdir)

    def __call__(self) -> Path:
        return self.root

SubjectName = Enum('SubjectName', {f'SUBJECT{k:03d}': f'subject{k:03d}' for k in range(1, 100 + 1)})
SubjectName.keys = classmethod(lambda cls: [m.value for m in cls])

class SessionName(Enum):
    BASELINE = "baseline"
    VALSALVA = "valsalva"
    PRESSOR = "pressor"
    LONG01 = "day01"
    LONG02 = "day02"
    LONG03 = "day03"
    LONG04 = "day04"
    LONG05 = "day05"

    @classmethod
    def _missing_(cls, value: str):
        """Handle case-insensitive lookup"""
        for member in cls:
            if member.value == value.lower():
                return member
        return None

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

@dataclass
class PviDataFile:
    name: str
    subject: str
    session: str
    path: Path

@dataclass
class PviMetaData(PviDataFile):
    num_periods: int
    num_frames: int
    bounds_periods: tuple
    bounds_frames: tuple
    shapes: dict
    masks: dict

class PviChannelGroup(Enum):
    HP = "pviHP"
    LP = "pviLP"

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

class RawStats(Enum):
    # must be in this order: duration, tmax
    DURATION = "duration"
    TMAX = "tMax"

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

class PviSignalGroup(Enum):
    # Used when loading and formatting raw tensors. Separate from InputMode
    IMAGE = "img"
    SIGNAL = "signal"
    RESISTANCE = "resistance"
    REACTANCE = "reactance"

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

class SplitMode(Enum):
    GLOBAL = "global"
    WITHIN = "within"
    DISJOINT = "disjoint"
    MIXED = "mixed"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        if value  in ["global"]:
            return cls.GLOBAL

        elif value in ["local", "intra_subjects", "intra", "within"]:
            return cls.WITHIN

        elif value in ["subjects", "inter_subjects", "inter", "disjoint"]:
            return cls.DISJOINT

        elif value in ["mixed", "pooled", "pooling"]:
            return cls.MIXED

        return None

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

class InputMode(Enum):
    # Used when formatting loaded tensors. Separate from PviMode
    IMAGE = "img"
    IMPEDANCE = "impedance"
    SIGNAL = "signal"
    RESISTANCE = "resistance"
    REACTANCE = "reactance"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        if value  in ["img", "image", "3d"]:
            return cls.IMAGE
        elif value in ["impedance", "bioimpedance", "bioz", "bio-z", "imp", "2d"]:
            return cls.IMPEDANCE
        elif value in ["signal", "1d"]:
            return cls.SIGNAL
        elif value in ["resistance", "r"]:
            return cls.RESISTANCE
        elif value in ["reactance", "x"]:
            return cls.REACTANCE

        return None

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

class OutputMode(Enum):
    SYSTOLIC = "sbp"
    DIASTOLIC = "dbp"
    FIDUCIALS = "fiducials"
    WAVEFORM = "waveform"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        if value in ["systolic", "sbp", "sys", "systolic_bp", "max", "maximum"]:
            return cls.SYSTOLIC
        elif value in ["diastolic", "dbp", "dia", "diastolic_bp", "min", "minimum"]:
            return cls.DIASTOLIC
        elif value in ["fiducials", "minmax", "fiducial"]:
            return cls.FIDUCIALS
        elif value in ["waveform", "full", "full_waveform"]:
            return cls.WAVEFORM

        return None

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

class SequenceMask(Enum):
    MASK01 = "mask01"
    MASK05 = "mask05"
    MASK10 = "mask10"
    MASK15 = "mask15"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        if value in ["mask01", "seq01", "mask_01", "seq_01"]:
            return cls.MASK01
        elif value in ["mask05", "seq05", "mask_05", "seq_05"]:
            return cls.MASK05
        elif value in ["mask10", "seq10", "mask_10", "seq_10"]:
            return cls.MASK10
        elif value in ["mask15", "seq15", "mask_15", "seq_15", "15", 15]:
            return cls.MASK15

        return None

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

class TrainingBranch(Enum):
    ''' DO NOT CHANGE THE STRING VALUES BECAUSE THEY CORRESPOND TO FOLDER NAMES'''
    MAIN = "main"
    LONGITUDINAL = "longitudinal"
    HOLDOUT = "holdout"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        if value in ["main", "train-test", "dev"]:
            return cls.MAIN
        elif value in ["longitudinal", "long"]:
            return cls.LONGITUDINAL
        elif value in ["holdout", "test-exclusive"]:
            return cls.HOLDOUT

        return None

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

class ArtifactType(Enum):
    ''' DO NOT CHANGE THE STRING VALUES BECAUSE THEY CORRESPOND TO FOLDER NAMES'''
    CONFIGS = "configs"
    CHECKPOINTS = "checkpoints"
    RESULTS = "results"
    STATISTICS = "statistics"
    HISTORY = "history"

    @classmethod
    def keys(cls) -> list[str]:
        return [member.value for member in cls]

@dataclass
class DefaultStringFormat:
    datetime: str = '%Y-%m-%d %H:%M:%S'
    tqdm: str = '{l_bar}{bar:15}|{n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_fmt}{postfix}]'