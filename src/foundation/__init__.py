"""PVI foundation model: a shared core + per-subject readout heads.

The design follows the "core + readout" recipe popularised by data-driven models
of sensory cortex (and the Nature foundation-model paper this repo is built around):
a single backbone (the *core*) is pretrained across the pooled population so it
learns a general representation, while each subject gets a lightweight *readout*
head. Transfer to a new subject = freeze the pretrained core and fit a fresh
readout on that subject's data.

Public API:
    PviCore             - shared backbone (src/foundation/core.py)
    SubjectReadout      - per-subject head (src/foundation/readout.py)
    PviFoundationModel  - core + readouts, a drop-in BasePviLearner subclass
    FoundationConfig    - hyper-parameter container
"""

from src.foundation.config import FoundationConfig
from src.foundation.core import PviCore
from src.foundation.readout import SubjectReadout
from src.foundation.foundation_model import PviFoundationModel, SHARED_READOUT

__all__ = [
    "FoundationConfig",
    "PviCore",
    "SubjectReadout",
    "PviFoundationModel",
    "SHARED_READOUT",
]
