"""Smoke tests: confirm the relocated `src` package is importable and that the
reused training stack runs end-to-end on a synthetic batch.

The bundled `src/utils/test.h5` is an empty placeholder (no real recordings),
so these tests build an in-memory batch shaped like the dataset's output
(`pviHP`, `pviLP`, `bp`, `stats`) rather than reading data from disk. Real data
must be supplied via `PVIPROJECT_ROOT` for actual training (see the README).

Run:  python -m pytest tests/test_smoke.py -q   (from the repo root)
"""

from pathlib import Path

import torch

from src.models.base_model import BasePviLearner
from src.models.loss_functions import MorphologyLoss
from src.models.early_stopper import EarlyStopper          # alias of EarlyStopCounter
from src.models.workflow_v3 import TrainingWorkflow         # import must succeed
from src.models.trainer_v3 import ModelTrainer              # import must succeed
from src.pipeline.data_preparation_lazy import PviLazyDataset  # import must succeed
from src.foundation import (
    PviFoundationModel, FoundationConfig, PviCore, SubjectReadout, SHARED_READOUT,
)

# Small synthetic problem: single-channel 1-D PVI signal of length T.
B, C, T, OUT = 4, 1, 50, 8
SHAPES = {"input": (C, T), "output": (OUT,), "stats": (1,)}


def _batch(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "pviHP": torch.randn(B, C, T, generator=g),
        "pviLP": torch.randn(B, C, T, generator=g),
        "bp": torch.randn(B, OUT, generator=g),
        "stats": torch.randn(B, 1, generator=g),
    }


def _model(**kw) -> PviFoundationModel:
    return PviFoundationModel(SHAPES, num_features=32, num_hidden_layers=2,
                              verbose=False, **kw)


def test_imports_and_aliases():
    assert issubclass(PviFoundationModel, BasePviLearner)
    assert EarlyStopper.__name__ == "EarlyStopCounter"  # compatibility alias
    assert callable(ModelTrainer) and callable(TrainingWorkflow)
    assert PviLazyDataset is not None


def test_bundled_sample_h5_opens():
    import h5py
    sample = Path(__file__).resolve().parents[1] / "src" / "utils" / "test.h5"
    assert sample.exists()
    with h5py.File(sample, "r") as f:
        # placeholder file: opens cleanly even though it carries no recordings
        assert isinstance(list(f.keys()), list)


def test_foundation_forward_shapes():
    model = _model(subjects=["subject001"])
    assert SHARED_READOUT in model.readouts and "subject001" in model.readouts

    seqs, stats, targets = model.process_batch(_batch())
    out = model(seqs, stats)
    assert out.shape == (B, OUT)
    assert targets.shape == (B, OUT)

    # per-subject routing leaves no permanent state change
    out_sub = model.forward_for(seqs, stats, "subject001")
    assert out_sub.shape == (B, OUT)
    assert model.active == SHARED_READOUT


def test_core_and_readout_are_composable():
    flat = B * 0 + (C * (FoundationConfig().diff + 1) * T + 1)  # channels*(diff+1)*T + stats
    core = PviCore(flat, num_features=16, num_hidden_layers=1)
    readout = SubjectReadout(16, OUT)
    feats = core(torch.randn(B, flat))
    assert feats.shape == (B, 16)
    assert readout(feats).shape == (B, OUT)


def test_training_step_reduces_loss():
    """Mirror ModelTrainer.train_epoch's inner loop to exercise the reused stack."""
    torch.manual_seed(0)
    model = _model()
    loss_fn = MorphologyLoss(base_loss=torch.nn.MSELoss(), base_weight=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    batch = _batch(seed=1)
    losses = []
    for _ in range(10):
        seqs, stats, targets = model.process_batch(batch)
        preds = model(seqs, stats)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert losses[-1] < losses[0]  # overfits a fixed batch


def test_transfer_trains_readout_only():
    model = _model()
    model.freeze_core()
    model.add_readout("subject042")
    model.set_active("subject042")

    assert all(not p.requires_grad for p in model.core.parameters())
    assert all(p.requires_grad for p in model.readouts["subject042"].parameters())

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)
    loss_fn = MorphologyLoss()

    seqs, stats, targets = model.process_batch(_batch(seed=2))
    loss = loss_fn(model(seqs, stats), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # core received no gradients
    assert all(p.grad is None for p in model.core.parameters())
