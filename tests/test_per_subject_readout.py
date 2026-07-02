"""Per-subject readout routing during foundation-cohort pretraining (PLAN.md §14).

The default single ``SHARED_READOUT`` forces one head to fit every subject's absolute
BP, which pressures the shared core to *encode* subject identity - the opposite of a
transferable core. ``enable_per_subject_readouts`` gives each subject its own head and
routes each sample to it, so the core is pushed to encode only subject-shared structure.
These tests exercise that routing on synthetic batches (no real data needed).
"""

import torch

from src.foundation import PviFoundationModel, SHARED_READOUT

B, C, T, OUT = 4, 1, 50, 8
SHAPES = {"input": (C, T), "output": (OUT,), "stats": (1,)}


def _model(**kw) -> PviFoundationModel:
    return PviFoundationModel(SHAPES, num_features=32, num_hidden_layers=2,
                              verbose=False, **kw)


def _uniform_batch(subject_idx: list[int]) -> dict[str, torch.Tensor]:
    """Identical inputs across the batch, so any output difference is pure routing."""
    g = torch.Generator().manual_seed(0)
    hp = torch.randn(1, C, T, generator=g).repeat(B, 1, 1)
    lp = torch.randn(1, C, T, generator=g).repeat(B, 1, 1)
    st = torch.randn(1, 1, generator=g).repeat(B, 1)
    return {
        "pviHP": hp,
        "pviLP": lp,
        "bp": torch.zeros(B, OUT),
        "stats": st,
        "subject_idx": torch.tensor(subject_idx, dtype=torch.long),
    }


def test_multi_readout_routes_by_subject():
    torch.manual_seed(0)
    model = _model()
    model.enable_per_subject_readouts([1, 2])
    assert model._multi_readout and "1" in model.readouts and "2" in model.readouts

    batch = _uniform_batch([1, 1, 2, 2])
    seqs, stats, _ = model.process_batch(batch)
    out = model(seqs, stats)

    assert out.shape == (B, OUT)
    # identical inputs -> identical output within a subject, different across subjects
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[2], out[3])
    assert not torch.allclose(out[0], out[2])


def test_multi_readout_disabled_is_shared():
    torch.manual_seed(0)
    model = _model()
    batch = _uniform_batch([1, 1, 2, 2])
    seqs, stats, _ = model.process_batch(batch)
    out = model(seqs, stats)

    # one shared head -> identical output for identical inputs regardless of subject
    assert torch.allclose(out[0], out[3])
    feats = model.forward_core(seqs, stats)
    assert torch.allclose(out, model.readouts[SHARED_READOUT](feats))


def test_multi_readout_unknown_subject_falls_back_to_shared():
    model = _model()
    model.enable_per_subject_readouts([1])
    batch = _uniform_batch([1, 99, 1, 99])  # subject 99 has no readout
    seqs, stats, _ = model.process_batch(batch)
    out = model(seqs, stats)                 # must not raise
    assert out.shape == (B, OUT)


def test_multi_readout_backward_updates_only_present_readouts():
    torch.manual_seed(0)
    model = _model()
    model.enable_per_subject_readouts([1, 2])
    model.add_readout("3")  # a readout that exists but is absent from the batch
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    batch = _uniform_batch([1, 1, 2, 2])
    seqs, stats, targets = model.process_batch(batch)
    loss = torch.nn.functional.mse_loss(model(seqs, stats), targets)
    opt.zero_grad()
    loss.backward()

    assert any(p.grad is not None for p in model.readouts["1"].parameters())
    assert any(p.grad is not None for p in model.readouts["2"].parameters())
    # subjects absent from the batch (and the untrained shared head) get no gradient
    assert all(p.grad is None for p in model.readouts["3"].parameters())
    assert all(p.grad is None for p in model.readouts[SHARED_READOUT].parameters())
    # the shared core is updated by every subject in the batch
    assert any(p.grad is not None for p in model.core.parameters())
