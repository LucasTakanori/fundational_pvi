"""Tests for multi-task readout (BP + maneuver + HR) on the shared core."""

import pytest
import torch

from src.foundation import PviFoundationModel
from src.foundation.multitask import (
    attach_multitask_heads, multitask_loss, MultiTaskWeights,
    MANEUVER_HEAD, HR_HEAD,
)

B, C, T, OUT = 4, 1, 50, 8
SHAPES = {"input": (C, T), "output": (OUT,), "stats": (1,)}


def _model():
    m = PviFoundationModel(SHAPES, num_features=16, num_hidden_layers=2, verbose=False)
    return attach_multitask_heads(m, num_maneuvers=3, hr=True)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    seqs = {"pviHP": torch.randn(B, C, T, generator=g),
            "pviLP": torch.randn(B, C, T, generator=g)}
    stats = torch.randn(B, 1, generator=g)
    bp = torch.randn(B, OUT, generator=g)
    maneuver = torch.randint(0, 3, (B,), generator=g)
    hr = torch.randn(B, 1, generator=g)
    return seqs, stats, bp, maneuver, hr


def test_heads_attached():
    m = _model()
    assert MANEUVER_HEAD in m.aux_heads and HR_HEAD in m.aux_heads


def test_multitask_loss_components_and_finite():
    m = _model()
    seqs, stats, bp, man, hr = _inputs()
    losses = multitask_loss(m, seqs, stats, bp, maneuver_targets=man, hr_targets=hr)
    assert set(losses) == {"bp", "maneuver", "hr", "total"}
    assert all(torch.isfinite(v) for v in losses.values())


def test_multitask_loss_bp_only_when_aux_absent():
    m = PviFoundationModel(SHAPES, num_features=16, num_hidden_layers=2, verbose=False)
    seqs, stats, bp, _, _ = _inputs()
    losses = multitask_loss(m, seqs, stats, bp)  # no aux targets
    assert set(losses) == {"bp", "total"}
    assert torch.allclose(losses["total"], losses["bp"])  # default w_bp = 1.0


def test_missing_head_raises():
    m = PviFoundationModel(SHAPES, num_features=16, num_hidden_layers=2, verbose=False)
    seqs, stats, bp, man, _ = _inputs()
    with pytest.raises(KeyError):
        multitask_loss(m, seqs, stats, bp, maneuver_targets=man)


def test_weights_scale_total():
    m = _model()
    seqs, stats, bp, man, hr = _inputs()
    losses = multitask_loss(m, seqs, stats, bp, maneuver_targets=man, hr_targets=hr,
                            weights=MultiTaskWeights(bp=1.0, maneuver=0.5, hr=0.25))
    expected = losses["bp"] + 0.5 * losses["maneuver"] + 0.25 * losses["hr"]
    assert torch.allclose(losses["total"], expected, atol=1e-6)


def test_multitask_training_step_reduces_total():
    torch.manual_seed(0)
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    seqs, stats, bp, man, hr = _inputs(seed=1)

    first = last = None
    for i in range(15):
        losses = multitask_loss(m, seqs, stats, bp, maneuver_targets=man, hr_targets=hr)
        opt.zero_grad(); losses["total"].backward(); opt.step()
        if i == 0:
            first = losses["total"].item()
        last = losses["total"].item()
    assert last < first
