"""Milestone 1 tests: core/readout split, freeze-and-transfer, multi-head API.

Exercises the `BasePviLearner` contract on the real architectures using a
synthetic 1-D batch (no data files needed). Confirms:
  * forward == forward_readout o forward_core  (the refactor preserves behaviour)
  * freeze/transfer copies the core, freezes it, and leaves it numerically
    unchanged while only the readout updates
  * auxiliary heads run on the shared core representation
"""

import copy

import pytest
import torch

from src.models.base_model import transfer_core, load_core_from_state_dict
from src.models.mlp_models import PviLinearRegression, PviMLP
from src.models.cnn_models import PviCNN
from src.models.attn_models import PviCNNTransformer
from src.models.loss_functions import MorphologyLoss

B, C, T, OUT = 4, 1, 50, 8
SHAPES = {"input": (C, T), "output": (OUT,), "stats": (1,)}


def _batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    seqs = {"pviHP": torch.randn(B, C, T, generator=g),
            "pviLP": torch.randn(B, C, T, generator=g)}
    stats = torch.randn(B, 1, generator=g)
    targets = torch.randn(B, OUT, generator=g)
    return seqs, stats, targets


# Builders kept tiny so the suite runs fast on CPU.
ARCHES = {
    "linear": lambda: PviLinearRegression(SHAPES),
    "mlp": lambda: PviMLP(SHAPES, num_features=32, num_hidden_layers=2),
    "cnn": lambda: PviCNN(SHAPES, num_conv_layers=2),
    "transformer": lambda: PviCNNTransformer(SHAPES, projection_dim=20, cnn_depth=2, mlp_depth=2),
}


@pytest.mark.parametrize("name", list(ARCHES))
def test_forward_is_core_then_readout(name):
    model = ARCHES[name]().eval()
    seqs, stats, _ = _batch()
    with torch.no_grad():
        full = model(seqs, stats)
        composed = model.forward_readout(model.forward_core(seqs, stats), stats)
    assert full.shape == (B, OUT)
    assert torch.allclose(full, composed, atol=1e-6)
    assert isinstance(model.feature_size, int) and model.feature_size > 0


@pytest.mark.parametrize("name", list(ARCHES))
def test_core_has_named_state(name):
    """core/readout are real submodules => checkpoint keys are namespaced."""
    model = ARCHES[name]()
    keys = list(model.state_dict())
    assert any(k.startswith("core.") for k in keys) or isinstance(model.core, torch.nn.Identity)
    assert any(k.startswith("readout.") for k in keys)


def test_transfer_core_copies_and_freezes():
    src = PviCNN(SHAPES, num_conv_layers=2)
    dst = PviCNN(SHAPES, num_conv_layers=2)
    transfer_core(src, dst, freeze=True)

    assert all(not p.requires_grad for p in dst.core.parameters())
    assert all(p.requires_grad for p in dst.readout.parameters())
    for k, v in src.core.state_dict().items():
        assert torch.equal(v, dst.core.state_dict()[k])


def test_load_core_from_full_state_dict():
    src = PviCNN(SHAPES, num_conv_layers=2)
    dst = PviCNN(SHAPES, num_conv_layers=2)
    load_core_from_state_dict(dst, src.state_dict(), freeze=True)
    for k in src.core.state_dict():
        assert torch.equal(src.core.state_dict()[k], dst.core.state_dict()[k])


def test_frozen_core_unchanged_only_readout_trains():
    model = PviCNN(SHAPES, num_conv_layers=2)
    model.freeze_core()

    # Snapshot core *parameters* (BatchNorm running-stat buffers legitimately
    # update in train mode even when params are frozen, so we don't assert on them).
    core_params_before = {n: p.detach().clone() for n, p in model.core.named_parameters()}
    readout_before = copy.deepcopy(model.readout.state_dict())

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    loss_fn = MorphologyLoss(base_loss=torch.nn.MSELoss(), base_weight=0.2)

    seqs, stats, targets = _batch(seed=3)
    for _ in range(3):
        loss = loss_fn(model(seqs, stats), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # frozen core params received no gradient and did not move
    assert all(p.grad is None for p in model.core.parameters())
    for n, p in model.core.named_parameters():
        assert torch.equal(p, core_params_before[n])
    # readout moved
    changed = any(not torch.equal(v, readout_before[k]) for k, v in model.readout.state_dict().items())
    assert changed


@pytest.mark.parametrize("name", list(ARCHES))
def test_aux_head_runs_on_core(name):
    model = ARCHES[name]()
    model.add_aux_head("maneuver", out_features=3)
    seqs, stats, _ = _batch()
    out = model(seqs, stats, head="maneuver")
    assert out.shape == (B, 3)
    # unknown head raises
    with pytest.raises(KeyError):
        model(seqs, stats, head="does_not_exist")
