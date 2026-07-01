"""Tests for production architecture wiring (foundation model factory)."""

import torch

from src.foundation.config import FoundationConfig
from src.foundation.model_factory import build_foundation_model, build_ssl_model
from src.foundation import PviFoundationModel, PviSSLModel
from src.models.loss_functions import MorphologyLoss

B, OUT = 4, 50
# impedance-like: 64 channels (32 R + 32 X), diff=2 -> 192 effective channels in model
SHAPES_IMP = {"input": (64, 250), "output": (OUT,), "stats": (2, 5)}


def _batch(shapes, seed=0):
    g = torch.Generator().manual_seed(seed)
    c, t = shapes["input"]
    return {
        "pviHP": torch.randn(B, c, t, generator=g),
        "pviLP": torch.randn(B, c, t, generator=g),
        "bp": torch.randn(B, OUT, generator=g),
        "stats": torch.randn(B, *shapes["stats"], generator=g),
    }


def _train_readout_one_step(model, shapes):
    model.add_readout("subject001")
    model.set_active("subject001")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    loss_fn = MorphologyLoss()
    batch = _batch(shapes)
    seqs, stats, targets = model.process_batch(batch)
    loss = loss_fn(model(seqs, stats), targets)
    loss.backward()
    opt.step()
    return float(loss)


def test_foundation_crt_forward_and_transfer_readout():
    cfg = FoundationConfig(arch="crt", input_mode="impedance")
    model = build_foundation_model(SHAPES_IMP, cfg)
    assert isinstance(model, PviFoundationModel)
    assert model.arch == "crt"
    model.set_active("shared")
    batch = _batch(SHAPES_IMP)
    seqs, stats, _ = model.process_batch(batch)
    out = model(seqs, stats)
    assert out.shape == (B, OUT)
    _train_readout_one_step(model, SHAPES_IMP)


def test_foundation_mae_forward():
    cfg = FoundationConfig(arch="mae", input_mode="impedance")
    model = build_foundation_model(SHAPES_IMP, cfg)
    assert model.arch == "mae"
    model.set_active("shared")
    seqs, stats, _ = model.process_batch(_batch(SHAPES_IMP))
    assert model(seqs, stats).shape == (B, OUT)


def test_ssl_mae_pretext_loss_finite():
    cfg = FoundationConfig(arch="mae", ssl_arch="mae")
    model = build_ssl_model(SHAPES_IMP, cfg)
    assert isinstance(model, PviSSLModel)
    losses = model.pretext_loss(_batch(SHAPES_IMP))
    assert all(torch.isfinite(v) for v in losses.values())


def test_ssl_mae_core_loads_into_foundation_mae():
    cfg = FoundationConfig(arch="mae", ssl_arch="mae")
    ssl = build_ssl_model(SHAPES_IMP, cfg)
    fnd = build_foundation_model(SHAPES_IMP, cfg)
    fnd.load_core_state_dict(ssl.core_state_dict(), freeze=True)
    for key in ssl.core.state_dict():
        assert torch.equal(ssl.core.state_dict()[key], fnd.core.state_dict()[key])
