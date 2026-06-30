"""Compile/shape tests for the MAE-style masked Transformer core (candidate 2)."""

import torch

from src.models.mae_transformer import PviMaskedTransformer
from src.models._model_mapper import ml_session_mapper

B, OUT = 3, 8
SHAPES_1D = {"input": (2, 40), "output": (OUT,), "stats": (1,)}
SHAPES_IMG = {"input": (1, 6, 6, 12), "output": (OUT,), "stats": (2,)}


def _batch(shapes, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs = (B,) + shapes["input"]
    return {"pviHP": torch.randn(*xs, generator=g),
            "pviLP": torch.randn(*xs, generator=g),
            "bp": torch.randn(B, OUT, generator=g),
            "stats": torch.randn(B, shapes["stats"][0], generator=g)}


def _check(shapes):
    model = PviMaskedTransformer(shapes, d_model=16, num_layers=1, mlp_depth=2, verbose=False).eval()
    seqs, stats, targets = model.process_batch(_batch(shapes))
    with torch.no_grad():
        full = model(seqs, stats)
        comp = model.forward_readout(model.forward_core(seqs, stats), stats)
    assert full.shape == (B, OUT)
    assert torch.allclose(full, comp, atol=1e-6)
    assert model.feature_size == model.d_model * shapes["input"][-1]
    return model


def test_forward_1d_and_image():
    _check(SHAPES_1D)
    _check(SHAPES_IMG)


def test_backward_and_freeze_contract():
    model = PviMaskedTransformer(SHAPES_1D, d_model=16, num_layers=1, verbose=False)
    seqs, stats, targets = model.process_batch(_batch(SHAPES_1D))
    (model(seqs, stats) - targets).pow(2).mean().backward()
    assert any(p.grad is not None for p in model.parameters())

    model.freeze_core()
    assert all(not p.requires_grad for p in model.core.parameters())
    assert all(p.requires_grad for p in model.readout.parameters())


def test_registered_in_mapper():
    model_cls, _, _ = ml_session_mapper("ps20-mae-signal-to-waveform")
    assert model_cls is PviMaskedTransformer
