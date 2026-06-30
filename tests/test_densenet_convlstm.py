"""Compile/shape tests for the paper-faithful DenseNet+ConvLSTM core (no training)."""

import pytest
import torch

from src.models.densenet_convlstm import (
    PviDenseNetConvLSTM, CausalConv3d, ConvLSTMCell, SpatialBilinearReadout,
)
from src.foundation import PviFoundationModel  # for transfer-shape note (different core)

B, C, H, T, OUT = 2, 1, 8, 6, 8
SHAPES_IMG = {"input": (C, H, H, T), "output": (OUT,), "stats": (2,)}


def _batch(diff_channels=C):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, C, H, H, T, generator=g)
    return {"pviHP": x, "pviLP": x.clone(),
            "bp": torch.randn(B, OUT, generator=g),
            "stats": torch.randn(B, 2, generator=g)}


def _model(**kw):
    return PviDenseNetConvLSTM(SHAPES_IMG, num_blocks=2, growth=4, layers_per_block=2,
                               hidden_ch=8, mod_ch=4, num_positions=4, readout_hidden=32,
                               verbose=False, **kw)


def test_causal_conv_is_time_causal():
    conv = CausalConv3d(1, 1, k_t=3, k_s=3)
    conv.eval()
    x = torch.randn(1, 1, 4, 4, 5)
    y1 = conv(x)
    x2 = x.clone()
    x2[..., -1] += 10.0                      # perturb only the last time step
    y2 = conv(x2)
    # causal: earlier outputs unchanged by a future-time perturbation
    assert torch.allclose(y1[..., :-1], y2[..., :-1], atol=1e-5)
    assert not torch.allclose(y1[..., -1], y2[..., -1])


def test_convlstm_cell_shapes():
    cell = ConvLSTMCell(3, 5)
    h = torch.zeros(B, 5, 4, 4); c = torch.zeros(B, 5, 4, 4)
    x = torch.randn(B, 3, 4, 4)
    h2, c2 = cell(x, (h, c))
    assert h2.shape == (B, 5, 4, 4) and c2.shape == (B, 5, 4, 4)


def test_spatial_readout_shapes():
    ro = SpatialBilinearReadout(num_positions=4, channels=8, seq_len=T, output_size=OUT, hidden=16)
    fmap = torch.randn(B, 8, 2, 2, T)
    assert ro(fmap).shape == (B, OUT)


def test_forward_and_composition():
    model = _model().eval()
    batch = _batch()
    seqs, stats, _ = model.process_batch(batch)
    with torch.no_grad():
        full = model(seqs, stats)
        comp = model.forward_readout(model.forward_core(seqs, stats), stats)
    assert full.shape == (B, OUT)
    assert torch.allclose(full, comp, atol=1e-6)
    assert model.feature_size == model.hidden_ch * model._feat_hw[0] * model._feat_hw[1] * T


def test_backward_runs():
    model = _model()
    seqs, stats, targets = model.process_batch(_batch())
    loss = (model(seqs, stats) - targets).pow(2).mean()
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_freeze_core_contract():
    model = _model()
    model.freeze_core()
    assert all(not p.requires_grad for p in model.core.parameters())
    assert all(p.requires_grad for p in model.readout.parameters())


def test_requires_image_input():
    with pytest.raises(ValueError):
        PviDenseNetConvLSTM({"input": (1, T), "output": (OUT,), "stats": (2,)}, verbose=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA on this box")
def test_runs_on_gpu():
    model = _model().cuda()
    batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in _batch().items()}
    seqs, stats, _ = model.process_batch(batch)
    assert model(seqs, stats).shape == (B, OUT)
