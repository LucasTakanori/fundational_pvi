"""Milestone 3 tests: SSL masking/forecast utilities + dual-pretext model."""

import pytest
import torch

from src.foundation import (
    PviSSLModel, PviFoundationModel, random_channel_time_mask, split_past_future,
)

B, C, T, OUT = 4, 2, 40, 8
SHAPES_1D = {"input": (C, T), "output": (OUT,), "stats": (1,)}
SHAPES_IMG = {"input": (1, 8, 8, T), "output": (OUT,), "stats": (1,)}


def _batch(shapes, seed=0):
    g = torch.Generator().manual_seed(seed)
    in_shape = shapes["input"]
    x_shape = (B,) + in_shape
    return {
        "pviHP": torch.randn(*x_shape, generator=g),
        "pviLP": torch.randn(*x_shape, generator=g),
        "bp": torch.randn(B, OUT, generator=g),
        "stats": torch.randn(B, 1, generator=g),
    }


# --------------------------------------------------------- masking / split utils
def test_mask_shapes_and_zeroing():
    x = torch.randn(B, C, T)
    xm, mask = random_channel_time_mask(x, mask_ratio=0.5)
    assert xm.shape == x.shape and mask.shape == x.shape
    assert mask.dtype == torch.bool
    assert torch.all(xm[mask] == 0)           # hidden entries zeroed
    assert torch.equal(xm[~mask], x[~mask])   # visible entries untouched


def test_mask_ratio_is_approximate():
    x = torch.randn(8, 4, 200)
    _, mask = random_channel_time_mask(x, mask_ratio=0.3)
    frac = mask.float().mean().item()
    assert 0.2 < frac < 0.4


def test_mask_broadcasts_over_spatial():
    x = torch.randn(B, 1, 8, 8, T)
    xm, mask = random_channel_time_mask(x, mask_ratio=0.5)
    assert mask.shape == x.shape
    # a masked (channel,time) hides the whole HxW frame -> spatial slices agree
    assert torch.equal(mask[:, :, 0, 0, :][:, :, None, None, :].expand_as(mask), mask)


def test_split_past_future():
    x = torch.randn(B, C, T)
    past, future = split_past_future(x, horizon=10)
    assert past.shape == (B, C, T - 10)
    assert future.shape == (B, C, 10)
    with pytest.raises(ValueError):
        split_past_future(x, horizon=T)


# ---------------------------------------------------------------- SSL model (1D)
def test_ssl_pretext_loss_finite():
    model = PviSSLModel(SHAPES_1D, num_features=16, num_hidden_layers=2, horizon=8, verbose=False)
    losses = model.pretext_loss(_batch(SHAPES_1D))
    assert set(losses) == {"total", "mask", "forecast"}
    assert all(torch.isfinite(v) for v in losses.values())
    assert losses["total"].item() >= 0


def test_ssl_encode_shape():
    model = PviSSLModel(SHAPES_1D, num_features=16, num_hidden_layers=2, verbose=False)
    seqs, stats, _ = model.process_batch(_batch(SHAPES_1D))
    feats = model.encode(seqs, stats)
    assert feats.shape == (B, 16)


def test_ssl_training_step_reduces_loss():
    torch.manual_seed(0)
    model = PviSSLModel(SHAPES_1D, num_features=16, num_hidden_layers=2, horizon=8, verbose=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(0)
    batch = _batch(SHAPES_1D, seed=1)

    first = last = None
    for i in range(15):
        losses = model.pretext_loss(batch, generator=g)
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()
        if i == 0:
            first = losses["total"].item()
        last = losses["total"].item()
    assert last < first


def test_ssl_core_transfers_to_foundation():
    """The SSL-pretrained core loads straight into the supervised foundation model."""
    ssl = PviSSLModel(SHAPES_1D, num_features=16, num_hidden_layers=2, verbose=False)
    fnd = PviFoundationModel(SHAPES_1D, num_features=16, num_hidden_layers=2, verbose=False)

    fnd.load_core_state_dict(ssl.core_state_dict(), freeze=True)
    for k in ssl.core.state_dict():
        assert torch.equal(ssl.core.state_dict()[k], fnd.core.state_dict()[k])
    assert all(not p.requires_grad for p in fnd.core.parameters())


# -------------------------------------------------------------- SSL model (image)
def test_ssl_image_pretext_runs():
    model = PviSSLModel(SHAPES_IMG, num_features=8, num_hidden_layers=1,
                        diff=0, horizon=4, verbose=False)
    losses = model.pretext_loss(_batch(SHAPES_IMG))
    assert all(torch.isfinite(v) for v in losses.values())
