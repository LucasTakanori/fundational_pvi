"""Tests for the learned EIT reconstruction sub-track (src/models/eit_recon.py)."""

import pytest
import torch

from src.models.eit_recon import (
    EITReconstructor, EITForwardOperator,
    circular_fov_mask, reconstruction_loss, data_consistency_loss,
)

B, C, T, H = 3, 64, 6, 16


def test_circular_fov_mask():
    m = circular_fov_mask(H)
    assert m.shape == (H, H) and m.dtype == torch.bool
    assert bool(m[H // 2, H // 2])           # centre is inside
    assert not bool(m[0, 0])                  # corner is outside the disc
    assert 0 < m.sum().item() < H * H


def test_reconstructor_shapes_and_backward():
    model = EITReconstructor(in_channels=C, img_size=H, hidden=64, num_hidden_layers=1)
    x = torch.randn(B, C, T, requires_grad=True)
    img = model(x)
    assert img.shape == (B, 1, H, H, T)
    img.pow(2).mean().backward()
    assert x.grad is not None


def test_reconstructor_fov_zeros_outside_disc():
    model = EITReconstructor(in_channels=C, img_size=H, hidden=64, num_hidden_layers=1, apply_fov=True)
    img = model(torch.randn(B, C, T))
    outside = ~circular_fov_mask(H)
    # every pixel outside the FOV is exactly zero, across batch/time
    assert torch.all(img[:, :, outside, :] == 0)


def test_forward_operator_shapes():
    op = EITForwardOperator(img_size=H, num_measurements=C, learnable=True)
    img = torch.randn(B, 1, H, H, T)
    v = op(img)
    assert v.shape == (B, C, T)
    # fixed physics operator: pass a measurement matrix, params frozen
    W = torch.randn(C, H * H)
    op_fixed = EITForwardOperator(img_size=H, num_measurements=C, weight=W, learnable=False)
    assert not any(p.requires_grad for p in op_fixed.parameters())
    assert torch.allclose(op_fixed.weight, W)
    with pytest.raises(ValueError):
        EITForwardOperator(img_size=H, num_measurements=C, weight=torch.randn(C, 5), learnable=False)


def test_losses_finite_and_fov_weighted():
    pred = torch.randn(B, 1, H, H, T)
    target = torch.randn(B, 1, H, H, T)
    fov = circular_fov_mask(H)
    assert torch.isfinite(reconstruction_loss(pred, target))
    assert torch.isfinite(reconstruction_loss(pred, target, fov=fov))
    # masked loss ignores outside-FOV error: perfect inside-disc match -> ~0
    matched = target.clone()
    matched[:, :, ~fov, :] += 5.0           # corrupt only outside the FOV
    assert reconstruction_loss(matched, target, fov=fov).item() == pytest.approx(0.0, abs=1e-6)

    op = EITForwardOperator(img_size=H, num_measurements=C)
    measured = torch.randn(B, C, T)
    assert torch.isfinite(data_consistency_loss(pred, measured, op))


def test_training_step_reduces_reconstruction_loss():
    torch.manual_seed(0)
    model = EITReconstructor(in_channels=C, img_size=H, hidden=64, num_hidden_layers=1, apply_fov=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randn(B, C, T)
    target = torch.randn(B, 1, H, H, T)
    first = last = None
    for i in range(15):
        loss = reconstruction_loss(model(x), target)
        opt.zero_grad(); loss.backward(); opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
