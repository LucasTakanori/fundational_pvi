"""Tests for gradient-reversal subject adversary."""

import torch

from src.foundation.adversarial import (
    GradientReversalFunction,
    GradientReversalLayer,
    SubjectAdversaryHead,
    dann_lambda_schedule,
)
from src.foundation.core import PviCore


def test_grl_identity_forward():
    x = torch.randn(4, 32, requires_grad=True)
    y = GradientReversalFunction.apply(x, 0.5)
    assert torch.allclose(x, y)


def test_grl_flips_gradient_sign():
    x = torch.randn(4, 32, requires_grad=True)
    y = GradientReversalLayer(lambda_=0.7)(x)
    loss = y.sum()
    loss.backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.7))


def test_dann_lambda_schedule_monotonic_bounded():
    vals = [dann_lambda_schedule(p / 10.0) for p in range(11)]
    assert vals[0] == 0.0
    assert all(0.0 <= v < 1.0 for v in vals)
    assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))


def test_subject_adversary_head_logits_shape():
    head = SubjectAdversaryHead(feature_size=64, num_subjects=100, lambda_=0.3)
    x = torch.randn(8, 64)
    logits = head(x)
    assert logits.shape == (8, 100)
