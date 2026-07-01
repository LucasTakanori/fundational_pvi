"""Tests for shape/bias decomposition."""

import torch

from src.analysis.decomposition import (
    apply_affine_correction,
    decompose_predictions,
    fit_affine_correction,
    per_subject_bias,
)


def test_affine_correction_closes_bias():
    targets = torch.linspace(60, 120, 50).unsqueeze(0)
    preds = targets * 2.0 + 10.0
    scale, offset = fit_affine_correction(preds, targets)
    corrected = apply_affine_correction(preds, scale, offset)
    assert abs(per_subject_bias(corrected, targets)) < 1.0


def test_decompose_predictions_keys():
    preds = torch.randn(4, 50) * 10 + 100
    targets = torch.randn(4, 50) * 10 + 100
    out = decompose_predictions(preds, targets)
    for key in ("cc_abs", "waveform_r", "bias_mmhg", "amae", "corrected_amae"):
        assert key in out
