"""Shape/bias decomposition, affine correction, and label-gap diagnostics (PLAN.md §3.3)."""

from __future__ import annotations

import numpy as np
import torch
from scipy import stats

from src.models.perf_metrics import bp_accuracy, metrics_waveform


def waveform_correlation(predictions, targets) -> float:
    """Mean Pearson r over samples (full waveform, not min/max only)."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if predictions.shape[0] < 2:
        return 0.0
    rs = []
    for i in range(predictions.shape[0]):
        r = stats.pearsonr(predictions[i].flatten(), targets[i].flatten()).statistic
        if np.isfinite(r):
            rs.append(float(r))
    return float(np.mean(rs)) if rs else 0.0


def per_subject_bias(predictions, targets) -> float:
    """Mean signed error (predicted - true), mmHg."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    return float(np.mean(predictions - targets))


def fit_affine_correction(predictions, targets) -> tuple[float, float]:
    """Fit scale + offset mapping pred -> true (least squares, 2 params)."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy().reshape(-1)
    else:
        predictions = np.asarray(predictions).reshape(-1)
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy().reshape(-1)
    else:
        targets = np.asarray(targets).reshape(-1)
    if len(predictions) < 2:
        return 1.0, 0.0
    A = np.vstack([predictions, np.ones_like(predictions)]).T
    scale, offset = np.linalg.lstsq(A, targets, rcond=None)[0]
    return float(scale), float(offset)


def apply_affine_correction(predictions, scale: float, offset: float):
    if isinstance(predictions, torch.Tensor):
        return predictions * scale + offset
    return np.asarray(predictions) * scale + offset


def compute_label_gap(subject_targets, cohort_targets) -> float:
    """Model-independent distribution shift: Wasserstein-1 on flattened BP values."""
    if isinstance(subject_targets, torch.Tensor):
        subject_targets = subject_targets.detach().cpu().numpy().reshape(-1)
    else:
        subject_targets = np.asarray(subject_targets).reshape(-1)
    if isinstance(cohort_targets, torch.Tensor):
        cohort_targets = cohort_targets.detach().cpu().numpy().reshape(-1)
    else:
        cohort_targets = np.asarray(cohort_targets).reshape(-1)
    return float(stats.wasserstein_distance(subject_targets, cohort_targets))


def decompose_predictions(predictions, targets, cohort_targets=None) -> dict[str, float]:
    """Shape/bias decomposition plus optional affine-corrected accuracy."""
    out = {
        "cc_abs": bp_accuracy(predictions, targets),
        "waveform_r": waveform_correlation(predictions, targets),
        "bias_mmhg": per_subject_bias(predictions, targets),
    }
    out.update(metrics_waveform(predictions, targets))

    scale, offset = fit_affine_correction(predictions, targets)
    corrected = apply_affine_correction(predictions, scale, offset)
    out["affine_scale"] = scale
    out["affine_offset"] = offset
    out.update({
        f"corrected_{k}": v
        for k, v in metrics_waveform(
            torch.as_tensor(corrected) if not isinstance(corrected, torch.Tensor) else corrected,
            targets,
        ).items()
    })
    out["corrected_cc_abs"] = bp_accuracy(corrected, targets)

    if cohort_targets is not None:
        out["label_gap"] = compute_label_gap(targets, cohort_targets)
    return out
