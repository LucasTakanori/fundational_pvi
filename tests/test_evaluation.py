"""Tests for the evaluation primitives (Exp A/B/C/G scoring)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.foundation import PviFoundationModel
from src.foundation.evaluation import collect_predictions, evaluate, evaluate_ood

C, T, OUT = 1, 50, 8
SHAPES = {"input": (C, T), "output": (OUT,), "stats": (1,)}


class _SyntheticDataset(Dataset):
    """Emits per-sample dicts; default collate stacks them into batches."""
    def __init__(self, n=12, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = [{
            "pviHP": torch.randn(C, T, generator=g),
            "pviLP": torch.randn(C, T, generator=g),
            "bp": torch.randn(OUT, generator=g),
            "stats": torch.randn(1, generator=g),
        } for _ in range(n)]

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _loader(n=12, seed=0, batch_size=4):
    return DataLoader(_SyntheticDataset(n, seed), batch_size=batch_size)


def _model():
    return PviFoundationModel(SHAPES, num_features=16, num_hidden_layers=2, verbose=False)


def test_collect_predictions_shapes():
    preds, targets = collect_predictions(_model(), _loader(n=12, batch_size=4))
    assert preds.shape == (12, OUT)
    assert targets.shape == (12, OUT)


def test_evaluate_returns_metrics():
    metrics = evaluate(_model(), _loader())
    assert set(metrics) >= {"cc_abs", "amae", "armse"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_evaluate_extra_fields_passthrough():
    metrics = evaluate(_model(), _loader(), extra={"method": "foundation_U", "budget_min": 8.0})
    assert metrics["method"] == "foundation_U" and metrics["budget_min"] == 8.0


def test_evaluate_ood_tags():
    metrics = evaluate_ood(_model(), _loader(), maneuver="valsalva")
    assert metrics["maneuver"] == "valsalva" and metrics["ood"] is True
    assert "cc_abs" in metrics


def test_empty_loader_raises():
    empty = DataLoader(_SyntheticDataset(0))
    with pytest.raises(ValueError):
        collect_predictions(_model(), empty)
