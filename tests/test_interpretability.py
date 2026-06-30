"""Tests for the interpretability / digital-twin utilities (Exp D/E)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.foundation import PviFoundationModel
from src.analysis.interpretability import (
    input_sensitivity, gradient_saliency, extract_latents,
    readout_weight_matrix, functional_barcode_probe,
)

B, C, T, OUT, F = 4, 3, 50, 8, 16
SHAPES = {"input": (C, T), "output": (OUT,), "stats": (1,)}


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    seqs = {"pviHP": torch.randn(B, C, T, generator=g),
            "pviLP": torch.randn(B, C, T, generator=g)}
    stats = torch.randn(B, 1, generator=g)
    return seqs, stats


def _model():
    return PviFoundationModel(SHAPES, num_features=F, num_hidden_layers=2, verbose=False)


def test_input_sensitivity_per_channel():
    seqs, stats = _inputs()
    sens = input_sensitivity(_model(), seqs, stats, eps=0.5)
    assert set(sens) == {"pviHP", "pviLP"}
    assert len(sens["pviHP"]) == C and len(sens["pviLP"]) == C
    assert all(v >= 0 and np.isfinite(v) for v in sens["pviHP"])
    assert any(v > 0 for v in sens["pviHP"])  # model responds to perturbation


def test_gradient_saliency_shapes():
    seqs, stats = _inputs()
    sal = gradient_saliency(_model(), seqs, stats)
    assert sal["pviHP"].shape == (C,) and sal["pviLP"].shape == (C,)
    assert torch.all(torch.isfinite(sal["pviHP"]))


class _LabeledDataset(Dataset):
    def __init__(self, n=12):
        g = torch.Generator().manual_seed(0)
        self.items = [{
            "pviHP": torch.randn(C, T, generator=g),
            "pviLP": torch.randn(C, T, generator=g),
            "bp": torch.randn(OUT, generator=g),
            "stats": torch.randn(1, generator=g),
            "maneuver": torch.tensor(i % 3),
        } for i in range(n)]

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def test_extract_latents_shapes_and_labels():
    loader = DataLoader(_LabeledDataset(12), batch_size=4)
    Z, L = extract_latents(_model(), loader, label_keys=("maneuver",))
    assert Z.shape == (12, F)
    assert "maneuver" in L and L["maneuver"].shape == (12,)


def test_readout_weight_matrix():
    m = _model()
    for s in ("subject001", "subject002", "subject003"):
        m.add_readout(s)
    X, subs = readout_weight_matrix(m, subjects=["subject001", "subject002", "subject003"])
    assert X.shape[0] == 3 and X.ndim == 2
    assert subs == ["subject001", "subject002", "subject003"]


def test_functional_barcode_probe_separable():
    # two clearly separable clusters -> probe recovers the label
    rng = np.random.default_rng(0)
    n = 40
    X = np.vstack([rng.normal(-3, 0.5, (n, 8)), rng.normal(3, 0.5, (n, 8))])
    y = np.array([0] * n + [1] * n)
    acc = functional_barcode_probe(X, y, test_size=0.3, seed=0)
    assert acc > 0.9
