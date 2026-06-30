"""Tests for the experiment driver (Exp A/B orchestration)."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.foundation import PviFoundationModel
from src.foundation.experiments import train_model, run_budget_curve
from src.analysis.budget_curves import aggregate_budget_results

C, T, OUT = 1, 50, 8
SHAPES = {"input": (C, T), "output": (OUT,), "stats": (1,)}


class _SamplesDataset(Dataset):
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


class _DatasetStub:
    """Minimal duck-typed dataset: set_train_budget / get_dataloaders / shapes."""
    def __init__(self):
        self._train = _SamplesDataset(16, seed=0)
        self._test = _SamplesDataset(8, seed=1)
        self.last_budget = None

    def set_train_budget(self, minutes=None, seconds_per_sequence=None, seed=0):
        self.last_budget = (minutes, seconds_per_sequence, seed)  # record the call

    def get_dataloaders(self):
        return {"train": DataLoader(self._train, batch_size=4),
                "test": DataLoader(self._test, batch_size=4)}

    @property
    def shapes(self):
        return SHAPES


def test_train_model_updates_only_trainable():
    model = PviFoundationModel(SHAPES, num_features=16, num_hidden_layers=2, verbose=False)
    model.freeze_core()
    core_before = {n: p.detach().clone() for n, p in model.core.named_parameters()}

    loader = DataLoader(_SamplesDataset(8), batch_size=4)
    train_model(model, loader, epochs=1, lr=1e-2)

    for n, p in model.core.named_parameters():
        assert torch.equal(p, core_before[n])  # frozen core untouched


def test_run_budget_curve_record_structure():
    ds = _DatasetStub()
    factories = {
        "individual": lambda s: PviFoundationModel(s, num_features=16, num_hidden_layers=2, verbose=False),
        "foundation": lambda s: PviFoundationModel(s, num_features=16, num_hidden_layers=2, verbose=False).freeze_core(),
    }
    records = run_budget_curve(ds, factories, budgets_min=[4.0, 8.0],
                               seconds_per_sequence=60.0, seeds=[0], epochs=1, subject="subject001")

    # 2 budgets x 1 seed x 2 methods = 4 records
    assert len(records) == 4
    for r in records:
        assert set(r) >= {"method", "budget_min", "seed", "subject", "cc_abs", "amae", "armse"}
        assert r["subject"] == "subject001"
        assert np.isfinite(r["cc_abs"])
    assert ds.last_budget == (8.0, 60.0, 0)  # budget control was driven

    # records flow straight into the aggregation/curve API
    agg = aggregate_budget_results(records)
    assert set(agg["method"]) == {"individual", "foundation"}
    assert set(agg["budget_min"]) == {4.0, 8.0}
