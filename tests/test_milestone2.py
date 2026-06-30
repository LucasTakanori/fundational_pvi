"""Milestone 2 tests: training-budget control + metrics/plot aggregation."""

import math

import numpy as np
import pytest
import torch

from src.pipeline._data_preparation import subsample_budget, PviConfiguredDataset
from src.analysis.budget_curves import (
    compute_run_metrics, aggregate_budget_results, plot_budget_curves,
)

MASKS = [(i, i + 4) for i in range(1, 101, 5)]  # 20 sequences


# --------------------------------------------------------------- subsample_budget
def test_subsample_by_count():
    out = subsample_budget(MASKS, n_seq=5, seed=0)
    assert len(out) == 5
    assert set(out).issubset(set(MASKS))          # subset of the originals
    assert len(set(out)) == 5                       # no duplicates
    assert out == sorted(out, key=MASKS.index)      # original order preserved


def test_subsample_caps_and_floors():
    assert len(subsample_budget(MASKS, n_seq=999, seed=0)) == len(MASKS)
    assert subsample_budget(MASKS, n_seq=0, seed=0) == []


def test_subsample_minutes_to_count():
    # 2 min at 60 s/seq -> ceil(120/60) = 2 sequences
    out = subsample_budget(MASKS, minutes=2.0, seconds_per_sequence=60.0, seed=1)
    assert len(out) == 2


def test_subsample_reproducible_and_seed_sensitive():
    a = subsample_budget(MASKS, n_seq=6, seed=42)
    b = subsample_budget(MASKS, n_seq=6, seed=42)
    c = subsample_budget(MASKS, n_seq=6, seed=7)
    assert a == b
    assert a != c  # extremely unlikely to coincide for 6/20


def test_subsample_arg_validation():
    with pytest.raises(ValueError):
        subsample_budget(MASKS)                       # neither
    with pytest.raises(ValueError):
        subsample_budget(MASKS, n_seq=5, minutes=2)   # both
    with pytest.raises(ValueError):
        subsample_budget(MASKS, minutes=2)            # minutes w/o seconds_per_sequence


# ----------------------------------------------------- set_train_budget (integration)
class _StubDataset(PviConfiguredDataset):
    """Minimal concrete dataset to exercise set_train_budget end-to-end."""
    def __init__(self):
        self.active_mask = list(MASKS)
        self.train_mask = list(MASKS[:16])
        self.test_mask = list(MASKS[16:])
        self._split_params = {}
        self._VALID_CONFIGS_KEYS = ["input_mode", "output_mode", "mask_key"]
        self.input_mode = self.output_mode = self.mask_key = None

    def build(self): return self
    def get_partition(self): return self.subsets
    def __len__(self): return len(self.active_mask)
    def __getitem__(self, idx): return {"x": idx}  # avoid h5io in the stub


def test_set_train_budget_rebuilds_subsets():
    ds = _StubDataset()
    ds.set_train_budget(n_seq=4, seed=0)
    assert len(ds.train_mask) == 4
    assert len(ds.subsets["train"]) == 4
    assert len(ds.subsets["test"]) == 4               # test set untouched
    assert ds._split_params["budget"]["available"] == 16

    # second call samples from the FULL train partition, not the reduced one
    ds.set_train_budget(n_seq=10, seed=0)
    assert len(ds.train_mask) == 10
    assert len(ds.subsets["train"]) == 10

    ds.reset_train_budget()
    assert len(ds.train_mask) == 16
    assert "budget" not in ds._split_params


# ------------------------------------------------------------- compute_run_metrics
def test_compute_run_metrics_perfect_prediction():
    torch.manual_seed(0)
    targets = torch.randn(16, 50)
    m = compute_run_metrics(targets.clone(), targets)
    assert m["cc_abs"] == pytest.approx(1.0, abs=1e-5)
    assert m["amae"] == pytest.approx(0.0, abs=1e-6)
    assert m["armse"] == pytest.approx(0.0, abs=1e-6)


def test_compute_run_metrics_finite():
    torch.manual_seed(1)
    preds, targets = torch.randn(16, 50), torch.randn(16, 50)
    m = compute_run_metrics(preds, targets)
    assert set(m) == {"cc_abs", "amae", "armse"}
    assert all(np.isfinite(v) for v in m.values())


# --------------------------------------------------------- aggregate_budget_results
def _records():
    recs = []
    for method, base in (("individual", 0.5), ("foundation_U", 0.7)):
        for budget in (4.0, 8.0):
            for seed in (0, 1, 2):
                recs.append({"method": method, "subject": f"s{seed}", "budget_min": budget,
                             "seed": seed, "cc_abs": base + 0.01 * budget, "amae": 7.0, "armse": 9.0})
    return recs


def test_aggregate_groups_and_stats():
    agg = aggregate_budget_results(_records())
    row = agg[(agg["method"] == "foundation_U") & (agg["budget_min"] == 8.0) & (agg["metric"] == "cc_abs")]
    assert len(row) == 1
    assert row["mean"].item() == pytest.approx(0.7 + 0.08)
    assert row["n"].item() == 3
    # methods and metrics all present
    assert set(agg["method"]) == {"individual", "foundation_U"}
    assert set(agg["metric"]) == {"cc_abs", "amae", "armse"}


def test_aggregate_empty():
    agg = aggregate_budget_results([])
    assert list(agg.columns) == ["method", "budget_min", "metric", "mean", "std", "sem", "n"]
    assert agg.empty


# ------------------------------------------------------------------ plot
def test_plot_budget_curves_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    agg = aggregate_budget_results(_records())
    out = tmp_path / "curve.png"
    fig = plot_budget_curves(agg, metric="cc_abs", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0
    assert fig is not None
