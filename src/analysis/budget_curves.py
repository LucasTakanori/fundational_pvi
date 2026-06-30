"""Aggregate and plot data-efficiency / foundation-vs-individual curves.

Turns per-run metric records (one per subject x budget x seed x method) into
mean +/- sem curves per method, for the Exp A/B figures in PLAN.md. Metrics are
computed with the repo's existing functions (`src/models/perf_metrics.py`):
  * cc_abs  -> bp_accuracy     (primary; Pearson r of SBP/DBP)
  * amae/armse -> metrics_waveform

A "record" is a flat dict, e.g.:
    {"method": "foundation_U", "subject": "subject013", "budget_min": 8.0,
     "seed": 0, "cc_abs": 0.71, "amae": 6.2, "armse": 8.1}
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.models.perf_metrics import bp_accuracy, metrics_waveform

DEFAULT_METRICS = ("cc_abs", "amae", "armse")


def compute_run_metrics(predictions, targets) -> dict[str, float]:
    """Metrics for one run, reusing perf_metrics (predictions/targets: (B, L))."""
    out = {"cc_abs": bp_accuracy(predictions, targets)}
    out.update(metrics_waveform(predictions, targets))  # amae, armse
    return out


def load_records(path: str) -> list[dict]:
    """Load run records from a .jsonl (one JSON object per line) or .json list."""
    with open(path) as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return list(json.load(f))


def aggregate_budget_results(records: list[dict] | pd.DataFrame,
                             x: str = "budget_min",
                             group: str = "method",
                             metrics=DEFAULT_METRICS) -> pd.DataFrame:
    """Mean / std / sem / n per (method, budget) for each metric (long form)."""
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=[group, x, "metric", "mean", "std", "sem", "n"])

    present = [m for m in metrics if m in df.columns]
    rows = []
    for (g, xv), sub in df.groupby([group, x]):
        for m in present:
            vals = sub[m].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            n = len(vals)
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            rows.append({group: g, x: xv, "metric": m,
                         "mean": float(vals.mean()) if n else float("nan"),
                         "std": std,
                         "sem": std / np.sqrt(n) if n > 1 else 0.0,
                         "n": n})
    return pd.DataFrame(rows).sort_values([group, "metric", x]).reset_index(drop=True)


def plot_budget_curves(agg: pd.DataFrame,
                       metric: str = "cc_abs",
                       x: str = "budget_min",
                       group: str = "method",
                       out_path: str = None,
                       ylabel: str = None):
    """Plot mean +/- sem vs budget, one line per method. Returns the Figure.

    Requires matplotlib (optional dependency); raises a clear error if missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("plot_budget_curves requires matplotlib (pip install matplotlib).") from e

    sub = agg[agg["metric"] == metric]
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, g in sub.groupby(group):
        g = g.sort_values(x)
        ax.plot(g[x], g["mean"], marker="o", label=str(method))
        ax.fill_between(g[x], g["mean"] - g["sem"], g["mean"] + g["sem"], alpha=0.2)

    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(f"{metric} vs {x}")
    ax.legend()
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
