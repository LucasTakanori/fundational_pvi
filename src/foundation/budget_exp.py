"""Exp A/B/G CLI — data-efficiency curves (individual vs foundation transfer).

Example (Exp B on one subject):

    python -m src.foundation.budget_exp --subject subject013 \\
        --core-s artifacts/foundation-pretrain/main/foundation_core.pt \\
        --core-u artifacts/foundation-ssl-pretrain/main/foundation_core_U.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.packages import *  # noqa: F401,F403

from src.pipeline.data_discovery import ProjectPathManager
from src.utils.primitives import DEFAULT_TRAIN_DEVICE
from src.analysis.budget_curves import aggregate_budget_results
from src.foundation.config import FoundationConfig
from src.foundation.experiments import run_budget_curve
from src.foundation.model_factory import build_foundation_model, foundation_kwargs
from src.foundation.foundation_model import PviFoundationModel
from src.foundation.transfer import build_subject_dataset


def _model_kwargs(cfg: FoundationConfig) -> dict:
    return foundation_kwargs(cfg)


def _foundation_factory(core_path: str | Path, subject: str, cfg: FoundationConfig):
    def factory(shapes):
        model = build_foundation_model(shapes, cfg)
        state = torch.load(core_path, map_location="cpu")
        model.load_core_state_dict(state, freeze=True)
        model.add_readout(subject)
        model.set_active(subject)
        return model
    return factory


def _individual_factory(subject: str, cfg: FoundationConfig):
    def factory(shapes):
        model = build_foundation_model(shapes, cfg)
        model.add_readout(subject)
        model.set_active(subject)
        return model
    return factory


def run_exp(subject: str,
            cfg: FoundationConfig,
            budgets_min: list[float],
            seeds: list[int],
            seconds_per_sequence: float,
            epochs_per_budget: int,
            core_s: str | Path | None,
            core_u: str | Path | None,
            logdir: str,
            ds_root=None,
            device=None) -> list[dict]:
    ds = build_subject_dataset(cfg, subject, ds_root=ds_root)
    factories: dict[str, callable] = {
        "individual": _individual_factory(subject, cfg),
    }
    if core_s:
        factories["foundation_S"] = _foundation_factory(core_s, subject, cfg)
    if core_u:
        factories["foundation_U"] = _foundation_factory(core_u, subject, cfg)

    device = device or DEFAULT_TRAIN_DEVICE
    records = run_budget_curve(
        ds,
        factories,
        budgets_min=budgets_min,
        seconds_per_sequence=seconds_per_sequence,
        seeds=seeds,
        epochs=epochs_per_budget,
        device=device,
        subject=subject,
    )

    pm = ProjectPathManager(branch="main", target=logdir)
    out_dir = pm.logdir
    out_dir.mkdir(parents=True, exist_ok=True)

    records_path = out_dir / "budget_records.json"
    with open(records_path, "w") as f:
        json.dump(records, f, indent=2)

    agg = aggregate_budget_results(records)
    agg_path = out_dir / "budget_aggregate.csv"
    agg.to_csv(agg_path, index=False)

    print(f"[budget_exp] wrote {records_path} ({len(records)} records)", flush=True)
    print(f"[budget_exp] wrote {agg_path}", flush=True)
    return records


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--core-s", default="artifacts/foundation-pretrain/main/foundation_core.pt",
                   help="Supervised cohort core (foundation_S). Omit to skip.")
    p.add_argument("--core-u", default="artifacts/foundation-ssl-pretrain/main/foundation_core_U.pt",
                   help="SSL cohort core (foundation_U). Omit to skip.")
    p.add_argument("--budgets-min", default="4,8,16,32,64",
                   help="Comma-separated training budgets in minutes.")
    p.add_argument("--seeds", default="0", help="Comma-separated RNG seeds.")
    p.add_argument("--seconds-per-sequence", type=float, default=10.0,
                   help="Wall-clock seconds per training sequence (budget → n_seq).")
    p.add_argument("--epochs-per-budget", type=int, default=50,
                   help="Training epochs at each budget point.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--arch", default="crt", help="Foundation architecture tag.")
    p.add_argument("--logdir", default=None,
                   help="Default: exp-b-{subject} under artifacts/.")
    p.add_argument("--ds-root", default=None)
    p.add_argument("--no-core-s", action="store_true", help="Skip foundation_S.")
    p.add_argument("--no-core-u", action="store_true", help="Skip foundation_U.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = FoundationConfig(batch_size=args.batch_size, arch=args.arch)
    logdir = args.logdir or f"exp-b-{args.subject}"
    core_s = None if args.no_core_s else args.core_s
    core_u = None if args.no_core_u else args.core_u
    try:
        run_exp(
            subject=args.subject,
            cfg=cfg,
            budgets_min=_parse_float_list(args.budgets_min),
            seeds=_parse_int_list(args.seeds),
            seconds_per_sequence=args.seconds_per_sequence,
            epochs_per_budget=args.epochs_per_budget,
            core_s=core_s,
            core_u=core_u,
            logdir=logdir,
            ds_root=args.ds_root,
        )
    except FileNotFoundError as e:
        print(f"[budget_exp] {e}")
