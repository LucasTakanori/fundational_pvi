"""Exp A/B/G CLI — data-efficiency curves (individual vs foundation transfer)."""

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
from src.foundation.transfer import build_subject_dataset


def _model_kwargs(cfg: FoundationConfig) -> dict:
    return foundation_kwargs(cfg)


def _apply_calibration_cfg(cfg: FoundationConfig, mechanism: str) -> FoundationConfig:
    cfg = FoundationConfig(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})
    if mechanism == "linear":
        cfg.readout_hidden = 0
    elif mechanism == "mlp":
        if cfg.readout_hidden <= 0:
            cfg.readout_hidden = 64
    return cfg


def _foundation_factory(core_path: str | Path, subject: str, cfg: FoundationConfig,
                        load_pretrained: bool = True):
    def factory(shapes):
        model = build_foundation_model(shapes, cfg)
        if load_pretrained:
            state = torch.load(core_path, map_location="cpu")
            model.load_core_state_dict(state, freeze=True)
        else:
            for p in model.core.parameters():
                p.requires_grad_(False)
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
            calibration: str = "linear",
            matched_capacity: bool = False,
            partial_finetune_layers: int = 1,
            ds_root=None,
            device=None) -> list[dict]:
    cfg = _apply_calibration_cfg(cfg, calibration)
    ds = build_subject_dataset(cfg, subject, ds_root=ds_root)
    factories: dict[str, callable] = {
        "individual": _individual_factory(subject, cfg),
    }
    if core_s:
        factories["foundation_S"] = _foundation_factory(core_s, subject, cfg, load_pretrained=True)
        if matched_capacity:
            factories["matched_capacity_S"] = _foundation_factory(
                core_s, subject, cfg, load_pretrained=False,
            )
    if core_u:
        factories["foundation_U"] = _foundation_factory(core_u, subject, cfg, load_pretrained=True)
        if matched_capacity:
            factories["matched_capacity_U"] = _foundation_factory(
                core_u, subject, cfg, load_pretrained=False,
            )

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
        calibration=calibration,
        partial_finetune_layers=partial_finetune_layers,
    )

    pm = ProjectPathManager(branch=cfg.branch, target=logdir)
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
    p.add_argument("--seconds-per-sequence", type=float, default=10.0)
    p.add_argument("--epochs-per-budget", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--arch", default="crt", help="Foundation architecture tag.")
    p.add_argument("--branch", default="main",
                   help="TrainingBranch for subject lookup (use holdout only for final eval).")
    p.add_argument("--calibration-mechanism", default="linear",
                   choices=["linear", "mlp", "affine", "partial_finetune"],
                   help="Calibration mechanism per PLAN.md §3.2.")
    p.add_argument("--readout-hidden", type=int, default=0,
                   help="Readout MLP hidden size (used when mechanism=mlp).")
    p.add_argument("--partial-finetune-layers", type=int, default=1)
    p.add_argument("--matched-capacity-control", action="store_true",
                   help="Add random-init frozen-core control (§8.1).")
    p.add_argument("--logdir", default=None)
    p.add_argument("--ds-root", default=None)
    p.add_argument("--no-core-s", action="store_true")
    p.add_argument("--no-core-u", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = FoundationConfig(
        batch_size=args.batch_size,
        arch=args.arch,
        branch=args.branch,
        readout_hidden=args.readout_hidden,
    )
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
            calibration=args.calibration_mechanism,
            matched_capacity=args.matched_capacity_control,
            partial_finetune_layers=args.partial_finetune_layers,
            ds_root=args.ds_root,
        )
    except FileNotFoundError as e:
        print(f"[budget_exp] {e}")
