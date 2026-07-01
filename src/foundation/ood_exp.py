"""Exp C driver — OOD evaluation across maneuvers (PLAN.md Phase 6)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.packages import *  # noqa: F401,F403

from src.utils.primitives import InputMode, OutputMode, SequenceMask, DEFAULT_TRAIN_DEVICE
from src.pipeline.data_discovery import PviDatasetInventory, ProjectPathManager
from src.pipeline.data_preparation_lazy import PviLazyDataset
from src.foundation.config import FoundationConfig
from src.foundation.evaluation import evaluate_ood, collect_predictions
from src.foundation.model_factory import build_foundation_model
from src.foundation.transfer import load_core_meta
from src.analysis.decomposition import decompose_predictions
from src.foundation.experiments import train_model


def _files_for_subject_maneuver(inventory, subject: str, maneuver: str):
    return [f for f in inventory if f.subject == subject and f.session == maneuver]


def build_maneuver_dataset(cfg: FoundationConfig, subject: str, maneuver: str, ds_root=None):
    inventory = PviDatasetInventory(branch=cfg.branch, ds_root=ds_root)
    files = _files_for_subject_maneuver(inventory, subject, maneuver)
    if not files:
        raise FileNotFoundError(
            f"No {maneuver} session for {subject} under {inventory.target_dir}"
        )
    ds = PviLazyDataset(
        ds_files=files,
        input_mode=InputMode(cfg.input_mode),
        output_mode=OutputMode(cfg.output_mode),
        mask_key=SequenceMask(cfg.mask_key),
        max_cache=4,
        persistent_handle=True,
    ).build()
    ds.set_partition(test_size=0.1, shuffle=True, split_mode="within", random_state=cfg.split_seed)
    ds.get_partition()
    ds.set_dataloaders(batch_size=cfg.batch_size, shuffle=False, stratified=False)
    return ds


def run_ood_exp(subject: str,
                core_path: str | Path,
                cfg: FoundationConfig,
                train_maneuver: str = "baseline",
                eval_maneuvers: list[str] | None = None,
                max_epochs: int = 50,
                logdir: str | None = None,
                ds_root=None,
                device=None) -> list[dict]:
    eval_maneuvers = eval_maneuvers or ["valsalva", "pressor"]
    device = device or DEFAULT_TRAIN_DEVICE

    meta = load_core_meta(core_path)
    if meta and meta.get("arch"):
        cfg.arch = meta["arch"]
        if meta.get("input_mode"):
            cfg.input_mode = meta["input_mode"]

    train_ds = build_maneuver_dataset(cfg, subject, train_maneuver, ds_root=ds_root)
    model = build_foundation_model(train_ds.shapes, cfg)
    state = torch.load(core_path, map_location="cpu")
    model.load_core_state_dict(state, freeze=True)
    model.add_readout(subject)
    model.set_active(subject)

    loaders = train_ds.get_dataloaders()
    train_model(model, loaders["train"], epochs=max_epochs, device=device)

    records = []
    for maneuver in eval_maneuvers:
        ood_ds = build_maneuver_dataset(cfg, subject, maneuver, ds_root=ds_root)
        ood_loaders = ood_ds.get_dataloaders()
        metrics = evaluate_ood(model, ood_loaders["test"], maneuver=maneuver, device=device)
        preds, targets = collect_predictions(model, ood_loaders["test"], device=device)
        metrics.update(decompose_predictions(preds, targets))
        metrics.update({
            "subject": subject,
            "train_maneuver": train_maneuver,
            "eval_maneuver": maneuver,
        })
        records.append(metrics)

    logdir = logdir or f"exp-c-{subject}"
    pm = ProjectPathManager(branch=cfg.branch, target=logdir)
    out = pm.logdir / "ood_records.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[ood_exp] wrote {out} ({len(records)} records)", flush=True)
    return records


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--core", required=True, dest="core_path")
    p.add_argument("--branch", default="main")
    p.add_argument("--train-maneuver", default="baseline")
    p.add_argument("--eval-maneuvers", default="valsalva,pressor")
    p.add_argument("--input-mode", default="impedance")
    p.add_argument("--output-mode", default="waveform")
    p.add_argument("--arch", default="crt")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--logdir", default=None)
    p.add_argument("--ds-root", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = FoundationConfig(
        input_mode=args.input_mode,
        output_mode=args.output_mode,
        branch=args.branch,
        arch=args.arch,
        batch_size=args.batch_size,
    )
    try:
        run_ood_exp(
            subject=args.subject,
            core_path=args.core_path,
            cfg=cfg,
            train_maneuver=args.train_maneuver,
            eval_maneuvers=[m.strip() for m in args.eval_maneuvers.split(",") if m.strip()],
            max_epochs=args.max_epochs,
            logdir=args.logdir,
            ds_root=args.ds_root,
        )
    except FileNotFoundError as e:
        print(f"[ood_exp] {e}")
