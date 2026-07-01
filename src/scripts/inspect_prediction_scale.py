"""Inspect raw prediction scale vs targets (PLAN.md Phase 2 diagnostic)."""

from __future__ import annotations

import argparse

from src.packages import *  # noqa: F401,F403

from src.utils.primitives import DEFAULT_TRAIN_DEVICE
from src.foundation.config import FoundationConfig
from src.foundation.evaluation import collect_predictions
from src.foundation.transfer import load_transferred_model


def _stats_tensor(t: torch.Tensor) -> dict[str, float]:
    t = t.detach().float().cpu()
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
    }


def inspect_scale(model, loader, device=None) -> dict:
    device = device or DEFAULT_TRAIN_DEVICE
    preds, targets = collect_predictions(model, loader, device=device)
    return {
        "predictions": _stats_tensor(preds),
        "targets": _stats_tensor(targets),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True)
    p.add_argument("--core", required=True)
    p.add_argument("--branch", default="main")
    p.add_argument("--input-mode", default="impedance")
    p.add_argument("--arch", default=None)
    p.add_argument("--ds-root", default=None)
    p.add_argument(
        "--transfer-logdir",
        default=None,
        help="Transfer artifact logdir (e.g. debug-ssl-scale-check-transfer). "
             "Loads the trained readout from checkpoints/ under this dir.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Transfer checkpoint .pth file or checkpoints/ directory.",
    )
    p.add_argument(
        "--no-best-checkpoint",
        action="store_true",
        help="Use latest checkpoint instead of *_best.pth when resolving logdir.",
    )
    args = p.parse_args()

    cfg = FoundationConfig(input_mode=args.input_mode, branch=args.branch)
    if args.arch:
        cfg.arch = args.arch

    model, ds = load_transferred_model(
        args.subject,
        args.core,
        cfg,
        ds_root=args.ds_root,
        checkpoint=args.checkpoint,
        transfer_logdir=args.transfer_logdir,
        branch=args.branch,
        use_best_checkpoint=not args.no_best_checkpoint,
    )
    device = DEFAULT_TRAIN_DEVICE
    model = model.to(device)
    loaders = ds.get_dataloaders()
    stats = inspect_scale(model, loaders["test"], device=device)

    src = args.checkpoint or args.transfer_logdir or args.core
    print(f"[scale_check] subject={args.subject}  source={src}")
    for side in ("predictions", "targets"):
        s = stats[side]
        print(f"  {side:12s}  min={s['min']:.2f}  max={s['max']:.2f}  "
              f"mean={s['mean']:.2f}  std={s['std']:.2f}")
    print("  (True BP is typically ~40–200 mmHg.)")


if __name__ == "__main__":
    main()
