"""Build the Parquet + HuggingFace datasets cache for foundation training.

Run once (or when data / split config changes):

    python -m src.scripts.build_pvi_cache --cache-root /mmfs1/scratch/$USER/pvi_cache/v1

Then set ``PVI_CACHE_ROOT`` (see ``env/cluster.env``) before ``pretrain`` / ``ssl_pretrain``.
"""

from __future__ import annotations

import argparse
import os

from src.foundation.config import FoundationConfig
from src.pipeline.pvi_cache import build_pvi_cache, cache_is_valid


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-mode", default="signal")
    p.add_argument("--output-mode", default="waveform")
    p.add_argument("--mask-key", default="mask05")
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--branch", default="main",
                   help="TrainingBranch: main|holdout|longitudinal (PLAN.md §7.1).")
    p.add_argument("--split-mode", default="disjoint",
                   help="SplitMode: disjoint (PD, subject-holdout) | within (PW, "
                        "same-subjects-in-train-and-test) | global.")
    p.add_argument("--shard-rows", type=int, default=8192)
    p.add_argument("--ds-root", default=None)
    p.add_argument(
        "--cache-root",
        default=os.environ.get("PVI_CACHE_ROOT"),
        help="Output directory (default: $PVI_CACHE_ROOT).",
    )
    p.add_argument("--force", action="store_true", help="Rebuild even if cache looks valid.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.cache_root:
        raise SystemExit(
            "Provide --cache-root or set PVI_CACHE_ROOT (e.g. /mmfs1/scratch/$USER/pvi_cache/v1)."
        )

    cfg = FoundationConfig(
        input_mode=args.input_mode,
        output_mode=args.output_mode,
        mask_key=args.mask_key,
        test_size=args.test_size,
        split_seed=args.split_seed,
        branch=args.branch,
        split_mode=args.split_mode,
    )

    if not args.force and cache_is_valid(args.cache_root, cfg):
        print(f"[build_pvi_cache] cache already valid at {args.cache_root}", flush=True)
        return

    build_pvi_cache(
        cfg,
        cache_root=args.cache_root,
        ds_root=args.ds_root,
        shard_rows=args.shard_rows,
    )


if __name__ == "__main__":
    main()
