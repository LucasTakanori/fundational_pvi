"""Return 0 if Parquet cache at --cache-root matches training config."""

from __future__ import annotations

import argparse
import sys

from src.foundation.config import FoundationConfig
from src.pipeline.pvi_cache import cache_is_valid, load_manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--input-mode", default="impedance")
    p.add_argument("--output-mode", default="waveform")
    p.add_argument("--mask-key", default="mask05")
    args = p.parse_args()

    cfg = FoundationConfig(
        input_mode=args.input_mode,
        output_mode=args.output_mode,
        mask_key=args.mask_key,
    )
    if cache_is_valid(args.cache_root, cfg):
        m = load_manifest(args.cache_root)
        print(
            f"[check_cache] OK  {args.cache_root}  "
            f"mode={m.get('input_mode')}  train={m.get('num_train'):,}  test={m.get('num_test'):,}",
            flush=True,
        )
        return 0

    print(f"[check_cache] MISSING or incompatible cache at {args.cache_root}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
