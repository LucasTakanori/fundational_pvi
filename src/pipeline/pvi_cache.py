"""Build and load Parquet + HuggingFace ``datasets`` caches for PVI training.

Phase A (once): HDF5 on NFS → sharded Parquet under ``$PVI_CACHE_ROOT``.
Phase B (train): ``datasets.load_dataset('parquet', ...)`` + fast DataLoader.
"""

from __future__ import annotations

from collections import defaultdict
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.pipeline.data_discovery import PviDatasetInventory
from src.pipeline.data_preparation_lazy import PviLazyDataset
from src.utils import h5io
from src.utils.primitives import InputMode, OutputMode, SequenceMask, resolve_data_root


CACHE_VERSION = 1
DEFAULT_SHARD_ROWS = 8192

TENSOR_COLUMNS = ("bp", "stats", "pviHP", "pviLP")
META_COLUMNS = ("subject", "session", "source_name")


def cache_key(cfg) -> dict:
    """Manifest fields that must match for a cache to be considered compatible."""
    return {
        "version": CACHE_VERSION,
        "input_mode": str(cfg.input_mode),
        "output_mode": str(cfg.output_mode),
        "mask_key": str(cfg.mask_key),
        "test_size": float(cfg.test_size),
        "split_mode": "disjoint",
        "seed": int(getattr(cfg, "split_seed", 42)),
    }


def manifest_path(cache_root: Path) -> Path:
    return Path(cache_root) / "manifest.json"


def load_manifest(cache_root: Path) -> dict:
    with open(manifest_path(cache_root), encoding="utf-8") as f:
        return json.load(f)


def manifest_matches(manifest: dict, cfg) -> bool:
    want = cache_key(cfg)
    for k, v in want.items():
        if manifest.get(k) != v:
            return False
    return True


def cache_is_valid(cache_root: str | Path, cfg) -> bool:
    root = Path(cache_root)
    if not manifest_path(root).is_file():
        return False
    manifest = load_manifest(root)
    if not manifest_matches(manifest, cfg):
        return False
    for split in ("train", "test"):
        if not any((root / split).glob("*.parquet")):
            return False
    return True


def _tensor_shapes(sample: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {k: list(v.shape) for k, v in sample.items() if k in TENSOR_COLUMNS}


def _sample_to_row(sample: dict, meta: dict) -> dict:
    row = dict(meta)
    for k in TENSOR_COLUMNS:
        row[k] = sample[k].astype(np.float32).reshape(-1).tolist()
    return row


def _flush_shard(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path, compression="zstd")


def _global_mask_set(masks: list[tuple[int, ...]]) -> set[tuple[int, ...]]:
    return {tuple(m) for m in masks}


def build_pvi_cache(cfg,
                    cache_root: str | Path,
                    ds_root=None,
                    shard_rows: int = DEFAULT_SHARD_ROWS,
                    verbose: bool = True) -> Path:
    """Export train/test Parquet shards + ``manifest.json``."""
    cache_root = Path(cache_root)
    ds_root = resolve_data_root(ds_root)
    split_seed = int(getattr(cfg, "split_seed", 42))

    inventory = PviDatasetInventory(branch="main", ds_root=ds_root)
    if len(inventory) == 0:
        raise FileNotFoundError(f"No HDF5 files under {inventory.target_dir}")

    lazy = PviLazyDataset(
        ds_files=inventory,
        input_mode=InputMode(cfg.input_mode),
        output_mode=OutputMode(cfg.output_mode),
        mask_key=SequenceMask(cfg.mask_key),
        max_cache=8,
        persistent_handle=False,
    ).build()
    lazy.set_partition(
        test_size=cfg.test_size,
        shuffle=True,
        split_mode="disjoint",
        random_state=split_seed,
    )
    lazy.get_partition()

    train_globals = _global_mask_set(lazy.train_mask)
    test_globals = _global_mask_set(lazy.test_mask)

    global_to_split: dict[tuple[int, ...], str] = {}
    for gmask in lazy.mappings.masks_global:
        if gmask in train_globals:
            global_to_split[gmask] = "train"
        elif gmask in test_globals:
            global_to_split[gmask] = "test"

    input_mode = InputMode(cfg.input_mode)
    output_mode = OutputMode(cfg.output_mode)

    # Sample-level mappings stay aligned after disjoint partition; groupings.masks_global
    # must not be zipped per-file post-partition (subject merge can alias those lists).
    by_file: dict = defaultdict(list)
    for ds_raw, lmask, gmask in zip(
        lazy.mappings.files,
        lazy.mappings.masks_local,
        lazy.mappings.masks_global,
        strict=True,
    ):
        by_file[ds_raw].append((lmask, gmask))

    buffers: dict[str, list[dict]] = {"train": [], "test": []}
    shard_ids: dict[str, int] = {"train": 0, "test": 0}
    counts: dict[str, int] = {"train": 0, "test": 0}
    tensor_shapes: dict[str, list[int]] | None = None

    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    file_iter = tqdm(lazy.groupings.files, desc="build_pvi_cache", disable=not verbose)
    for ds_raw in file_iter:
        raw = ds_raw.extract_tensors(input_mode=input_mode, idx=None)
        formatted = h5io.format_raw_tensors(
            raw_data=raw,
            input_mode=input_mode,
            output_mode=output_mode,
            period_length=ds_raw.period_length,
        )
        meta_base = {
            "subject": ds_raw.subject,
            "session": ds_raw.session,
            "source_name": ds_raw.name,
        }

        for bounds, gmask in by_file[ds_raw]:
            split = global_to_split.get(gmask)
            if split is None:
                continue
            sample = h5io.slice_sequences(
                data=formatted,
                bounds=bounds,
                period_length=ds_raw.period_length,
            )
            sample_np = {k: v.detach().cpu().numpy() for k, v in sample.items()}
            if tensor_shapes is None:
                tensor_shapes = _tensor_shapes(sample_np)

            buffers[split].append(_sample_to_row(sample_np, meta_base))
            counts[split] += 1

            if len(buffers[split]) >= shard_rows:
                out = cache_root / split / f"shard-{shard_ids[split]:05d}.parquet"
                _flush_shard(buffers[split], out)
                buffers[split].clear()
                shard_ids[split] += 1

    for split in ("train", "test"):
        if buffers[split]:
            out = cache_root / split / f"shard-{shard_ids[split]:05d}.parquet"
            _flush_shard(buffers[split], out)

    manifest = {
        **cache_key(cfg),
        "shard_rows": int(shard_rows),
        "num_train": counts["train"],
        "num_test": counts["test"],
        "tensor_shapes": tensor_shapes,
        "columns": list(TENSOR_COLUMNS) + list(META_COLUMNS),
        "source_data_root": str(ds_root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path(cache_root), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print(
            f"[pvi_cache] wrote {counts['train']:,} train + {counts['test']:,} test "
            f"samples → {cache_root}",
            flush=True,
        )
    return cache_root


def load_hf_dataset(cache_root: str | Path):
    """Load cached Parquet splits with HuggingFace ``datasets``."""
    from datasets import load_dataset

    root = Path(cache_root)
    return load_dataset(
        "parquet",
        data_files={
            "train": str(root / "train" / "*.parquet"),
            "test": str(root / "test" / "*.parquet"),
        },
    )
