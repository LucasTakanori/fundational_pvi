"""Unit tests for Parquet cache helpers (no full HDF5 cohort required)."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.foundation.config import FoundationConfig
from src.pipeline.pvi_cache import (
    TENSOR_COLUMNS,
    cache_is_valid,
    cache_key,
    manifest_matches,
)
from src.pipeline.pvi_parquet_dataset import PviParquetSplit


def _write_mini_cache(tmp_path: Path) -> Path:
    cfg = FoundationConfig()
    root = tmp_path / "mini_cache"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)

    tensor_shapes = {
        "bp": [50],
        "stats": [2, 5],
        "pviHP": [1, 250],
        "pviLP": [1, 250],
    }
    rows = []
    for i in range(4):
        sample = {
            "bp": np.random.randn(50).astype(np.float32),
            "stats": np.random.randn(2, 5).astype(np.float32),
            "pviHP": np.random.randn(1, 250).astype(np.float32),
            "pviLP": np.random.randn(1, 250).astype(np.float32),
        }
        row = {
            "subject": f"subj{i}",
            "session": "s1",
            "source_name": f"file{i}",
        }
        for k in TENSOR_COLUMNS:
            row[k] = sample[k].reshape(-1).tolist()
        rows.append(row)

    pq.write_table(pa.Table.from_pylist(rows[:3]), root / "train" / "shard-00000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[3:]), root / "test" / "shard-00000.parquet")

    manifest = {
        **cache_key(cfg),
        "shard_rows": 8192,
        "num_train": 3,
        "num_test": 1,
        "tensor_shapes": tensor_shapes,
        "columns": list(TENSOR_COLUMNS) + ["subject", "session", "source_name"],
        "source_data_root": "/tmp",
        "created_utc": "2026-01-01T00:00:00+00:00",
    }
    with open(root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return root


def test_manifest_matches_and_cache_valid(tmp_path):
    root = _write_mini_cache(tmp_path)
    cfg = FoundationConfig()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_matches(manifest, cfg)
    assert cache_is_valid(root, cfg)


def test_parquet_split_getitem(tmp_path):
    from datasets import Dataset

    tensor_shapes = {
        "bp": [50],
        "stats": [2, 5],
        "pviHP": [1, 250],
        "pviLP": [1, 250],
    }
    row = {
        "bp": np.arange(50, dtype=np.float32).tolist(),
        "stats": np.ones(10, dtype=np.float32).tolist(),
        "pviHP": np.zeros(250, dtype=np.float32).tolist(),
        "pviLP": np.ones(250, dtype=np.float32).tolist(),
    }
    hf = Dataset.from_list([row])
    split = PviParquetSplit(hf, tensor_shapes)
    sample = split[0]
    assert set(sample.keys()) == set(TENSOR_COLUMNS)
    assert sample["bp"].shape == torch.Size([50])
    assert sample["pviHP"].shape == torch.Size([1, 250])
