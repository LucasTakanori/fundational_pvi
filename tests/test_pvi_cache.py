"""Unit tests for Parquet cache helpers (no full HDF5 cohort required)."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.foundation.config import FoundationConfig
from src.pipeline.pvi_cache import (
    META_COLUMNS,
    TENSOR_COLUMNS,
    _flush_shard,
    _sample_to_row,
    cache_is_valid,
    cache_key,
    manifest_matches,
)
from src.pipeline.pvi_parquet_dataset import PviParquetSplit, PviParquetCohort


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


def test_cache_key_includes_branch_and_split_mode():
    """A cache is only valid for the exact (branch, split_mode) it was built for --
    PLAN.md §3.8/§9: PW ('within'), PD ('disjoint'), and the true holdout branch must
    never be silently interchanged."""
    key_main_disjoint = cache_key(FoundationConfig(branch="main", split_mode="disjoint"))
    key_main_within = cache_key(FoundationConfig(branch="main", split_mode="within"))
    key_holdout_disjoint = cache_key(FoundationConfig(branch="holdout", split_mode="disjoint"))

    assert key_main_disjoint["branch"] == "main"
    assert key_main_disjoint["split_mode"] == "disjoint"
    # Different split_mode or branch must produce a different (non-matching) key.
    assert key_main_disjoint != key_main_within
    assert key_main_disjoint != key_holdout_disjoint
    assert key_main_within != key_holdout_disjoint


def test_cache_built_for_one_split_mode_rejected_by_a_different_config(tmp_path):
    """A PW-built cache must not silently satisfy a PD (disjoint) config, and vice versa --
    this is exactly the gap that made 'run PW/PD/holdout via the fast path' unsupported."""
    root = _write_mini_cache(tmp_path)  # built with default cfg: branch=main, split_mode=disjoint
    assert cache_is_valid(root, FoundationConfig(branch="main", split_mode="disjoint"))
    assert not cache_is_valid(root, FoundationConfig(branch="main", split_mode="within"))
    assert not cache_is_valid(root, FoundationConfig(branch="holdout", split_mode="disjoint"))


def test_parquet_cohort_reports_true_mask_bucket(tmp_path):
    """get_raw_statistics() must report the sequence count under the mask_key actually
    used to build the cache, not a hardcoded 'mask05' bucket regardless of config."""
    root = _write_mini_cache(tmp_path)
    ds = PviParquetCohort(cache_root=root, input_mode="signal",
                          output_mode="waveform", mask_key="mask10").build()
    stats = ds.get_raw_statistics()
    assert stats["num_seq10"] == ds.manifest["num_train"] + ds.manifest["num_test"]
    assert stats["num_seq05"] == 0
    assert stats["sqi"] is None  # not reconstructable from Parquet alone; must not fake 0.0


def test_shard_stores_float32_not_float64(tmp_path):
    """pa.Table.from_pylist() infers `double` from numpy .tolist() output (always
    64-bit Python floats) even when the source array was cast to float32 -- silently
    doubling storage/IO. _flush_shard must force a float32 schema so the cache is
    faithful to (and no less efficient than) the intended dtype."""
    sample = {
        "bp": np.random.randn(50).astype(np.float64),  # source may arrive as float64
        "stats": np.random.randn(2, 5).astype(np.float64),
        "pviHP": np.random.randn(1, 250).astype(np.float64),
        "pviLP": np.random.randn(1, 250).astype(np.float64),
    }
    meta = {"subject": "subj0", "session": "s1", "source_name": "file0"}
    row = _sample_to_row(sample, meta)

    out = tmp_path / "shard-00000.parquet"
    _flush_shard([row], out)

    table = pq.read_table(out)
    for col in TENSOR_COLUMNS:
        field_type = table.schema.field(col).type
        assert field_type == pa.list_(pa.float32()), (
            f"{col} stored as {field_type}, expected list<float32> (storage regression)"
        )
    for col in META_COLUMNS:
        assert table.schema.field(col).type == pa.string()

    # values still round-trip correctly (float32-precision-equivalent to the source)
    stored_bp = np.array(table.column("bp")[0].as_py(), dtype=np.float32)
    assert np.allclose(stored_bp, sample["bp"].astype(np.float32))


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
