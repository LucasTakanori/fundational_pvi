"""Fast training cohort backed by Parquet shards + HuggingFace ``datasets``."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.pipeline._data_preparation import PviConfiguredDataset
from src.pipeline.pvi_cache import (
    TENSOR_COLUMNS,
    cache_is_valid,
    load_hf_dataset,
    load_manifest,
)
from src.utils.primitives import InputMode, OutputMode, SequenceMask, subject_name_to_idx


class PviParquetSplit(Dataset):
    """One train or test split materialized from Parquet."""

    def __init__(self, hf_split, tensor_shapes: dict[str, list[int]]) -> None:
        self.hf = hf_split
        self.tensor_shapes = tensor_shapes
        self.hf.set_format(type="torch", columns=list(TENSOR_COLUMNS))
        # Precompute once — per-row with_format(None) in __getitem__ was ~10x slower.
        raw_subjects = hf_split.with_format(None)["subject"]
        self._subject_ids = [
            subject_name_to_idx(str(s.decode() if isinstance(s, bytes) else s))
            for s in raw_subjects
        ]

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.hf[idx]
        out: dict[str, torch.Tensor] = {}
        for key, shape in self.tensor_shapes.items():
            tensor = row[key]
            if not torch.is_tensor(tensor):
                tensor = torch.as_tensor(tensor, dtype=torch.float32)
            else:
                tensor = tensor.to(dtype=torch.float32)
            out[key] = tensor.view(*shape)
        out["subject_idx"] = torch.tensor(self._subject_ids[idx], dtype=torch.long)
        return out


class PviParquetCohort(PviConfiguredDataset):
    """Drop-in replacement for ``PviLazyDataset`` when ``PVI_CACHE_ROOT`` is set."""

    def __init__(self,
                 cache_root: str | Path,
                 input_mode: str | InputMode,
                 output_mode: str | OutputMode,
                 mask_key: str | SequenceMask,
                 name: str = "dataset_parquet") -> None:
        super().__init__(input_mode=input_mode, output_mode=output_mode, mask_key=mask_key)
        self.cache_root = Path(cache_root)
        self.name = name
        self._alias = f"{self.name} (Parquet)"
        self.path = str(self.cache_root)
        self.manifest: dict = {}
        self.tensor_shapes: dict[str, list[int]] = {}
        self._hf = None
        self.train_ds: PviParquetSplit | None = None
        self.test_ds: PviParquetSplit | None = None
        self.raws = []
        self.period_length = 0
        self.num_periods = 0
        self.clear_cache_every_epoch = False

    @classmethod
    def from_cache(cls, cache_root: str | Path, cfg) -> "PviParquetCohort":
        root = Path(cache_root)
        if not cache_is_valid(root, cfg):
            raise FileNotFoundError(
                f"Parquet cache at '{root}' is missing or incompatible with the current "
                f"FoundationConfig. Run `python -m src.scripts.build_pvi_cache` first."
            )
        return cls(
            cache_root=root,
            input_mode=cfg.input_mode,
            output_mode=cfg.output_mode,
            mask_key=cfg.mask_key,
        )

    def build(self) -> "PviParquetCohort":
        self.manifest = load_manifest(self.cache_root)
        self.tensor_shapes = self.manifest["tensor_shapes"]
        self.period_length = int(self.tensor_shapes["pviHP"][-1])
        self.num_periods = int(self.manifest["num_train"] + self.manifest["num_test"])
        return self

    def set_partition(self,
                      test_size: float = 0.1,
                      shuffle: bool = True,
                      **kwargs) -> None:
        params = {"test_size": test_size, "shuffle": shuffle, **kwargs}
        self._split_params = params

    def get_partition(self) -> dict[str, Dataset]:
        if self._hf is None:
            self._hf = load_hf_dataset(self.cache_root)

        self.train_ds = PviParquetSplit(self._hf["train"], self.tensor_shapes)
        self.test_ds = PviParquetSplit(self._hf["test"], self.tensor_shapes)
        for split_ds in (self.train_ds, self.test_ds):
            for kw in self._VALID_CONFIGS_KEYS:
                setattr(split_ds, kw, getattr(self, kw))

        n_train = len(self.train_ds)
        n_test = len(self.test_ds)
        self.active_mask = [(i, i + 1) for i in range(n_train + n_test)]
        self.train_mask = self.active_mask[:n_train]
        self.test_mask = self.active_mask[n_train:]

        self.subsets = {"train": self.train_ds, "test": self.test_ds}
        return self.subsets

    def set_dataloaders(self,
                        batch_size: int = 32,
                        shuffle: bool = True,
                        stratified: bool = False,
                        **kwargs) -> None:
        params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "stratified": False,
            **{k: v for k, v in kwargs.items()
               if k not in ("stratified", "cluster_size", "persistent_handle")},
        }
        self._loader_params = params

    def get_dataloaders(self) -> dict[str, DataLoader]:
        if not self.subsets:
            raise AttributeError("Call get_partition() before get_dataloaders().")
        if not self._loader_params:
            raise AttributeError("Call set_dataloaders() before get_dataloaders().")

        loader_params = {
            k: v for k, v in self._loader_params.items()
            if k not in ("stratified", "cluster_size", "persistent_handle")
        }
        worker_keys = ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")
        worker_params = {k: loader_params[k] for k in worker_keys if k in loader_params}
        batch_size = loader_params["batch_size"]
        shuffle = loader_params.get("shuffle", True)

        self.loaders = {
            "train": DataLoader(
                self.subsets["train"],
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
                **worker_params,
            ),
            "test": DataLoader(
                self.subsets["test"],
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                **worker_params,
            ),
        }
        return self.loaders

    def __len__(self) -> int:
        return len(self.active_mask)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Use train/test subsets directly with the Parquet cache.")

    @property
    def shapes(self) -> dict:
        ts = self.tensor_shapes
        return {
            "input": tuple(ts["pviHP"]),
            "output": tuple(ts["bp"]),
            "stats": tuple(ts["stats"]),
        }

    def to(self, device=None, dtype=None, **kwargs) -> "PviParquetCohort":
        return self

    def load_state_dict(self, state_dict: dict) -> None:
        for kw in self._VALID_MASK_ATTRIBUTES:
            if kw in state_dict:
                setattr(self, kw, state_dict[kw])

    def get_raw_statistics(self) -> dict[str, float | int]:
        # SQI (signal quality index) is computed from raw HDF5 masks, which the
        # cache-materialized rows don't carry; not reconstructable from Parquet alone.
        # Report the true sequence count against the *actual* mask_key used to build the
        # cache, not a hardcoded 'mask05' bucket regardless of config (PLAN.md §7 note).
        total = int(self.manifest.get("num_train", 0) + self.manifest.get("num_test", 0))
        counts = {f"num_seq{k}": 0 for k in ("01", "05", "10", "15")}
        mask_suffix = str(self.mask_key.value if hasattr(self.mask_key, "value") else self.mask_key)
        mask_suffix = mask_suffix.replace("mask", "") or "05"
        counts[f"num_seq{mask_suffix}"] = total
        return {"sqi": None, "num_periods": self.num_periods, **counts}

    def get_params_shallow(self) -> dict:
        counts = {
            "num_periods": self.num_periods,
            "num_sequences": len(self.active_mask),
            "num_train": len(self.train_mask),
            "num_test": len(self.test_mask),
        }
        return {
            "class": type(self).__name__,
            "name": self.name,
            "constituents": [self.cache_root.name],
            "configs": self.configs,
            "counts": counts,
            "cache_root": str(self.cache_root),
            "split_params": self._split_params,
            "batch_params": self._loader_params,
            "shapes": self.shapes,
            "raw_stats": self.get_raw_statistics(),
        }

    def _validate_components(self, components) -> None:
        pass
