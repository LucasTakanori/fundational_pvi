"""Pretrain the shared PVI foundation core on the pooled population.

Builds a multi-subject lazy dataset, trains a `PviFoundationModel` through its
SHARED_READOUT (so every subject's data updates the common core), and saves the
trained core for transfer (see `src/foundation/transfer.py`).

Run from the repo root:

    python -m src.foundation.pretrain --input-mode signal --output-mode waveform --max-epochs 50

Datasets are discovered under `$PVI_DATA_ROOT/main` (cluster default from
`env/cluster.env`), `$PVIPROJECT_ROOT/datasets/main`, or `./data/datasets/main`;
override with `--ds-root` if needed.
"""

import argparse
import os

from src.packages import *  # noqa: F401,F403  (torch, nn, optim, Path, ...)

from src.utils.primitives import (
    InputMode, OutputMode, SequenceMask,
    DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE,
)
from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_preparation_lazy import PviLazyDataset
from src.pipeline.pvi_cache import cache_is_valid
from src.pipeline.pvi_parquet_dataset import PviParquetCohort
from src.models.loss_functions import MorphologyLoss
from src.models.early_stopper import EarlyStopper
from src.models.workflow_v3 import TrainingWorkflow

from src.foundation.config import FoundationConfig
from src.foundation.export_core import export_core_from_checkpoint
from src.foundation.model_factory import build_foundation_model
from src.foundation.foundation_model import SHARED_READOUT


def dataloader_kwargs(cfg: FoundationConfig) -> dict:
    """DataLoader options shared by supervised pretrain and SSL."""
    params = {
        "batch_size": cfg.batch_size,
        "shuffle": not cfg.stratified,
        "stratified": cfg.stratified,
        "pin_memory": cfg.pin_memory,
    }
    if cfg.stratified:
        params["cluster_size"] = cfg.cluster_size
    if cfg.num_workers > 0:
        params.update({
            "num_workers": cfg.num_workers,
            "prefetch_factor": cfg.prefetch_factor,
            "persistent_workers": cfg.persistent_workers,
        })
    return params


def resolve_cache_root(cfg: FoundationConfig) -> str | None:
    root = cfg.cache_root or os.environ.get("PVI_CACHE_ROOT")
    if root and cache_is_valid(root, cfg):
        return root
    return None


def build_population_dataset(cfg: FoundationConfig, ds_root=None):
    cache_root = resolve_cache_root(cfg)
    if cache_root:
        print(f"[pretrain] using Parquet cache at {cache_root}", flush=True)
        ds = PviParquetCohort.from_cache(cache_root, cfg).build()
        ds.set_partition(
            test_size=cfg.test_size,
            shuffle=True,
            split_mode="disjoint",
            random_state=cfg.split_seed,
        )
        ds.get_partition()
        loader_kw = dataloader_kwargs(cfg)
        loader_kw["stratified"] = False
        if loader_kw.get("num_workers", 0) == 0:
            loader_kw["num_workers"] = cfg.cache_num_workers
            loader_kw.setdefault("prefetch_factor", cfg.prefetch_factor)
            loader_kw.setdefault("persistent_workers", cfg.persistent_workers)
        ds.set_dataloaders(**loader_kw)
        return ds

    inventory = PviDatasetInventory(branch="main", ds_root=ds_root)
    if len(inventory) == 0:
        raise FileNotFoundError(
            f"No '*_masked.h5' datasets found under '{inventory.target_dir}'. "
            f"Set PVI_DATA_ROOT, PVIPROJECT_ROOT (or pass --ds-root) to a directory "
            f"whose '{inventory.branch.value}/' subfolder contains masked HDF5 files."
        )

    # Persistent HDF5 handles are not fork-safe; disable when using worker processes.
    persistent = cfg.persistent_h5 and cfg.num_workers == 0

    ds = PviLazyDataset(ds_files=inventory,
                        input_mode=InputMode(cfg.input_mode),
                        output_mode=OutputMode(cfg.output_mode),
                        mask_key=SequenceMask(cfg.mask_key),
                        max_cache=cfg.max_cache,
                        persistent_handle=persistent).build()

    ds.clear_cache_every_epoch = cfg.clear_cache_every_epoch
    ds.set_partition(
        test_size=cfg.test_size,
        shuffle=True,
        split_mode="disjoint",
        random_state=cfg.split_seed,
    )
    ds.get_partition()
    ds.set_dataloaders(**dataloader_kwargs(cfg))
    return ds


def main(cfg: FoundationConfig = None,
         logdir: str = "foundation-pretrain",
         ds_root=None) -> TrainingWorkflow:
    cfg = cfg or FoundationConfig()

    pm = ProjectPathManager(branch="main", target=logdir)
    ds = build_population_dataset(cfg, ds_root=ds_root)

    # Pooled pretraining uses the single shared readout.
    model = build_foundation_model(ds.shapes, cfg)
    model.set_active(SHARED_READOUT)

    loss_fn = MorphologyLoss(base_loss=nn.MSELoss(), base_weight=cfg.mse_weight)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, mode="min", factor=0.8)
    stopper = EarlyStopper(patience=50, delta=1e-4, mode="max", threshold=0.5, verbose=True)

    wf = TrainingWorkflow(path_manager=pm,
                          dataset=ds,
                          model=model,
                          loss_func=loss_fn,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          stopper=stopper)
    wf.set_checkpoint_interval(minutes=120, epochs=10)
    wf.set_eval_interval(epochs=cfg.eval_every)
    wf.initiate_training(use_checkpoint=True,
                         device=DEFAULT_TRAIN_DEVICE,
                         dtype=DEFAULT_TRAIN_DTYPE,
                         use_amp=cfg.use_amp)
    wf.run(min_epochs=cfg.min_epochs, max_epochs=cfg.max_epochs)
    wf.export_artifacts()

    core_path = pm.logdir / "foundation_core.pt"
    meta_path = pm.logdir / "foundation_core_meta.json"
    try:
        export_core_from_checkpoint(
            checkpoint=pm.logdir / "checkpoints",
            output=core_path,
            use_best=True,
        )
    except FileNotFoundError:
        export_core_from_checkpoint(
            checkpoint=pm.logdir / "checkpoints",
            output=core_path,
            use_best=False,
        )
    import json
    meta = {
        "arch": cfg.arch,
        "input_mode": cfg.input_mode,
        "output_mode": cfg.output_mode,
        "shapes": {k: list(v) for k, v in ds.shapes.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pretrain] Saved shared core -> {core_path}")
    print(f"[pretrain] Saved core metadata -> {meta_path}")
    return wf


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-mode", default="signal")
    p.add_argument("--output-mode", default="waveform")
    p.add_argument("--mask-key", default="mask05")
    p.add_argument("--arch", default="mlp", help="Core architecture: mlp|crt|mae|cnn (see MODELS.md).")
    p.add_argument("--num-features", type=int, default=512)
    p.add_argument("--num-hidden-layers", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-cache", type=int, default=150)
    p.add_argument("--cluster-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--clear-cache-each-epoch", action="store_true")
    p.add_argument("--min-epochs", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--logdir", default="foundation-pretrain")
    p.add_argument("--ds-root", default=None, help="Override the dataset root directory.")
    p.add_argument("--cache-root", default=None, help="Parquet cache directory (or set PVI_CACHE_ROOT).")
    p.add_argument("--cache-num-workers", type=int, default=None,
                   help="DataLoader workers when training from cache (default: config).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cache_num_workers = args.cache_num_workers if args.cache_num_workers is not None else FoundationConfig.cache_num_workers
    cfg = FoundationConfig(input_mode=args.input_mode,
                           output_mode=args.output_mode,
                           mask_key=args.mask_key,
                           arch=args.arch,
                           num_features=args.num_features,
                           num_hidden_layers=args.num_hidden_layers,
                           batch_size=args.batch_size,
                           max_cache=args.max_cache,
                           cluster_size=args.cluster_size,
                           num_workers=args.num_workers,
                           cache_root=args.cache_root,
                           cache_num_workers=cache_num_workers,
                           eval_every=args.eval_every,
                           use_amp=not args.no_amp,
                           clear_cache_every_epoch=args.clear_cache_each_epoch,
                           min_epochs=args.min_epochs,
                           max_epochs=args.max_epochs)
    try:
        main(cfg, logdir=args.logdir, ds_root=args.ds_root)
    except FileNotFoundError as e:
        print(f"[pretrain] {e}")
