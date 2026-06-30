"""Pretrain the shared PVI foundation core on the pooled population.

Builds a multi-subject lazy dataset, trains a `PviFoundationModel` through its
SHARED_READOUT (so every subject's data updates the common core), and saves the
trained core for transfer (see `src/foundation/transfer.py`).

Run from the repo root:

    python -m src.foundation.pretrain --input-mode signal --output-mode waveform --max-epochs 50

Datasets are discovered under `$PVIPROJECT_ROOT/datasets/main` (or `./data/datasets/main`
by default); point `PVIPROJECT_ROOT` (or `--ds-root`) at your `*_masked.h5` files.
"""

import argparse

from src.packages import *  # noqa: F401,F403  (torch, nn, optim, Path, ...)

from src.utils.primitives import (
    InputMode, OutputMode, SequenceMask,
    DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE,
)
from src.pipeline.data_discovery import ProjectPathManager, PviDatasetInventory
from src.pipeline.data_preparation_lazy import PviLazyDataset
from src.models.loss_functions import MorphologyLoss
from src.models.early_stopper import EarlyStopper
from src.models.workflow_v3 import TrainingWorkflow

from src.foundation.config import FoundationConfig
from src.foundation.foundation_model import PviFoundationModel, SHARED_READOUT


def build_population_dataset(cfg: FoundationConfig, ds_root=None) -> PviLazyDataset:
    inventory = PviDatasetInventory(branch="main", ds_root=ds_root)
    if len(inventory) == 0:
        raise FileNotFoundError(
            f"No '*_masked.h5' datasets found under '{inventory.target_dir}'. "
            f"Set PVIPROJECT_ROOT (or pass --ds-root) to a directory containing "
            f"a 'datasets/main/' folder of masked HDF5 files."
        )

    ds = PviLazyDataset(ds_files=inventory,
                        input_mode=InputMode(cfg.input_mode),
                        output_mode=OutputMode(cfg.output_mode),
                        mask_key=SequenceMask(cfg.mask_key),
                        max_cache=50,
                        persistent_handle=True).build()

    ds.set_partition(test_size=cfg.test_size, shuffle=True, split_mode="disjoint")
    ds.set_dataloaders(batch_size=cfg.batch_size, shuffle=False, stratified=True)
    return ds


def main(cfg: FoundationConfig = None,
         logdir: str = "foundation-pretrain",
         ds_root=None) -> TrainingWorkflow:
    cfg = cfg or FoundationConfig()

    pm = ProjectPathManager(branch="main", target=logdir)
    ds = build_population_dataset(cfg, ds_root=ds_root)

    # Pooled pretraining uses the single shared readout.
    model = PviFoundationModel(ds.shapes,
                               num_features=cfg.num_features,
                               num_hidden_layers=cfg.num_hidden_layers,
                               readout_hidden=cfg.readout_hidden,
                               diff=cfg.diff,
                               use_stats=cfg.use_stats)
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
    wf.initiate_training(use_checkpoint=True, device=DEFAULT_TRAIN_DEVICE, dtype=DEFAULT_TRAIN_DTYPE)
    wf.run(min_epochs=cfg.min_epochs, max_epochs=cfg.max_epochs)
    wf.export_artifacts()

    core_path = pm.logdir / "foundation_core.pt"
    torch.save(model.core_state_dict(), core_path)
    print(f"[pretrain] Saved shared core -> {core_path}")
    return wf


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-mode", default="signal")
    p.add_argument("--output-mode", default="waveform")
    p.add_argument("--mask-key", default="mask05")
    p.add_argument("--num-features", type=int, default=200)
    p.add_argument("--num-hidden-layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-epochs", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--logdir", default="foundation-pretrain")
    p.add_argument("--ds-root", default=None, help="Override the dataset root directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = FoundationConfig(input_mode=args.input_mode,
                           output_mode=args.output_mode,
                           mask_key=args.mask_key,
                           num_features=args.num_features,
                           num_hidden_layers=args.num_hidden_layers,
                           batch_size=args.batch_size,
                           min_epochs=args.min_epochs,
                           max_epochs=args.max_epochs)
    try:
        main(cfg, logdir=args.logdir, ds_root=args.ds_root)
    except FileNotFoundError as e:
        print(f"[pretrain] {e}")
