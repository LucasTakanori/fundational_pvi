"""Transfer a pretrained foundation core to a new (held-out) subject.

This is the core+readout transfer protocol: load the shared core produced by
`src/foundation/pretrain.py`, freeze it, attach a fresh `SubjectReadout` for the
target subject, and train *only* that readout on the subject's data.

Run from the repo root:

    python -m src.foundation.transfer --subject subject013 \
        --core artifacts/foundation-pretrain/main/foundation_core.pt
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
from src.foundation.foundation_model import PviFoundationModel


def build_subject_dataset(cfg: FoundationConfig, subject: str, ds_root=None) -> PviLazyDataset:
    inventory = PviDatasetInventory(branch="main", ds_root=ds_root)
    files = [f for f in inventory if f.subject == subject]
    if not files:
        available = sorted({f.subject for f in inventory})
        raise FileNotFoundError(
            f"No datasets for subject '{subject}' under '{inventory.target_dir}'. "
            f"Available subjects: {available or '(none found)'}"
        )

    ds = PviLazyDataset(ds_files=files,
                        input_mode=InputMode(cfg.input_mode),
                        output_mode=OutputMode(cfg.output_mode),
                        mask_key=SequenceMask(cfg.mask_key),
                        max_cache=10,
                        persistent_handle=True).build()

    ds.set_partition(test_size=cfg.test_size, shuffle=True, split_mode="disjoint")
    ds.set_dataloaders(batch_size=cfg.batch_size, shuffle=False, stratified=False)
    return ds


def main(subject: str,
         core_path,
         cfg: FoundationConfig = None,
         logdir: str = None,
         ds_root=None) -> TrainingWorkflow:
    cfg = cfg or FoundationConfig()
    logdir = logdir or f"foundation-transfer-{subject}"

    pm = ProjectPathManager(branch="main", target=logdir)
    ds = build_subject_dataset(cfg, subject, ds_root=ds_root)

    model = PviFoundationModel(ds.shapes,
                               num_features=cfg.num_features,
                               num_hidden_layers=cfg.num_hidden_layers,
                               readout_hidden=cfg.readout_hidden,
                               diff=cfg.diff,
                               use_stats=cfg.use_stats)

    # Load + freeze the pretrained core; train only a fresh readout for `subject`.
    state = torch.load(core_path, map_location="cpu")
    model.load_core_state_dict(state, freeze=True)
    model.add_readout(subject)
    model.set_active(subject)

    trainable = [p for p in model.parameters() if p.requires_grad]
    loss_fn = MorphologyLoss(base_loss=nn.MSELoss(), base_weight=cfg.mse_weight)
    optimizer = optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
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
    return wf


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject", required=True, help="Target subject id, e.g. subject013.")
    p.add_argument("--core", required=True, dest="core_path",
                   help="Path to the pretrained foundation core (.pt).")
    p.add_argument("--input-mode", default="signal")
    p.add_argument("--output-mode", default="waveform")
    p.add_argument("--mask-key", default="mask05")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-epochs", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--logdir", default=None)
    p.add_argument("--ds-root", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = FoundationConfig(input_mode=args.input_mode,
                           output_mode=args.output_mode,
                           mask_key=args.mask_key,
                           batch_size=args.batch_size,
                           min_epochs=args.min_epochs,
                           max_epochs=args.max_epochs)
    try:
        main(args.subject, args.core_path, cfg=cfg, logdir=args.logdir, ds_root=args.ds_root)
    except FileNotFoundError as e:
        print(f"[transfer] {e}")
