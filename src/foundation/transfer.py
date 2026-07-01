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
from src.foundation.model_factory import build_foundation_model
from src.foundation.foundation_model import PviFoundationModel


def infer_core_tag(core_path: str | Path, core_tag: str | None = None) -> str:
    """Short label for artifact dirs (e.g. coreS, coreU)."""
    if core_tag:
        return core_tag
    stem = Path(core_path).stem.lower()
    if stem.endswith("_u") or "core_u" in stem or "ssl" in stem:
        return "coreU"
    if "core" in stem:
        return "coreS"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe[:32] or "core"


def resolve_transfer_logdir(subject: str,
                            core_path: str | Path,
                            core_tag: str | None = None,
                            logdir: str | None = None) -> str:
    if logdir:
        return logdir
    tag = infer_core_tag(core_path, core_tag)
    return f"foundation-transfer-{subject}-{tag}"


def build_subject_dataset(cfg: FoundationConfig, subject: str, ds_root=None) -> PviLazyDataset:
    """Subject-scoped dataset for readout-only transfer.

    Always uses per-subject HDF5 (not the population parquet cache): transfer
    needs a within-subject train/test split, and the cache stores population
    disjoint splits where each subject lives entirely in train *or* test.
    """
    branch = getattr(cfg, "branch", "main")
    inventory = PviDatasetInventory(branch=branch, ds_root=ds_root)
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

    # Single-subject transfer: split sequences within subject (not disjoint).
    ds.set_partition(
        test_size=cfg.test_size,
        shuffle=True,
        split_mode="within",
        random_state=cfg.split_seed,
    )
    ds.get_partition()
    persistent = cfg.persistent_h5 and cfg.num_workers == 0
    ds.persistent_handle = persistent
    ds.set_dataloaders(
        batch_size=cfg.batch_size,
        shuffle=True,
        stratified=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return ds


def load_core_meta(core_path: str | Path) -> dict | None:
    meta_path = Path(core_path).with_name(
        Path(core_path).stem + "_meta.json"
    )
    if not meta_path.is_file():
        return None
    import json
    with open(meta_path) as f:
        return json.load(f)


def transfer_checkpoints_dir(logdir: str,
                             branch: str = "main",
                             artifacts_root: str | Path | None = None) -> Path:
    """``artifacts/{logdir}/{branch}/checkpoints`` from a transfer ``--logdir``."""
    root = Path(artifacts_root or os.environ.get("PVIPROJECT_ROOT", Path.cwd()))
    return root / "artifacts" / logdir / branch / "checkpoints"


def resolve_transfer_checkpoint(transfer_logdir: str | None = None,
                                checkpoint: str | Path | None = None,
                                branch: str = "main",
                                use_best: bool = True,
                                artifacts_root: str | Path | None = None) -> Path:
    """Resolve a transfer workflow checkpoint (.pth) from logdir or explicit path."""
    from src.foundation.export_core import resolve_checkpoint

    def _resolve_dir(ckpt_dir: Path) -> Path:
        if use_best:
            try:
                return resolve_checkpoint(ckpt_dir, use_best=True)
            except FileNotFoundError:
                pass
        return resolve_checkpoint(ckpt_dir, use_best=False)

    if checkpoint is not None:
        path = Path(checkpoint)
        if path.is_dir():
            return _resolve_dir(path)
        if not path.is_file():
            raise FileNotFoundError(f"Transfer checkpoint not found: {path}")
        return path
    if not transfer_logdir:
        raise ValueError("Provide transfer_logdir or checkpoint.")
    ckpt_dir = transfer_checkpoints_dir(
        transfer_logdir, branch=branch, artifacts_root=artifacts_root,
    )
    return _resolve_dir(ckpt_dir)


def load_transferred_model(subject: str,
                           core_path: str | Path,
                           cfg: FoundationConfig,
                           ds_root=None,
                           checkpoint: str | Path | None = None,
                           transfer_logdir: str | None = None,
                           branch: str | None = None,
                           use_best_checkpoint: bool = True,
                           artifacts_root: str | Path | None = None):
    """Build subject model; optionally restore a trained readout from transfer ckpt."""
    branch = branch or getattr(cfg, "branch", "main")
    meta = load_core_meta(core_path)
    if meta and meta.get("arch"):
        cfg.arch = meta["arch"]
        if meta.get("input_mode"):
            cfg.input_mode = meta["input_mode"]
        if meta.get("output_mode"):
            cfg.output_mode = meta["output_mode"]

    ds = build_subject_dataset(cfg, subject, ds_root=ds_root)
    model = build_foundation_model(ds.shapes, cfg)
    model.add_readout(subject)
    model.set_active(subject)

    if checkpoint or transfer_logdir:
        ckpt_path = resolve_transfer_checkpoint(
            transfer_logdir=transfer_logdir,
            checkpoint=checkpoint,
            branch=branch,
            use_best=use_best_checkpoint,
            artifacts_root=artifacts_root,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model" not in ckpt:
            raise KeyError(f"Checkpoint {ckpt_path} has no 'model' key.")
        readout_key = f"readouts.{subject}"
        if not any(k.startswith(readout_key) for k in ckpt["model"]):
            raise KeyError(
                f"No trained readout '{subject}' in {ckpt_path.name}. "
                f"Available readouts: "
                f"{sorted({k.split('.')[1] for k in ckpt['model'] if k.startswith('readouts.')})}"
            )
        model.load_state_dict(ckpt["model"], strict=False)
        model.set_active(subject)
        print(
            f"[transfer] loaded trained readout from {ckpt_path} "
            f"(epoch={ckpt.get('epoch', '?')})",
            flush=True,
        )
    else:
        state = torch.load(core_path, map_location="cpu")
        model.load_core_state_dict(state, freeze=True)

    return model, ds


def main(subject: str,
         core_path,
         cfg: FoundationConfig = None,
         logdir: str = None,
         core_tag: str = None,
         arch: str = None,
         ds_root=None) -> TrainingWorkflow:
    cfg = cfg or FoundationConfig()
    if arch:
        cfg.arch = arch
    else:
        meta = load_core_meta(core_path)
        if meta and meta.get("arch"):
            cfg.arch = meta["arch"]
            if meta.get("input_mode"):
                cfg.input_mode = meta["input_mode"]
            if meta.get("output_mode"):
                cfg.output_mode = meta["output_mode"]
    logdir = resolve_transfer_logdir(subject, core_path, core_tag=core_tag, logdir=logdir)

    pm = ProjectPathManager(branch=cfg.branch, target=logdir)
    ds = build_subject_dataset(cfg, subject, ds_root=ds_root)

    model = build_foundation_model(ds.shapes, cfg)

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
    wf.set_eval_interval(epochs=cfg.eval_every)
    wf.initiate_training(
        use_checkpoint=False,
        device=DEFAULT_TRAIN_DEVICE,
        dtype=DEFAULT_TRAIN_DTYPE,
        use_amp=cfg.use_amp,
    )
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
    p.add_argument("--branch", default="main",
                   help="TrainingBranch for subject lookup: main|holdout (PLAN.md §7.1).")
    p.add_argument("--readout-hidden", type=int, default=0,
                   help="MLP readout hidden size (0=linear).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--min-epochs", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--logdir", default=None,
                   help="Artifact logdir under artifacts/. Default: foundation-transfer-{subject}-{core-tag}.")
    p.add_argument("--core-tag", default=None,
                   help="Suffix for default logdir (e.g. coreS, coreU). Inferred from --core filename if omitted.")
    p.add_argument("--arch", default=None,
                   help="Core architecture (mlp|crt|mae|cnn). Inferred from *_meta.json next to --core if omitted.")
    p.add_argument("--ds-root", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = FoundationConfig(input_mode=args.input_mode,
                           output_mode=args.output_mode,
                           mask_key=args.mask_key,
                           branch=args.branch,
                           readout_hidden=args.readout_hidden,
                           batch_size=args.batch_size,
                           min_epochs=args.min_epochs,
                           max_epochs=args.max_epochs)
    try:
        main(args.subject, args.core_path, cfg=cfg, logdir=args.logdir,
             core_tag=args.core_tag, arch=args.arch, ds_root=args.ds_root)
    except FileNotFoundError as e:
        print(f"[transfer] {e}")
