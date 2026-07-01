"""Cohort SSL pretraining -> the shared foundation core U.

Trains `PviSSLModel` (masked reconstruction + forecasting) on the pooled
population (input only, no BP labels), then saves the shared `PviCore` for
transfer into `PviFoundationModel`.

Run from the repo root:

    python -m src.foundation.ssl_pretrain --input-mode signal --max-epochs 50
"""

import argparse
import sys

from src.packages import *  # noqa: F401,F403  (torch, optim, Path, ...)

from tqdm import tqdm

from src.utils.primitives import DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE
from src.pipeline.data_discovery import ProjectPathManager

from src.foundation.config import FoundationConfig
from src.foundation.model_factory import build_ssl_model
from src.foundation.ssl import PviSSLModel
from src.foundation.pretrain import build_population_dataset
from src.foundation.adversarial import attach_subject_adversary, dann_lambda_schedule, run_subject_adversary_step
from src.foundation.ssl_probe import run_ssl_linear_probe


def _log(msg: str) -> None:
    print(msg, flush=True)


def _to_device(batch: dict, device, dtype, non_blocking: bool = False) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, dtype=dtype, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def main(cfg: FoundationConfig = None,
         logdir: str = "foundation-ssl-pretrain",
         ds_root=None,
         device=None) -> PviSSLModel:
    cfg = cfg or FoundationConfig()
    device = device or DEFAULT_TRAIN_DEVICE
    use_amp = cfg.use_amp and getattr(device, "type", str(device)) == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    _log(f"[ssl] device={device}  amp={use_amp}  batch_size={cfg.batch_size}  "
         f"max_cache={cfg.max_cache}  stratified={cfg.stratified}")

    pm = ProjectPathManager(branch=cfg.branch, target=logdir)
    _log("[ssl] building cohort dataset...")
    ds = build_population_dataset(cfg, ds_root=ds_root)
    _log(f"[ssl] train={len(ds.train_mask):,}  test={len(ds.test_mask):,}  "
         f"cluster_size={getattr(cfg, 'cluster_size', 'n/a')}")

    model = build_ssl_model(ds.shapes, cfg).to(device=device, dtype=DEFAULT_TRAIN_DTYPE)

    if cfg.subject_adversary:
        attach_subject_adversary(model)
        _log(f"[ssl] subject-adversary head attached (weight={cfg.subject_adversary_weight})")

    probe_subjects = [s.strip() for s in cfg.probe_subjects.split(",") if s.strip()]

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        loaders = ds.get_dataloaders()
        n_batches = len(loaders["train"])
        _log(f"[ssl] epoch {epoch}/{cfg.max_epochs}  batches={n_batches}")

        running = {"total": 0.0, "mask": 0.0, "forecast": 0.0}
        pbar = tqdm(loaders["train"], desc=f"ssl e{epoch}", file=sys.stdout, mininterval=2.0)
        for batch in pbar:
            batch = _to_device(batch, device, DEFAULT_TRAIN_DTYPE, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
                losses = model.pretext_loss(batch)
            if use_amp:
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                optimizer.step()

            if cfg.subject_adversary and "subject_idx" in batch:
                progress = epoch / max(cfg.max_epochs, 1)
                lam = dann_lambda_schedule(progress, gamma=cfg.dann_gamma)
                run_subject_adversary_step(
                    model, batch, optimizer, lambda_=lam,
                    weight=cfg.subject_adversary_weight,
                    scaler=scaler, use_amp=use_amp,
                )

            for k in running:
                running[k] += float(losses[k])
            pbar.set_postfix(total=f"{float(losses['total']):.4f}")

        if n_batches:
            msg = "  ".join(f"{k}={running[k] / n_batches:.4f}" for k in running)
            _log(f"[ssl] epoch {epoch:4d}/{cfg.max_epochs}  {msg}")

        if cfg.probe_every > 0 and probe_subjects and epoch % cfg.probe_every == 0:
            probe = run_ssl_linear_probe(
                model, cfg, probe_subjects, device=device, ds_root=ds_root,
            )
            _log(f"[ssl] probe  cc_abs={probe['probe_cc_abs']:.4f}  "
                 f"amae={probe['probe_amae']:.4f}  (n_subjects={probe.get('probe_subjects', len(probe_subjects))})")

        if getattr(ds, "clear_cache_every_epoch", False):
            ds.cleanup(attrs="cache", placeholder=None)
            gc.collect()

    core_path = pm.logdir / "foundation_core_U.pt"
    meta_path = pm.logdir / "foundation_core_U_meta.json"
    torch.save(model.core_state_dict(), core_path)
    import json
    meta = {
        "arch": cfg.ssl_arch or cfg.arch,
        "input_mode": cfg.input_mode,
        "output_mode": cfg.output_mode,
        "shapes": {k: list(v) for k, v in ds.shapes.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    _log(f"[ssl] Saved SSL-pretrained core (U) -> {core_path}")
    _log(f"[ssl] Saved core metadata -> {meta_path}")
    return model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-mode", default="signal")
    p.add_argument("--output-mode", default="waveform")  # unused by SSL; keeps dataset happy
    p.add_argument("--arch", default="mae", help="SSL encoder arch: mlp|mae.")
    p.add_argument("--ssl-arch", default=None, dest="ssl_arch",
                   help="Override SSL arch (defaults to --arch).")
    p.add_argument("--mask-key", default="mask05")
    p.add_argument("--branch", default="main",
                   help="TrainingBranch: main|holdout|longitudinal (PLAN.md §7.1).")
    p.add_argument("--split-mode", default="disjoint",
                   help="SplitMode: disjoint (PD) | within (PW) | global (PLAN.md §3.8/§9).")
    p.add_argument("--num-features", type=int, default=512)
    p.add_argument("--num-hidden-layers", type=int, default=6)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-cache", type=int, default=150)
    p.add_argument("--cluster-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--clear-cache-each-epoch", action="store_true")
    p.add_argument("--stratified", action="store_true",
                   help="Use PviBatchSampler (default: plain shuffled DataLoader).")
    p.add_argument("--logdir", default="foundation-ssl-pretrain")
    p.add_argument("--ds-root", default=None)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--cache-num-workers", type=int, default=None)
    p.add_argument("--probe-every", type=int, default=10,
                   help="Run frozen linear BP probe every N epochs (0=off).")
    p.add_argument("--probe-subjects", default="subject001,subject013,subject020")
    p.add_argument("--probe-epochs", type=int, default=5)
    p.add_argument("--subject-adversary", action="store_true")
    p.add_argument("--subject-adversary-weight", type=float, default=1.0)
    p.add_argument("--dann-gamma", type=float, default=10.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cache_num_workers = args.cache_num_workers if args.cache_num_workers is not None else FoundationConfig.cache_num_workers
    cfg = FoundationConfig(input_mode=args.input_mode,
                           output_mode=args.output_mode,
                           mask_key=args.mask_key,
                           branch=args.branch,
                           split_mode=args.split_mode,
                           arch=args.arch,
                           ssl_arch=args.ssl_arch,
                           num_features=args.num_features,
                           num_hidden_layers=args.num_hidden_layers,
                           mask_ratio=args.mask_ratio,
                           horizon=args.horizon,
                           batch_size=args.batch_size,
                           max_cache=args.max_cache,
                           cluster_size=args.cluster_size,
                           stratified=args.stratified,
                           persistent_h5=False,
                           num_workers=args.num_workers,
                           cache_root=args.cache_root,
                           cache_num_workers=cache_num_workers,
                           use_amp=not args.no_amp,
                           clear_cache_every_epoch=args.clear_cache_each_epoch,
                           max_epochs=args.max_epochs,
                           probe_every=args.probe_every,
                           probe_subjects=args.probe_subjects,
                           probe_epochs=args.probe_epochs,
                           subject_adversary=args.subject_adversary,
                           subject_adversary_weight=args.subject_adversary_weight,
                           dann_gamma=args.dann_gamma)
    try:
        main(cfg, logdir=args.logdir, ds_root=args.ds_root)
    except FileNotFoundError as e:
        print(f"[ssl] {e}", flush=True)
