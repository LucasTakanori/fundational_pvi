"""Cohort SSL pretraining -> the shared foundation core U.

Trains `PviSSLModel` (masked reconstruction + forecasting) on the pooled
population (input only, no BP labels), then saves the shared `PviCore` for
transfer into `PviFoundationModel`.

Run from the repo root:

    python -m src.foundation.ssl_pretrain --input-mode signal --max-epochs 50

Datasets are discovered under `$PVIPROJECT_ROOT/datasets/main` (see README).
"""

import argparse

from src.packages import *  # noqa: F401,F403  (torch, optim, Path, ...)

from src.utils.primitives import DEFAULT_TRAIN_DEVICE, DEFAULT_TRAIN_DTYPE
from src.pipeline.data_discovery import ProjectPathManager

from src.foundation.config import FoundationConfig
from src.foundation.ssl import PviSSLModel
from src.foundation.pretrain import build_population_dataset


def _to_device(batch: dict, device, dtype) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v
    return out


def main(cfg: FoundationConfig = None,
         logdir: str = "foundation-ssl-pretrain",
         ds_root=None,
         device=None) -> PviSSLModel:
    cfg = cfg or FoundationConfig()
    device = device or DEFAULT_TRAIN_DEVICE

    pm = ProjectPathManager(branch="main", target=logdir)
    ds = build_population_dataset(cfg, ds_root=ds_root)
    loaders = ds.get_dataloaders()

    model = PviSSLModel(ds.shapes,
                        num_features=cfg.num_features,
                        num_hidden_layers=cfg.num_hidden_layers,
                        mask_ratio=cfg.mask_ratio,
                        horizon=cfg.horizon,
                        lambda_mask=cfg.lambda_mask,
                        lambda_forecast=cfg.lambda_forecast,
                        diff=cfg.diff,
                        use_stats=cfg.use_stats).to(device=device, dtype=DEFAULT_TRAIN_DTYPE)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for epoch in range(1, cfg.max_epochs + 1):
        running = {"total": 0.0, "mask": 0.0, "forecast": 0.0}
        n = 0
        for batch in loaders["train"]:
            batch = _to_device(batch, device, DEFAULT_TRAIN_DTYPE)
            losses = model.pretext_loss(batch)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            for k in running:
                running[k] += float(losses[k])
            n += 1
        if n:
            msg = "  ".join(f"{k}={running[k] / n:.4f}" for k in running)
            print(f"[ssl] epoch {epoch:4d}/{cfg.max_epochs}  {msg}")

    core_path = pm.logdir / "foundation_core_U.pt"
    torch.save(model.core_state_dict(), core_path)
    print(f"[ssl] Saved SSL-pretrained core (U) -> {core_path}")
    return model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-mode", default="signal")
    p.add_argument("--output-mode", default="waveform")  # unused by SSL; keeps dataset happy
    p.add_argument("--mask-key", default="mask05")
    p.add_argument("--num-features", type=int, default=200)
    p.add_argument("--num-hidden-layers", type=int, default=4)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--logdir", default="foundation-ssl-pretrain")
    p.add_argument("--ds-root", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg = FoundationConfig(input_mode=args.input_mode,
                           output_mode=args.output_mode,
                           mask_key=args.mask_key,
                           num_features=args.num_features,
                           num_hidden_layers=args.num_hidden_layers,
                           mask_ratio=args.mask_ratio,
                           horizon=args.horizon,
                           batch_size=args.batch_size,
                           max_epochs=args.max_epochs)
    try:
        main(cfg, logdir=args.logdir, ds_root=args.ds_root)
    except FileNotFoundError as e:
        print(f"[ssl] {e}")
