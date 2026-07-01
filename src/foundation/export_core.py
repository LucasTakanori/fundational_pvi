"""Extract ``foundation_core.pt`` from a training checkpoint (no dataset load)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.base_model import load_core_from_state_dict


def core_state_from_checkpoint(model_state: dict[str, torch.Tensor],
                               prefix: str = "core.") -> dict[str, torch.Tensor]:
    """Pull ``core.*`` tensors from a full model state dict."""
    out = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"No '{prefix}*' keys found in checkpoint model state.")
    return out


def resolve_checkpoint(ckpt_dir: Path, use_best: bool) -> Path:
    ckpt_dir = Path(ckpt_dir)
    if use_best:
        best = ckpt_dir / "dataset_parquet_checkpoints_best.pth"
        if best.is_file():
            return best
        lazy_best = ckpt_dir / "dataset_lazy_checkpoints_best.pth"
        if lazy_best.is_file():
            return lazy_best
        raise FileNotFoundError(f"No *_checkpoints_best.pth under {ckpt_dir}")
    for name in (
        "dataset_parquet_checkpoints.pth",
        "dataset_lazy_checkpoints.pth",
    ):
        path = ckpt_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No checkpoint .pth found under {ckpt_dir}")


def export_core_from_checkpoint(checkpoint: str | Path,
                                output: str | Path | None = None,
                                use_best: bool = False,
                                map_location: str | torch.device = "cpu") -> Path:
    """Write ``foundation_core.pt`` from a workflow checkpoint file."""
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = resolve_checkpoint(ckpt_path, use_best=use_best)

    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint {ckpt_path} has no 'model' key.")

    core_state = core_state_from_checkpoint(ckpt["model"])
    epoch = ckpt.get("epoch", "?")
    test_acc = None
    if "tracker" in ckpt and ckpt["tracker"].get("test_accuracy"):
        test_acc = ckpt["tracker"]["test_accuracy"][-1]

    if output is None:
        output = ckpt_path.resolve().parents[1] / "foundation_core.pt"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(core_state, output)
    msg = f"[export_core] saved {output}  (from {ckpt_path.name}, epoch={epoch}"
    if test_acc is not None:
        msg += f", test_accuracy={test_acc:.4f}"
    print(msg + ")", flush=True)
    return output


def export_core_via_model(checkpoint: str | Path,
                          shapes: dict,
                          num_features: int = 512,
                          num_hidden_layers: int = 6,
                          output: str | Path | None = None,
                          map_location: str | torch.device = "cpu") -> Path:
    """Load full ``PviFoundationModel`` when you need strict shape validation."""
    from src.foundation.foundation_model import PviFoundationModel

    ckpt_path = Path(checkpoint)
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model = PviFoundationModel(
        shapes,
        num_features=num_features,
        num_hidden_layers=num_hidden_layers,
        verbose=False,
    )
    load_core_from_state_dict(model, ckpt["model"])
    if output is None:
        output = ckpt_path.resolve().parents[1] / "foundation_core.pt"
    output = Path(output)
    torch.save(model.core_state_dict(), output)
    print(f"[export_core] saved {output} via model load", flush=True)
    return output


def shapes_from_configs(config_json: str | Path) -> dict:
    with open(config_json, encoding="utf-8") as f:
        cfg = json.load(f)
    shapes = cfg["dataset"]["shapes"]
    return {k: tuple(v) for k, v in shapes.items()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        default="artifacts/foundation-pretrain/main/checkpoints",
        help="Checkpoint .pth file or checkpoints/ directory.",
    )
    p.add_argument(
        "--output",
        default="artifacts/foundation-pretrain/main/foundation_core.pt",
        help="Output path for foundation_core.pt",
    )
    p.add_argument("--best", action="store_true", help="Prefer *_checkpoints_best.pth")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    export_core_from_checkpoint(
        checkpoint=args.checkpoint,
        output=args.output,
        use_best=args.best,
    )


if __name__ == "__main__":
    main()
