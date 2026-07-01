"""Experiment drivers: data-efficiency / foundation-vs-individual curves."""

from src.packages import *  # noqa: F401,F403  (torch, optim, nn, ...)

from src.models.loss_functions import MorphologyLoss
from src.foundation.evaluation import evaluate, collect_predictions
from src.analysis.decomposition import decompose_predictions, fit_affine_correction, apply_affine_correction


def _unfreeze_last_core_layers(model, n_layers: int = 1) -> None:
    """Unfreeze the last ``n_layers`` top-level core submodules for partial fine-tune."""
    if n_layers <= 0 or not hasattr(model, "core"):
        return
    for p in model.core.parameters():
        p.requires_grad_(False)
    children = list(model.core.children()) if isinstance(model.core, nn.ModuleDict) else []
    if isinstance(model.core, nn.ModuleDict):
        keys = list(model.core.keys())[-n_layers:]
        for k in keys:
            for p in model.core[k].parameters():
                p.requires_grad_(True)
    elif isinstance(model.core, nn.Sequential):
        for block in list(model.core.children())[-n_layers:]:
            for p in block.parameters():
                p.requires_grad_(True)


def train_model(model, train_loader, epochs: int = 1, lr: float = 5e-4,
                weight_decay: float = 1e-2, mse_weight: float = 0.2, device=None,
                calibration: str = "linear", partial_finetune_layers: int = 1):
    """Train a BasePviLearner on BP. Only trainable params are optimised."""
    if device is not None:
        model = model.to(device)
    model.train()

    if calibration == "partial_finetune":
        _unfreeze_last_core_layers(model, n_layers=partial_finetune_layers)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    loss_fn = MorphologyLoss(base_loss=nn.MSELoss(), base_weight=mse_weight)

    for _ in range(epochs):
        for batch in train_loader:
            if device is not None:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            seqs, stats, targets = model.process_batch(batch)
            loss = loss_fn(model(seqs, stats), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate_with_calibration(model, train_loader, test_loader, device=None) -> dict:
    """Fit affine correction on train, report corrected metrics on test."""
    preds_tr, targets_tr = collect_predictions(model, train_loader, device=device)
    scale, offset = fit_affine_correction(preds_tr, targets_tr)
    preds, targets = collect_predictions(model, test_loader, device=device)
    corrected = apply_affine_correction(preds, scale, offset)
    metrics = evaluate(model, test_loader, device=device)
    metrics.update(decompose_predictions(corrected, targets))
    metrics["calibration"] = "affine"
    metrics["affine_scale"] = scale
    metrics["affine_offset"] = offset
    return metrics


def run_budget_curve(dataset,
                     model_factories: dict,
                     budgets_min,
                     seconds_per_sequence: float,
                     seeds=(0,),
                     epochs: int = 1,
                     device=None,
                     subject: str = None,
                     calibration: str = "linear",
                     partial_finetune_layers: int = 1) -> list[dict]:
    """Sweep budget x seed x method; return one record per (budget, seed, method)."""
    records = []
    for budget in budgets_min:
        for seed in seeds:
            dataset.set_train_budget(minutes=budget,
                                     seconds_per_sequence=seconds_per_sequence,
                                     seed=seed)
            loaders = dataset.get_dataloaders()
            for method, factory in model_factories.items():
                model = factory(dataset.shapes)
                train_model(
                    model, loaders["train"], epochs=epochs, device=device,
                    calibration=calibration,
                    partial_finetune_layers=partial_finetune_layers,
                )
                rec = {"method": method, "budget_min": budget, "seed": seed,
                       "calibration": calibration}
                if subject is not None:
                    rec["subject"] = subject
                if calibration == "affine":
                    rec.update(evaluate_with_calibration(
                        model, loaders["train"], loaders["test"], device=device,
                    ))
                else:
                    preds, targets = collect_predictions(model, loaders["test"], device=device)
                    rec.update(evaluate(model, loaders["test"], device=device))
                    rec.update(decompose_predictions(preds, targets))
                records.append(rec)
    return records
