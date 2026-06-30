"""Experiment drivers: data-efficiency / foundation-vs-individual curves.

Ties the building blocks together for Exp A/B (and G): sweep a training-data
budget, train each method (individual end-to-end vs foundation transfer with a
frozen pretrained core), score on the held-out test set, and emit records that
`src/analysis/budget_curves.aggregate_budget_results` turns into curves.

The driver depends only on a small dataset interface
(`set_train_budget`, `get_dataloaders`, `shapes`), which `PviConfiguredDataset`
satisfies, and a dict of model factories `{method_name: factory(shapes) -> model}`.
For a foundation method, the factory loads + freezes the pretrained core (so only
the readout trains); for the individual baseline it returns a fresh model.
"""

from src.packages import *  # noqa: F401,F403  (torch, optim, nn, ...)

from src.models.loss_functions import MorphologyLoss
from src.foundation.evaluation import evaluate


def train_model(model, train_loader, epochs: int = 1, lr: float = 5e-4,
                weight_decay: float = 1e-2, mse_weight: float = 0.2, device=None):
    """Train a BasePviLearner on BP. Only trainable params are optimised, so a
    frozen core (foundation transfer) updates the readout alone."""
    if device is not None:
        model = model.to(device)
    model.train()

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


def run_budget_curve(dataset,
                     model_factories: dict,
                     budgets_min,
                     seconds_per_sequence: float,
                     seeds=(0,),
                     epochs: int = 1,
                     device=None,
                     subject: str = None) -> list[dict]:
    """Sweep budget x seed x method; return one record per (budget, seed, method).

    Each record: {method, budget_min, seed, [subject], cc_abs, amae, armse}.
    """
    records = []
    for budget in budgets_min:
        for seed in seeds:
            dataset.set_train_budget(minutes=budget,
                                     seconds_per_sequence=seconds_per_sequence,
                                     seed=seed)
            loaders = dataset.get_dataloaders()
            for method, factory in model_factories.items():
                model = factory(dataset.shapes)
                train_model(model, loaders["train"], epochs=epochs, device=device)
                rec = {"method": method, "budget_min": budget, "seed": seed}
                if subject is not None:
                    rec["subject"] = subject
                rec.update(evaluate(model, loaders["test"], device=device))
                records.append(rec)
    return records
