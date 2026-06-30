"""Evaluation primitives for the experiment matrix (Exp A/B/C/G).

All experiments reduce to "run a model over a dataloader and score it":
  * Exp A/B  - score a held-out subject's test loader (individual vs foundation);
  * Exp C    - score on a *different-maneuver* loader (OOD), the model having
               been trained on baseline only (`evaluate_ood`);
  * Exp G    - same scoring, comparing SSL (U) vs supervised (S) cores.

Scoring reuses `src/analysis/budget_curves.compute_run_metrics`
(cc_abs / amae / armse).
"""

from src.packages import *  # noqa: F401,F403  (torch, ...)

from src.analysis.budget_curves import compute_run_metrics


@torch.no_grad()
def collect_predictions(model, loader, device=None, head: str = None):
    """Run `model` over `loader`; return concatenated (predictions, targets)."""
    model.eval()
    preds, targets = [], []
    for batch in loader:
        if device is not None:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        seqs, stats, tgt = model.process_batch(batch)
        out = model(seqs, stats) if head is None else model(seqs, stats, head=head)
        preds.append(out.detach().cpu())
        targets.append(tgt.detach().cpu())
    if not preds:
        raise ValueError("Empty loader: nothing to evaluate.")
    return torch.cat(preds, dim=0), torch.cat(targets, dim=0)


def evaluate(model, loader, device=None, extra: dict = None) -> dict:
    """Score a model on a loader -> {cc_abs, amae, armse, **extra}."""
    preds, targets = collect_predictions(model, loader, device=device)
    metrics = compute_run_metrics(preds, targets)
    if extra:
        metrics.update(extra)
    return metrics


def evaluate_ood(model, ood_loader, maneuver: str, device=None, extra: dict = None) -> dict:
    """Out-of-distribution score on a different-maneuver loader (Exp C)."""
    tag = {"maneuver": maneuver, "ood": True}
    if extra:
        tag.update(extra)
    return evaluate(model, ood_loader, device=device, extra=tag)
