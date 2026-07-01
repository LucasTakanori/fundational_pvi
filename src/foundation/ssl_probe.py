"""Frozen linear-probe monitoring during SSL pretraining (PLAN.md §7.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.foundation.evaluation import evaluate
from src.foundation.transfer import build_subject_dataset
from src.foundation.config import FoundationConfig


class _ProbeReadout(nn.Module):
    """Minimal linear readout on frozen SSL core features."""

    def __init__(self, ssl_model, readout: nn.Linear) -> None:
        super().__init__()
        self.ssl_model = ssl_model
        self.readout = readout

    def process_batch(self, batch):
        return self.ssl_model.process_batch(batch)

    def forward(self, input_sequences, input_stats):
        feats = self.ssl_model.forward_core(input_sequences, input_stats)
        return self.readout(feats)


def run_ssl_linear_probe(
    ssl_model,
    cfg: FoundationConfig,
    subjects: list[str],
    device=None,
    ds_root=None,
) -> dict[str, float]:
    """Fit a cheap linear readout on fixed labeled subjects; return mean probe metrics."""
    device = device or torch.device("cpu")
    was_training = ssl_model.training
    ssl_model.eval()
    for p in ssl_model.core.parameters():
        p.requires_grad_(False)

    readout = nn.Linear(ssl_model.feature_size, ssl_model.output_size).to(device)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    train_loaders = []
    test_loaders = []
    shapes = None
    for subject in subjects:
        ds = build_subject_dataset(cfg, subject, ds_root=ds_root)
        shapes = ds.shapes
        loaders = ds.get_dataloaders()
        train_loaders.append(loaders["train"])
        test_loaders.append(loaders["test"])

    probe = _ProbeReadout(ssl_model, readout).to(device)
    probe.train()
    for _ in range(cfg.probe_epochs):
        for loader in train_loaders:
            for bi, batch in enumerate(loader):
                if bi >= cfg.probe_max_batches:
                    break
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                seqs, stats, targets = ssl_model.process_batch(batch)
                pred = readout(ssl_model.forward_core(seqs, stats))
                loss = loss_fn(pred, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    all_metrics = []
    probe.eval()
    with torch.no_grad():
        for loader in test_loaders:
            metrics = evaluate(probe, loader, device=device)
            all_metrics.append(metrics)

    if was_training:
        ssl_model.train()
    for p in ssl_model.core.parameters():
        p.requires_grad_(True)

    if not all_metrics:
        return {"probe_cc_abs": float("nan"), "probe_amae": float("nan")}

    cc = sum(m["cc_abs"] for m in all_metrics) / len(all_metrics)
    amae = sum(m["amae"] for m in all_metrics) / len(all_metrics)
    return {"probe_cc_abs": cc, "probe_amae": amae, "probe_subjects": len(subjects)}
