"""Domain-adversarial subject-invariance via gradient reversal (DANN/GRL)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.primitives import NUM_SUBJECTS

SUBJECT_ADVERSARY_HEAD = "subject_adversary"


class GradientReversalFunction(torch.autograd.Function):
    """Identity forward; negated-and-scaled gradient on backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


def dann_lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    """Standard sigmoid ramp-up in [0, 1). ``progress`` in [0, 1]."""
    import math
    progress = float(max(0.0, min(1.0, progress)))
    return float(2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class SubjectAdversaryHead(nn.Module):
    """GRL -> linear classifier over subject IDs."""

    def __init__(
        self,
        feature_size: int,
        num_subjects: int = NUM_SUBJECTS,
        lambda_: float = 0.0,
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=lambda_)
        self.classifier = nn.Linear(feature_size, num_subjects)

    def set_lambda(self, lambda_: float) -> None:
        self.grl.set_lambda(lambda_)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.grl(features))


def attach_subject_adversary(
    model,
    num_subjects: int = NUM_SUBJECTS,
    lambda_: float = 0.0,
) -> SubjectAdversaryHead:
    """Wire a subject-adversary head onto ``model`` via the aux-head API."""
    head = SubjectAdversaryHead(model.feature_size, num_subjects=num_subjects, lambda_=lambda_)
    model.add_aux_head(SUBJECT_ADVERSARY_HEAD, module=head)
    return head


def subject_labels_from_batch(batch: dict) -> torch.Tensor:
    """Read 0-based class labels from ``batch['subject_idx']`` (1–100 encoded)."""
    if "subject_idx" not in batch:
        raise KeyError("batch missing 'subject_idx'; build cohort with subject metadata first.")
    labels = batch["subject_idx"].long().view(-1) - 1
    return labels


def subject_adversary_loss(
    model,
    batch: dict,
    lambda_: float,
    head_name: str = SUBJECT_ADVERSARY_HEAD,
) -> torch.Tensor:
    """Cross-entropy subject classification on core features through the GRL head."""
    if head_name not in model.aux_heads:
        raise KeyError(f"aux head '{head_name}' not attached")
    seqs, stats, _ = model.process_batch(batch)
    features = model.forward_core(seqs, stats)
    head: SubjectAdversaryHead = model.aux_heads[head_name]
    head.set_lambda(lambda_)
    logits = head(features)
    labels = subject_labels_from_batch(batch)
    return F.cross_entropy(logits, labels)


def run_subject_adversary_step(
    model,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    lambda_: float,
    weight: float = 1.0,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    """One adversarial backward pass (after the main task loss step)."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
        loss = weight * subject_adversary_loss(model, batch, lambda_=lambda_)
    if scaler is not None and use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.detach())
