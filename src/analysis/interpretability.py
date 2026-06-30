"""Digital-twin & interpretability analyses (Exp D/E).

  * in-silico perturbation: nudge input channels and measure the BP response
    -> channel sensitivity (the "digital twin" probe);
  * gradient saliency: |d output / d input| per channel;
  * latent extraction: collect shared-core features for latent-structure analysis
    (UMAP by maneuver/HR/BP, Extended-Data-Fig-2 analog);
  * functional barcode: predict subject/maneuver from per-subject readout weights
    with a logistic-regression probe.
"""

from src.packages import *  # noqa: F401,F403  (torch, np, ...)


@torch.no_grad()
def input_sensitivity(model,
                      input_sequences: dict[str, torch.Tensor],
                      input_stats: torch.Tensor,
                      eps: float = 0.1) -> dict[str, list[float]]:
    """Per-channel BP sensitivity via a constant +eps shift on each channel.

    Returns {group: [mean |Delta output| per channel]} for pviHP / pviLP.
    """
    model.eval()
    base = model(input_sequences, input_stats)

    sens: dict[str, list[float]] = {}
    for key in ("pviHP", "pviLP"):
        if key not in input_sequences:
            continue
        x = input_sequences[key]
        per_channel = []
        for ch in range(x.shape[1]):
            perturbed = {k: v for k, v in input_sequences.items()}
            xp = x.clone()
            xp[:, ch] = xp[:, ch] + eps
            perturbed[key] = xp
            out = model(perturbed, input_stats)
            per_channel.append(float((out - base).abs().mean()))
        sens[key] = per_channel
    return sens


def gradient_saliency(model,
                      input_sequences: dict[str, torch.Tensor],
                      input_stats: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-channel saliency |d(sum output) / d input|, averaged over non-channel dims."""
    model.eval()
    seqs = {k: v.clone().requires_grad_(True) for k, v in input_sequences.items()}
    out = model(seqs, input_stats)
    model.zero_grad(set_to_none=True)
    out.sum().backward()

    saliency = {}
    for k, v in seqs.items():
        g = v.grad.abs()
        reduce_dims = (0,) + tuple(range(2, g.dim()))   # keep channel dim (1)
        saliency[k] = g.mean(dim=reduce_dims).detach()
    return saliency


@torch.no_grad()
def extract_latents(model, loader, label_keys=(), device=None):
    """Collect shared-core features (and optional labels) over a loader."""
    model.eval()
    feats = []
    labels = {k: [] for k in label_keys}
    for batch in loader:
        if device is not None:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        seqs, stats, _ = model.process_batch(batch)
        feats.append(model.forward_core(seqs, stats).detach().cpu())
        for k in label_keys:
            if k in batch:
                labels[k].append(batch[k].detach().cpu() if torch.is_tensor(batch[k]) else batch[k])

    Z = torch.cat(feats, dim=0)
    L = {k: torch.cat(v, dim=0) for k, v in labels.items() if v}
    return Z, L


def readout_weight_matrix(model, subjects=None):
    """Stack each subject readout's flattened weights -> (num_subjects, P) matrix."""
    subjects = list(subjects) if subjects is not None else list(model.subjects)
    rows = []
    for s in subjects:
        w = torch.cat([p.detach().flatten() for p in model.readouts[s].parameters()])
        rows.append(w)
    return torch.stack(rows, dim=0), subjects


def functional_barcode_probe(X, y, test_size: float = 0.3, seed: int = 0) -> float:
    """Logistic-regression probe: predict labels `y` from features `X`. Returns accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = X.numpy() if torch.is_tensor(X) else np.asarray(X)
    y = y.numpy() if torch.is_tensor(y) else np.asarray(y)

    strat = y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return float(clf.score(X_te, y_te))
