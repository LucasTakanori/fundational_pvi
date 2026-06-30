import numpy as np
from scipy import stats
import torch

import ot

def bp_accuracy(predictions: torch.Tensor | np.ndarray,
                targets: torch.Tensor|np.ndarray) -> float:

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Time is always last dimension
    fmin = lambda array: np.min(array, axis=-1).flatten()
    fmax = lambda array: np.max(array, axis=-1).flatten()

    dbp_predictions = fmin(predictions)
    dbp_targets = fmin(targets)

    sbp_predictions = fmax(predictions)
    sbp_targets = fmax(targets)

    if len(dbp_predictions) < 2 or len(sbp_predictions) < 2:
        # some batches only have 1 sample, so not enough datapoint to compute pearsonr
        return 0.0

    dbp_r = stats.pearsonr(dbp_predictions, dbp_targets).statistic
    sbp_r = stats.pearsonr(sbp_predictions, sbp_targets).statistic

    return float(0.5*(dbp_r + sbp_r))

def metrics_waveform(predictions: torch.Tensor,
                     targets: torch.Tensor) -> dict[str, float]:

    err = (predictions - targets).detach().cpu()

    dict_out = {
        'amae': float(err.abs().mean(dim=-1).mean()),
        'armse': float(err.square().mean(dim=-1).sqrt().mean())
    }

    return dict_out

def metrics_ensemble(D1: torch.Tensor,
                     D2: torch.Tensor) -> dict[str, float]:
    D1 = D1.detach().cpu()
    D2 = D2.detach().cpu()

    loss = ot.solve_sample(D1, D2, metric='euclidean').value.item()
    return loss

def metrics_fiducial(predictions: torch.Tensor,
                     targets: torch.Tensor) -> dict[str, float]:

    def _compute_correlation(X, Y, tag: str) -> dict[str, float]:
        rho = stats.pearsonr(X, Y).statistic
        pv = stats.pearsonr(X, Y).pvalue
        cc = 2 * rho * X.std() * Y.std() / (X.var() + Y.var() + (X.mean() - Y.mean()) ** 2)

        keys = ['r2', 'pv', 'cc']
        values = [rho**2, pv, cc]
        values = [float(v) for v in values]

        if tag:
            keys = ['_'.join([tag, k]) for k in keys]

        dict_out = dict(zip(keys,values))

        return dict_out

    def _compute_deviation(X, Y, tag: str) -> dict[str, float]:
        err = X - Y

        percent = lambda err, bound: sum(err.abs() < bound) / len(err) * 100

        keys = ['mae', 'sd', 'tol05', 'tol10', 'tol15']
        values = [err.abs().mean(),
                  err.abs().std(),
                  percent(err.abs(), 5),
                  percent(err.abs(), 10),
                  percent(err.abs(), 15)]
        values = [float(v) for v in values]

        if tag:
            keys = ['_'.join([tag, k]) for k in keys]

        dict_out = dict(zip(keys,values))

        return dict_out

    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()

    sbp = lambda tensor: tensor.max(dim=-1).values.flatten()
    dbp = lambda tensor: tensor.min(dim=-1).values.flatten()

    dict_out = {}
    dict_out |= _compute_correlation(X=sbp(predictions),
                                     Y=sbp(targets),
                                     tag='sbp')

    dict_out |= _compute_deviation(X=sbp(predictions),
                                   Y=sbp(targets),
                                   tag='sbp')

    dict_out |= _compute_correlation(X=dbp(predictions),
                                     Y=dbp(targets),
                                     tag='dbp')

    dict_out |= _compute_deviation(X=dbp(predictions),
                                   Y=dbp(targets),
                                   tag='dbp')

    return dict_out