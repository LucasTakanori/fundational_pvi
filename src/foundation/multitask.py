"""Multi-task readout: BP (primary) + auxiliary maneuver / HR heads.

Attaches light auxiliary heads on the shared core (via BasePviLearner's
`add_aux_head`) and computes a combined loss in a single core pass:

    L = w_bp * BP + w_maneuver * CE(maneuver) + w_hr * MSE(HR)

The auxiliary heads probe hemodynamic-state encoding (maneuver classifier:
baseline/valsalva/pressor) and cardiac timing (HR regressor), powering the
Exp D/E interpretability analyses. They share the same core features as the BP
readout, so adding them costs one extra linear per head.
"""

from dataclasses import dataclass, field

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)

from src.models.base_model import BasePviLearner
from src.models.loss_functions import MorphologyLoss

MANEUVER_HEAD = "maneuver"
HR_HEAD = "hr"


@dataclass
class MultiTaskWeights:
    bp: float = 1.0
    maneuver: float = 0.2
    hr: float = 0.2


def attach_multitask_heads(model: BasePviLearner,
                           num_maneuvers: int = 3,
                           hr: bool = True,
                           hidden: int = 0) -> BasePviLearner:
    """Add maneuver (classifier) and HR (regressor) heads on the shared core."""
    if num_maneuvers and num_maneuvers > 0:
        model.add_aux_head(MANEUVER_HEAD, out_features=num_maneuvers, hidden=hidden)
    if hr:
        model.add_aux_head(HR_HEAD, out_features=1, hidden=hidden)
    return model


def multitask_loss(model: BasePviLearner,
                   input_sequences: dict[str, torch.Tensor],
                   input_stats: torch.Tensor,
                   bp_targets: torch.Tensor,
                   maneuver_targets: torch.Tensor = None,
                   hr_targets: torch.Tensor = None,
                   weights: MultiTaskWeights = None,
                   bp_loss_fn: nn.Module = None) -> dict[str, torch.Tensor]:
    """Combined BP + auxiliary loss computed from a single core pass."""
    weights = weights or MultiTaskWeights()
    bp_loss_fn = bp_loss_fn or MorphologyLoss(base_loss=nn.MSELoss(), base_weight=0.2)

    features = model.forward_core(input_sequences, input_stats)

    out: dict[str, torch.Tensor] = {}
    bp_pred = model.forward_readout(features, input_stats)
    out["bp"] = bp_loss_fn(bp_pred, bp_targets)
    total = weights.bp * out["bp"]

    if maneuver_targets is not None:
        if MANEUVER_HEAD not in model.aux_heads:
            raise KeyError("maneuver head not attached; call attach_multitask_heads first.")
        logits = model.aux_heads[MANEUVER_HEAD](features)
        out["maneuver"] = nn.functional.cross_entropy(logits, maneuver_targets)
        total = total + weights.maneuver * out["maneuver"]

    if hr_targets is not None:
        if HR_HEAD not in model.aux_heads:
            raise KeyError("hr head not attached; call attach_multitask_heads first.")
        hr_pred = model.aux_heads[HR_HEAD](features)
        out["hr"] = nn.functional.mse_loss(hr_pred, hr_targets.view_as(hr_pred))
        total = total + weights.hr * out["hr"]

    out["total"] = total
    return out
