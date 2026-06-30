"""Self-supervised pretraining (core U): masked reconstruction + forecasting.

A multi-objective pretext that learns the shared foundation core *without* BP
labels, on the PVI input alone:

  * masked reconstruction (MAE/BERT-style): hide random channel x time entries
    and reconstruct them from the encoded (masked) sequence;
  * causal forecasting: encode a past window and predict the next `horizon`
    steps.

The two pretext heads are discarded after pretraining; only the shared
`PviCore` is kept and transferred into `PviFoundationModel` for BP readout
(`load_core_state_dict`). The encoder/core is the same `PviCore` used by the
supervised foundation model, so cores are interchangeable (matching input shape).
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)

from src.models.base_model import BasePviLearner
from src.foundation.core import PviCore


def random_channel_time_mask(x: torch.Tensor,
                             mask_ratio: float = 0.5,
                             generator: torch.Generator = None):
    """Mask random channel x time entries of `x`.

    `x` is (B, C, T) or (B, C, H, W, T); masking is applied per (sample, channel,
    time) and broadcast across any spatial dims (a masked channel/time hides the
    whole image at that step). Returns `(x_masked, mask)` with masked entries set
    to 0 and a boolean `mask` (True = hidden) broadcast to `x`'s shape.
    """
    B, C, T = x.shape[0], x.shape[1], x.shape[-1]
    base = torch.rand(B, C, T, generator=generator, device=x.device) < mask_ratio  # (B,C,T)

    if x.dim() == 3:
        m = base
    else:  # (B, C, H, W, T) -> broadcast over H, W
        m = base[:, :, None, None, :]

    x_masked = x.masked_fill(m, 0.0)
    return x_masked, m.expand_as(x)


def split_past_future(x: torch.Tensor, horizon: int):
    """Split `x` along the last (time) dim into (past, future) windows."""
    T = x.shape[-1]
    if not 0 < horizon < T:
        raise ValueError(f"horizon must be in (0, T={T}); got {horizon}.")
    return x[..., :T - horizon], x[..., T - horizon:]


class PviSSLModel(BasePviLearner):
    """PviCore encoder + two pretext heads (reconstruction, forecasting)."""

    def __init__(self,
                 data_shapes: dict[str, tuple[int, ...]],
                 num_features: int = 200,
                 num_hidden_layers: int = 4,
                 mask_ratio: float = 0.5,
                 horizon: int = 10,
                 lambda_mask: float = 1.0,
                 lambda_forecast: float = 1.0,
                 diff: int = 2,
                 use_stats: bool = True,
                 verbose: bool = True,
                 ) -> None:
        super().__init__(data_shapes=data_shapes, diff=diff,
                         use_stats=use_stats, verbose=verbose)

        self.num_features = int(num_features)
        self.num_hidden_layers = int(num_hidden_layers)
        self.mask_ratio = float(mask_ratio)
        self.horizon = int(horizon)
        self.lambda_mask = float(lambda_mask)
        self.lambda_forecast = float(lambda_forecast)

        self._make_layers()

    # ------------------------------------------------------------------ build
    def _spatial(self) -> int:
        # number of pixels per channel per time step (1 for 1-D signals)
        return self.input_shape[1] * self.input_shape[2] if self.input_ndims == 4 else 1

    def _make_layers(self) -> None:
        per_t = self.num_channels * self._spatial()   # processed channels x pixels
        self._per_t = per_t
        self.recon_size = per_t * self.sequence_length
        self.forecast_size = per_t * self.horizon

        flatten_size = self.recon_size + self.stats_size
        self.core = PviCore(flatten_size,
                            num_features=self.num_features,
                            num_hidden_layers=self.num_hidden_layers)
        self.feature_size = self.num_features

        # Pretext heads (discarded after pretraining).
        self.recon_head = nn.Linear(self.num_features, self.recon_size)
        self.forecast_head = nn.Linear(self.num_features, self.forecast_size)
        # BasePviLearner.readout points at the reconstruction decoder.
        self.readout = self.recon_head

    # --------------------------------------------------------------- encoding
    def _flatten_with_stats(self, s: torch.Tensor, input_stats: torch.Tensor) -> torch.Tensor:
        s = s.flatten(start_dim=1)
        if input_stats.numel():
            s = torch.hstack([s, input_stats.flatten(start_dim=1)])
        return s

    def forward_core(self,
                     input_sequences: dict[str, torch.Tensor],
                     input_stats: torch.Tensor) -> torch.Tensor:
        """Encode the (unmasked) sequence -> shared core features."""
        s = self._process_sequence(input_sequences)
        return self.core(self._flatten_with_stats(s, input_stats))

    encode = forward_core

    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        return self.recon_head(features)

    # ---------------------------------------------------------- pretext loss
    def pretext_loss(self, batch: dict[str, torch.Tensor],
                     generator: torch.Generator = None) -> dict[str, torch.Tensor]:
        input_sequences, input_stats, _ = self.process_batch(batch)
        s = self._process_sequence(input_sequences)        # (B, C, [H, W,] T)

        # --- masked reconstruction ---
        s_masked, mask = random_channel_time_mask(s, self.mask_ratio, generator=generator)
        feats_m = self.core(self._flatten_with_stats(s_masked, input_stats))
        recon = self.recon_head(feats_m).view_as(s)
        if mask.any():
            loss_mask = nn.functional.mse_loss(recon[mask], s[mask])
        else:
            loss_mask = nn.functional.mse_loss(recon, s)

        # --- causal forecasting (encode past -> predict future window) ---
        past, future = split_past_future(s, self.horizon)
        past_padded = nn.functional.pad(past, (0, self.horizon))   # pad time back to T
        feats_p = self.core(self._flatten_with_stats(past_padded, input_stats))
        pred = self.forecast_head(feats_p)
        loss_forecast = nn.functional.mse_loss(pred, future.flatten(start_dim=1))

        total = self.lambda_mask * loss_mask + self.lambda_forecast * loss_forecast
        return {"total": total, "mask": loss_mask, "forecast": loss_forecast}
