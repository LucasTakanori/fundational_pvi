"""Self-supervised pretraining (core U): masked reconstruction + forecasting.

Architectures:
  * ``mlp`` — ``PviCore`` encoder (scaffold).
  * ``mae`` — token Transformer encoder (production Core U).
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)

from src.models.base_model import BasePviLearner
from src.models.mae_transformer import _MAEEncoder
from src.foundation.core import PviCore
from src.foundation.arch import normalize_arch


def random_channel_time_mask(x: torch.Tensor,
                             mask_ratio: float = 0.5,
                             generator: torch.Generator = None):
    B, C, T = x.shape[0], x.shape[1], x.shape[-1]
    base = torch.rand(B, C, T, generator=generator, device=x.device) < mask_ratio

    if x.dim() == 3:
        m = base
    else:
        m = base[:, :, None, None, :]

    x_masked = x.masked_fill(m, 0.0)
    return x_masked, m.expand_as(x)


def split_past_future(x: torch.Tensor, horizon: int):
    T = x.shape[-1]
    if not 0 < horizon < T:
        raise ValueError(f"horizon must be in (0, T={T}); got {horizon}.")
    return x[..., :T - horizon], x[..., T - horizon:]


class PviSSLModel(BasePviLearner):
    SUPPORTED_ARCHES = frozenset({"mlp", "mae"})

    def __init__(self,
                 data_shapes: dict[str, tuple[int, ...]],
                 arch: str = "mlp",
                 num_features: int = 200,
                 num_hidden_layers: int = 4,
                 mask_ratio: float = 0.5,
                 horizon: int = 10,
                 lambda_mask: float = 1.0,
                 lambda_forecast: float = 1.0,
                 diff: int = 2,
                 use_stats: bool = True,
                 verbose: bool = True,
                 d_model: int = 64,
                 num_layers: int = 2,
                 ) -> None:
        self.arch = normalize_arch(arch)
        if self.arch not in self.SUPPORTED_ARCHES:
            raise ValueError(f"SSL arch must be one of {sorted(self.SUPPORTED_ARCHES)}.")

        super().__init__(data_shapes=data_shapes, diff=diff,
                         use_stats=use_stats, verbose=verbose)

        self.num_features = int(num_features)
        self.num_hidden_layers = int(num_hidden_layers)
        self.mask_ratio = float(mask_ratio)
        self.horizon = int(horizon)
        self.lambda_mask = float(lambda_mask)
        self.lambda_forecast = float(lambda_forecast)
        self.d_model = int(d_model)
        self.mae_num_layers = int(num_layers)

        self._make_layers()

    def _spatial(self) -> int:
        return self.input_shape[1] * self.input_shape[2] if self.input_ndims == 4 else 1

    def _per_time_features(self) -> int:
        return self.num_channels * self._spatial()

    def _tokens(self, input_sequences: dict[str, torch.Tensor]) -> torch.Tensor:
        s = self._process_sequence(input_sequences)
        if self.input_ndims == 4:
            s = s.flatten(start_dim=1, end_dim=-2)
        return s.transpose(-2, -1)

    def _make_layers(self) -> None:
        per_t = self._per_time_features()
        self._per_t = per_t
        self.recon_size = per_t * self.sequence_length
        self.forecast_size = per_t * self.horizon

        if self.arch == "mlp":
            flatten_size = self.recon_size + self.stats_size
            self.core = PviCore(flatten_size,
                                num_features=self.num_features,
                                num_hidden_layers=self.num_hidden_layers)
            self.feature_size = self.num_features
        else:
            p = self.d_model
            nheads = max([n for n in range(1, 10) if (not p % n) and (not n % 2)])
            self.core = _MAEEncoder(per_t, self.d_model, self.sequence_length,
                                    num_layers=self.mae_num_layers, num_heads=nheads)
            self.feature_size = self.d_model * self.sequence_length

        self.recon_head = nn.Linear(self.feature_size, self.recon_size)
        self.forecast_head = nn.Linear(self.feature_size, self.forecast_size)
        self.readout = self.recon_head

    def _flatten_with_stats(self, s: torch.Tensor, input_stats: torch.Tensor) -> torch.Tensor:
        s = s.flatten(start_dim=1)
        if self.arch == "mlp" and input_stats.numel():
            s = torch.hstack([s, input_stats.flatten(start_dim=1)])
        return s

    def _encode_sequence(self, s: torch.Tensor, input_stats: torch.Tensor) -> torch.Tensor:
        if self.arch == "mlp":
            return self.core(self._flatten_with_stats(s, input_stats))
        return self.core(self._tokens_from_tensor(s)).flatten(start_dim=1)

    def _tokens_from_tensor(self, s: torch.Tensor) -> torch.Tensor:
        if self.input_ndims == 4:
            s = s.flatten(start_dim=1, end_dim=-2)
        return s.transpose(-2, -1)

    def forward_core(self,
                     input_sequences: dict[str, torch.Tensor],
                     input_stats: torch.Tensor) -> torch.Tensor:
        s = self._process_sequence(input_sequences)
        return self._encode_sequence(s, input_stats)

    encode = forward_core

    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        return self.recon_head(features)

    def pretext_loss(self, batch: dict[str, torch.Tensor],
                     generator: torch.Generator = None) -> dict[str, torch.Tensor]:
        input_sequences, input_stats, _ = self.process_batch(batch)
        s = self._process_sequence(input_sequences)

        s_masked, mask = random_channel_time_mask(s, self.mask_ratio, generator=generator)
        feats_m = self._encode_sequence(s_masked, input_stats)
        recon = self.recon_head(feats_m).view_as(s)
        if mask.any():
            loss_mask = nn.functional.mse_loss(recon[mask], s[mask])
        else:
            loss_mask = nn.functional.mse_loss(recon, s)

        past, future = split_past_future(s, self.horizon)
        past_padded = nn.functional.pad(past, (0, self.horizon))
        feats_p = self._encode_sequence(past_padded, input_stats)
        pred = self.forecast_head(feats_p)
        loss_forecast = nn.functional.mse_loss(pred, future.flatten(start_dim=1))

        total = self.lambda_mask * loss_mask + self.lambda_forecast * loss_forecast
        return {"total": total, "mask": loss_mask, "forecast": loss_forecast}
