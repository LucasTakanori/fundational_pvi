"""Architecture candidate 2: masked spatiotemporal Transformer (MAE-style).

A conv-less, tokenized encoder: each time step becomes a token (channels, or
channels x pixels for images), linearly embedded, position-encoded, and passed
through a Transformer encoder. Pairs naturally with the SSL masked-reconstruction
pretext (hide channel x time tokens) and gives interpretable attention.

Fits the BasePviLearner core/readout contract:
  * core    = token embedding + learnable positional encoding + Transformer;
  * readout = MLP head (stats injected pre-readout).
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)

from src.models.base_model import BasePviLearner
from src.models.attn_models import Transformer


class _MAEEncoder(nn.Module):
    def __init__(self, feat_per_t: int, d_model: int, seq_len: int,
                 num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(feat_per_t, d_model)
        self.pos = nn.Parameter(torch.zeros(seq_len, d_model))
        self.transformer = Transformer(d_model, num_layers=num_layers,
                                       num_heads=num_heads, dim_mlp=4 * d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, feat_per_t) -> (B, T, d_model)
        x = self.embed(tokens) + self.pos[: tokens.shape[1]]
        return self.transformer(x)


class PviMaskedTransformer(BasePviLearner):
    def __init__(self,
                 data_shapes: dict[str, tuple[int, ...]],
                 d_model: int = 64,
                 num_layers: int = 2,
                 mlp_depth: int = 2,
                 diff: int = 2,
                 use_stats: bool = True,
                 verbose: bool = True,
                 ) -> None:
        super().__init__(data_shapes=data_shapes, diff=diff,
                         use_stats=use_stats, verbose=verbose)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.mlp_depth = int(mlp_depth)
        self._make_layers()

    def _feat_per_t(self) -> int:
        if self.input_ndims == 2:        # (C, T)
            return self.num_channels
        elif self.input_ndims == 4:      # (C, H, W, T)
            return self.num_channels * self.input_shape[1] * self.input_shape[2]
        return 0

    def _make_layers(self) -> None:
        p = self.d_model
        nheads = max([n for n in range(1, 10) if (not p % n) and (not n % 2)])

        self.core = _MAEEncoder(self._feat_per_t(), self.d_model, self.sequence_length,
                                num_layers=self.num_layers, num_heads=nheads)

        self.feature_size = self.d_model * self.sequence_length
        flatten_size = self.feature_size + self.stats_size

        layers = []
        current = flatten_size
        if self.mlp_depth >= 2:
            layers += [nn.Linear(current, 256), nn.ReLU(), nn.Dropout(0.1)]
            current = 256
        if self.mlp_depth >= 3:
            layers += [nn.Linear(current, 256), nn.ReLU(), nn.Dropout(0.1)]
            current = 256
        layers.append(nn.Linear(current, self.output_size))
        self.readout = nn.Sequential(*layers)

    def _tokens(self, input_sequences: dict[str, torch.Tensor]) -> torch.Tensor:
        s = self._process_sequence(input_sequences)         # (B, C, [H, W,] T)
        if self.input_ndims == 4:
            s = s.flatten(start_dim=1, end_dim=-2)            # (B, C*H*W, T)
        return s.transpose(-2, -1)                            # (B, T, feat_per_t)

    def forward_core(self,
                     input_sequences: dict[str, torch.Tensor],
                     input_stats: torch.Tensor) -> torch.Tensor:
        return self.core(self._tokens(input_sequences)).flatten(start_dim=1)

    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        if input_stats.numel():
            features = torch.hstack([features, input_stats.flatten(start_dim=1)])
        return self.readout(features)
