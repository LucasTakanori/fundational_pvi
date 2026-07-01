import torch
import torch.nn as nn

from src.models.base_model import BasePviLearner
from src.models.positional_encoder import (
    ResidualRecurrentPositionalEncoder as RRPE,
    SinusoidalPositionalEncoder,
    LearnablePositionalEncoder,
    RoPETransformer,
)


class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_mlp: int = 512,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_mlp = dim_mlp

        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_mlp, batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.layer, num_layers=num_layers)

    def forward(self, x):
        x = self.encoder(x)

        return x


class PviCNNTransformer(BasePviLearner):
    def __init__(
        self,
        data_shapes: dict[str, tuple[int, ...]],
        projection_dim: int = 100,
        rrpe_dim: int = 64,
        transformer_dim: int = 64,
        cnn_depth: int = 2,
        mlp_depth: int = 3,
        pe_type: str = "rrpe",
    ) -> None:
        super().__init__(data_shapes=data_shapes, diff=2, use_stats=True)

        self.projection_dim = projection_dim
        self.rrpe_dim = rrpe_dim
        self.transformer_dim = transformer_dim
        self.cnn_depth = cnn_depth
        self.mlp_depth = mlp_depth
        self.pe_type = pe_type

        self._make_layers()

    def _build_pe(self) -> nn.Module:
        if self.pe_type == "rrpe":
            return RRPE(input_size=self.projection_dim, hidden_size=self.rrpe_dim, recurrent_type="LSTM")
        elif self.pe_type == "sinusoidal":
            return SinusoidalPositionalEncoder(d_model=self.projection_dim, max_len=5000)
        elif self.pe_type == "learnable":
            return LearnablePositionalEncoder(d_model=self.projection_dim, max_len=5000)
        elif self.pe_type == "rope":
            return nn.Identity()
        elif self.pe_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown pe_type: {self.pe_type}")

    def _build_sequence_model(self, nheads: int) -> nn.Module:
        if self.pe_type == "rope":
            return RoPETransformer(
                d_model=self.projection_dim, num_layers=2, num_heads=nheads, dim_mlp=512,
            )
        return Transformer(d_model=self.projection_dim, num_layers=2, num_heads=nheads, dim_mlp=512)

    def _make_layers(self) -> None:
        conv_layers = nn.ModuleList()

        if self.input_ndims == 2:
            for i in range(0, self.cnn_depth):
                conv_layers.append(
                    nn.Sequential(
                        nn.Conv1d(self.num_channels, self.num_channels, kernel_size=5, padding=2),
                        nn.BatchNorm1d(self.num_channels),
                        nn.ReLU(),
                    )
                )

            channel_size = self.num_channels

        elif self.input_ndims == 4:
            for i in range(0, self.cnn_depth):
                conv_layers.append(
                    nn.Sequential(
                        nn.Conv3d(self.num_channels, self.num_channels, kernel_size=(3, 3, 5), padding=(1, 1, 2)),
                        nn.BatchNorm3d(self.num_channels),
                        nn.ReLU(),
                        nn.MaxPool3d(kernel_size=(2, 2, 1)),  # Time dimension is not pooled
                    )
                )

            final_img_dim = 40 // (2**self.cnn_depth)
            if final_img_dim < 1:
                final_img_dim = 1
            channel_size = self.num_channels * final_img_dim * final_img_dim

        else:
            # error case, should be already handled in the base class
            channel_size = None

        # num_heads must divide d_model, be even, and (for RoPE) yield an even head_dim.
        p = self.projection_dim
        nheads = max([
            n for n in range(1, 10)
            if (not p % n) and (not n % 2)
            and (self.pe_type != "rope" or ((p // n) % 2 == 0))
        ])

        # Core = conv body + projection + positional encoder + transformer.
        self.core = nn.ModuleDict({
            "conv_layers": conv_layers,
            "projection": nn.Linear(channel_size, self.projection_dim),
            "pe": self._build_pe(),
            "sequence_model": self._build_sequence_model(nheads),
        })

        self.feature_size = self.projection_dim * self.sequence_length
        flatten_size = self.feature_size + self.stats_size

        layers = []
        current_dim = flatten_size

        if self.mlp_depth >= 2:
            layers.append(nn.Linear(current_dim, 256))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = 256

        if self.mlp_depth >= 3:
            layers.append(nn.Linear(current_dim, 256))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            current_dim = 256

        layers.append(nn.Linear(current_dim, self.output_size))

        # Readout = the MLP head (stats injected pre-readout).
        self.readout = nn.Sequential(*layers)

    def forward_core(
        self, input_sequences: dict[str, torch.Tensor], input_stats: torch.Tensor
    ) -> torch.Tensor:
        s = self._process_sequence(input_sequences)  # shape: (B, C, H, W, T)

        for conv in self.core["conv_layers"]:
            s = conv(s)  # shape: (B, C', H', W', T)

        if self.input_ndims == 4:
            s = s.flatten(start_dim=1, end_dim=-2)  # shape: (B, C'*H'*W', T)

        s = s.transpose(-2, -1)  # Shape: (B, T, C'*H'*W')

        s = self.core["projection"](s)  # Shape: (B, T, P)
        s = self.core["pe"](s)  # Shape: (B, T, P)
        s = self.core["sequence_model"](s)  # Shape: (B, T, P)

        s = s.transpose(-2, -1)  # Shape: (B, P, T)
        s = s.flatten(start_dim=1)  # Shape: (B, P*T)
        return s

    def forward_readout(
        self, features: torch.Tensor, input_stats: torch.Tensor
    ) -> torch.Tensor:
        if input_stats.numel():  # if not empty
            features = torch.hstack([features, input_stats.flatten(start_dim=1)])
        return self.readout(features)


if __name__ == "__main__":
    pass
