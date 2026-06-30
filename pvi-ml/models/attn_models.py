import torch
import torch.nn as nn

from src.models.base_model import BasePviLearner
from src.models.positional_encoder import (
    ResidualRecurrentPositionalEncoder as RRPE,
    SinusoidalPositionalEncoder,
    LearnablePositionalEncoder,
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

    def _make_layers(self) -> None:
        self.conv_layers = nn.ModuleList()

        if self.input_ndims == 2:
            for i in range(0, self.cnn_depth):
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv1d(
                            self.num_channels,
                            self.num_channels,
                            kernel_size=5,
                            padding=2,
                        ),
                        nn.BatchNorm1d(self.num_channels),
                        nn.ReLU(),
                    )
                )

            channel_size = self.num_channels

        elif self.input_ndims == 4:
            for i in range(0, self.cnn_depth):
                self.conv_layers.append(
                    nn.Sequential(
                        nn.Conv3d(
                            self.num_channels,
                            self.num_channels,
                            kernel_size=(3, 3, 5),
                            padding=(1, 1, 2),
                        ),
                        nn.BatchNorm3d(self.num_channels),
                        nn.ReLU(),
                        nn.MaxPool3d(
                            kernel_size=(2, 2, 1)
                        ),  # Time dimension is not pooled
                    )
                )

            final_img_dim = 40 // (2**self.cnn_depth)
            if final_img_dim < 1:
                final_img_dim = 1
            channel_size = self.num_channels * final_img_dim * final_img_dim

        else:
            # error case, should be already handled in the base class
            channel_size = None

        self.projection = nn.Linear(channel_size, self.projection_dim)

        if self.pe_type == "rrpe":
            self.rrpe = RRPE(
                input_size=self.projection_dim,
                hidden_size=self.rrpe_dim,
                recurrent_type="LSTM",
            )
        elif self.pe_type == "sinusoidal":
            self.rrpe = SinusoidalPositionalEncoder(
                d_model=self.projection_dim, max_len=5000
            )
        elif self.pe_type == "learnable":
            self.rrpe = LearnablePositionalEncoder(
                d_model=self.projection_dim, max_len=5000
            )
        elif self.pe_type == "none":
            self.rrpe = nn.Identity()
        else:
            raise ValueError(f"Unknown pe_type: {self.pe_type}")

        # num_heads must be divisible by d_model and should be even
        # so we find the maximum even divisor of the projection dimension
        p = self.projection_dim
        nheads = max([n for n in range(1, 10) if (not p % n) and (not n % 2)])

        self.transformer = Transformer(
            d_model=self.projection_dim, num_layers=2, num_heads=nheads, dim_mlp=512
        )

        flatten_size = (self.projection_dim * self.sequence_length) + self.stats_size

        layers = []
        current_dim = flatten_size

        # If mlp_depth > 1, add hidden layers
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

        # Output layer
        layers.append(nn.Linear(current_dim, self.output_size))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self, input_sequences: dict[str, torch.Tensor], input_stats: torch.Tensor
    ) -> torch.Tensor:
        s = self._process_sequence(input_sequences)  # shape: (B, C, H, W, T)

        for conv in self.conv_layers:
            s = conv(s)  # shape: (B, C', H', W', T)

        if self.input_ndims == 4:
            s = s.flatten(start_dim=1, end_dim=-2)  # shape: (B, C'*H'*W', T)

        s = s.transpose(-2, -1)  # Shape: (B, T, C'*H'*W')

        s = self.projection(s)  # Shape: (B, T, P)
        s = self.rrpe(s)  # Shape: (B, T, P)
        s = self.transformer(s)  # Shape: (B, T, P)

        s = s.transpose(-2, -1)  # Shape: (B, P, T)
        s = s.flatten(start_dim=1)  # Shape: (B, P*T)

        if input_stats.numel():  # if not empty
            f = input_stats.flatten(start_dim=1)
            s = torch.hstack([s, f])

        y = self.mlp(s)

        return y


if __name__ == "__main__":
    pass
