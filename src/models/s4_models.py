from src.packages import *
from src.models.base_model import BasePviLearner
from src.models.positional_encoder import (
    ResidualRecurrentPositionalEncoder as RRPE,
    SinusoidalPositionalEncoder,
    LearnablePositionalEncoder,
)

try:
    from mambapy.mamba import Mamba, MambaConfig
except ImportError as _e:  # mambapy is optional (WIP S4/Mamba models)
    Mamba = MambaConfig = None
    _MAMBA_IMPORT_ERROR = _e

from torch.nn import functional as F

class MLP(nn.Module):

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.SiLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

class SlidingWindowAttention(nn.Module):
    def __init__(self,
                 n_embd: int=100,
                 head_dim: int=None,
                 block_size: int=250,
                 window_size: int=64):
        super().__init__()

        n_embd = n_embd
        head_dim = n_embd if head_dim is None else head_dim
        block_size = block_size
        window_size = window_size

        self.key = nn.Linear(n_embd, head_dim, bias=False)
        self.query = nn.Linear(n_embd, head_dim, bias=False)
        self.value = nn.Linear(n_embd, head_dim, bias=False)

        mask = torch.full((block_size, block_size), 0)  # creating the mask

        for col_idx in range(mask.shape[1]):
            col = mask[:, col_idx]
            col[col_idx:col_idx + window_size] = 1

        self.register_buffer('mask', mask)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _, _, P = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.transpose(-2, -1)

        weights = (q @ k) * (P ** -0.5)
        weights = weights.masked_fill(self.mask == 0, float('-inf'))  # apply the mask
        weights = F.softmax(weights, dim=-1)

        weights = self.dropout(weights)

        attention = weights @ v

        return attention

class MultiHeadSWA(nn.Module):
    def __init__(self,
                 num_heads: int=4,
                 n_embd:  int=100,
                 head_dim: int = None,
                 block_size: int = 250,
                 window_size: int = 64):

        super().__init__()

        head_dim = n_embd if head_dim is None else head_dim

        self.heads = nn.ModuleList([SlidingWindowAttention(n_embd, head_dim, block_size, window_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_dim, n_embd)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class SambaBlock(nn.Module):
    def __init__(self,
                 d_model: int=100,
                 block_size: int=250,
                 window_size: int=64,
                 num_heads: int=4,
                 mamba_layers: int=2,
                 ) -> None:
        super().__init__()

        config = MambaConfig(d_model=d_model, n_layers=mamba_layers)
        self.mamba = Mamba(config)
        self.mlp1 = MLP(n_embd=d_model)
        self.swa = MultiHeadSWA(num_heads=num_heads,
                                n_embd=d_model,
                                head_dim=4,
                                block_size=block_size,
                                window_size=window_size)
        self.mlp2 = MLP(n_embd=d_model)

    def forward(self, x):
        x = self.mamba(x)
        x = self.mlp1(x)
        x = self.swa(x)
        x = self.mlp2(x)

        return x

class PviSamba(BasePviLearner):
    def __init__(
        self,
        data_shapes: dict[str, tuple[int, ...]],
        mamba_layers: int = 2,
        samba_layers: int = 2,
        projection_dim: int = 100,
        rrpe_dim: int = 64,
        cnn_depth: int = 1,
        mlp_depth: int = 1,
        pe_type: str = "rrpe",
    ) -> None:

        super().__init__(data_shapes=data_shapes, diff=2, use_stats=True)

        self.projection_dim = projection_dim
        self.rrpe_dim = rrpe_dim
        self.mamba_layers = mamba_layers
        self.samba_layers = samba_layers
        self.cnn_depth = cnn_depth
        self.mlp_depth = mlp_depth
        self.pe_type = pe_type

        self._make_layers()

        print(f"{self._alias}: Total number of trainable weights: {self.num_params:,}")

    def _build_pe(self) -> nn.Module:
        if self.pe_type == "rrpe":
            return RRPE(input_size=self.projection_dim, hidden_size=self.rrpe_dim, recurrent_type="LSTM")
        elif self.pe_type == "sinusoidal":
            return SinusoidalPositionalEncoder(d_model=self.projection_dim, max_len=5000)
        elif self.pe_type == "learnable":
            return LearnablePositionalEncoder(d_model=self.projection_dim, max_len=5000)
        elif self.pe_type == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unknown pe_type: {self.pe_type}")

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
            channel_size = None

        nheads = max([n for n in range(1, 10) if (not self.projection_dim % n) and (not n % 2)])

        samba = nn.Sequential(
            *[
                SambaBlock(
                    d_model=self.projection_dim,
                    block_size=self.sequence_length,
                    window_size=64,
                    num_heads=nheads,
                    mamba_layers=self.mamba_layers,
                )
                for _ in range(self.samba_layers)
            ]
        )

        # Core = conv body + projection + positional encoder + Samba stack.
        self.core = nn.ModuleDict({
            "conv_layers": conv_layers,
            "projection": nn.Linear(channel_size, self.projection_dim),
            "pe": self._build_pe(),
            "sequence_model": samba,
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