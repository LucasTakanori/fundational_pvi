"""Paper-faithful core (candidate 4): 3D-conv DenseNet + Conv-LSTM.

Replicates the module stack of Wang et al. (Nature 2025) minus the perspective
module, adapted to the EIT image input:

  * DenseNet of *causal* (time-shifted) 3D spatiotemporal convolutions
    (dense connections, GELU, spatial pooling) -> spatiotemporal features;
  * Conv-LSTM recurrent cell (2D spatial-conv gates) over time ->
    a per-time core feature map H_t in R^{C x H' x W'};
  * a modulation encoder over per-cycle stats (duration/tMax), concatenated into
    the Conv-LSTM input (the behavioural-modulation analog);
  * a spatial bilinear readout: sample the core map at K *learned* spatial
    positions (the per-neuron spatial-readout analog; learned positions are
    interpretable - "where on the ring BP is read from") -> BP.

Fits the BasePviLearner core/readout contract (forward_core/forward_readout), so
it pretrains/transfers like the other cores. Image input only (input_ndims == 4).

Deviations (see PLAN.md): stats here are per-sample (not a behavioural time
series), so the modulation is an MLP rather than an LSTM; MSE/MorphologyLoss is
used downstream rather than Poisson NLL (continuous BP).
"""

from src.packages import *  # noqa: F401,F403  (torch, nn, ...)
from torch.nn import functional as F

from src.models.base_model import BasePviLearner


class CausalConv3d(nn.Module):
    """3D conv over (H, W, T) with causal (left-only) padding on the time axis."""

    def __init__(self, in_ch: int, out_ch: int, k_t: int = 3, k_s: int = 3):
        super().__init__()
        self.pad_t = k_t - 1
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(k_s, k_s, k_t),
                              padding=(k_s // 2, k_s // 2, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W, T); pad only the last (time) dim on the left
        x = F.pad(x, (self.pad_t, 0))
        return self.conv(x)


class DenseConv3DBlock(nn.Module):
    """Dense block of causal 3D convs, then spatial (not temporal) pooling."""

    def __init__(self, in_ch: int, growth: int = 8, n_layers: int = 3, k_t: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(CausalConv3d(ch, growth, k_t=k_t), nn.GELU()))
            ch += growth
        self.out_ch = ch
        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 1))  # halve H, W; keep T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return self.pool(x)


class ConvLSTMCell(nn.Module):
    """Conv-LSTM cell with 2D spatial-conv gates."""

    def __init__(self, in_ch: int, hidden_ch: int, k: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size=k, padding=k // 2)

    def forward(self, x: torch.Tensor, state):
        h, c = state
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ModulationEncoder(nn.Module):
    """Per-cycle stats -> spatial modulation map broadcast over (H, W)."""

    def __init__(self, stats_size: int, out_ch: int):
        super().__init__()
        self.out_ch = out_ch
        self.in_size = max(int(stats_size), 1)
        self.net = nn.Sequential(nn.Linear(self.in_size, out_ch), nn.GELU())

    def forward(self, stats: torch.Tensor, batch: int, H: int, W: int,
                device, dtype) -> torch.Tensor:
        if stats is None or stats.numel() == 0:
            v = torch.zeros(batch, self.out_ch, device=device, dtype=dtype)
        else:
            v = self.net(stats.flatten(start_dim=1))
        return v[:, :, None, None].expand(-1, -1, H, W)


class SpatialBilinearReadout(nn.Module):
    """Sample the core map at K learned spatial positions -> BP output."""

    def __init__(self, num_positions: int, channels: int, seq_len: int,
                 output_size: int, hidden: int = 256):
        super().__init__()
        self.K = num_positions
        self.positions = nn.Parameter(torch.zeros(num_positions, 2))  # center init; tanh-bounded
        self.head = nn.Sequential(
            nn.Linear(seq_len * channels * num_positions, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_size),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        # fmap: (B, C, H, W, T)
        B, C, H, W, T = fmap.shape
        x = fmap.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
        grid = torch.tanh(self.positions).view(1, 1, self.K, 2).expand(B * T, 1, self.K, 2)
        sampled = F.grid_sample(x, grid, align_corners=True)          # (B*T, C, 1, K)
        sampled = sampled.view(B, T, C, self.K).reshape(B, T * C * self.K)
        return self.head(sampled)


class PviDenseNetConvLSTM(BasePviLearner):
    def __init__(self,
                 data_shapes: dict[str, tuple[int, ...]],
                 diff: int = 2,
                 use_stats: bool = True,
                 num_blocks: int = 3,
                 growth: int = 8,
                 layers_per_block: int = 3,
                 hidden_ch: int = 16,
                 mod_ch: int = 8,
                 num_positions: int = 16,
                 readout_hidden: int = 256,
                 verbose: bool = True,
                 ) -> None:
        super().__init__(data_shapes=data_shapes, diff=diff,
                         use_stats=use_stats, verbose=verbose)

        if self.input_ndims != 4:
            raise ValueError("PviDenseNetConvLSTM expects image input (input_ndims == 4).")

        self.num_blocks = int(num_blocks)
        self.growth = int(growth)
        self.layers_per_block = int(layers_per_block)
        self.hidden_ch = int(hidden_ch)
        self.mod_ch = int(mod_ch)
        self.num_positions = int(num_positions)
        self.readout_hidden = int(readout_hidden)

        self._make_layers()

    def _make_layers(self) -> None:
        H0, W0 = self.input_shape[1], self.input_shape[2]

        blocks = []
        ch = self.num_channels
        for _ in range(self.num_blocks):
            blk = DenseConv3DBlock(ch, growth=self.growth, n_layers=self.layers_per_block)
            blocks.append(blk)
            ch = blk.out_ch
        dense = nn.Sequential(*blocks)

        Hf = max(H0 // (2 ** self.num_blocks), 1)
        Wf = max(W0 // (2 ** self.num_blocks), 1)
        self._feat_hw = (Hf, Wf)

        convlstm = ConvLSTMCell(ch + self.mod_ch, self.hidden_ch)
        modulation = ModulationEncoder(self.stats_size, self.mod_ch)

        # Core = dense backbone + modulation + Conv-LSTM (shared, transferable).
        self.core = nn.ModuleDict({"dense": dense, "convlstm": convlstm, "modulation": modulation})
        self._dense_out_ch = ch

        self._core_shape = (self.hidden_ch, Hf, Wf, self.sequence_length)
        self.feature_size = self.hidden_ch * Hf * Wf * self.sequence_length

        self.readout = SpatialBilinearReadout(self.num_positions, self.hidden_ch,
                                              self.sequence_length, self.output_size,
                                              hidden=self.readout_hidden)

    def forward_core(self,
                     input_sequences: dict[str, torch.Tensor],
                     input_stats: torch.Tensor) -> torch.Tensor:
        s = self._process_sequence(input_sequences)        # (B, C, H, W, T)
        z = self.core["dense"](s)                            # (B, Cd, Hf, Wf, T)
        B, Cd, Hf, Wf, T = z.shape

        mod = self.core["modulation"](input_stats, B, Hf, Wf, device=z.device, dtype=z.dtype)

        cell = self.core["convlstm"]
        h = z.new_zeros(B, cell.hidden_ch, Hf, Wf)
        c = z.new_zeros(B, cell.hidden_ch, Hf, Wf)
        outs = []
        for t in range(T):
            x_t = torch.cat([z[..., t], mod], dim=1)         # (B, Cd + mod_ch, Hf, Wf)
            h, c = cell(x_t, (h, c))
            outs.append(h)
        core_map = torch.stack(outs, dim=-1)                 # (B, hidden_ch, Hf, Wf, T)
        return core_map.flatten(start_dim=1)

    def forward_readout(self,
                        features: torch.Tensor,
                        input_stats: torch.Tensor) -> torch.Tensor:
        fmap = features.view(features.shape[0], *self._core_shape)
        return self.readout(fmap)
