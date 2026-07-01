import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualRecurrentPositionalEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        recurrent_type: str = "LSTM",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if recurrent_type.lower() == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True,
            )

        elif recurrent_type.lower() == "gru":
            self.encoder = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True,
            )

        else:  # vanilla rnn
            self.encoder = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True,
            )

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        # map to input_size so we can use residual
        self.fc2 = nn.Linear(4 * hidden_size, input_size)

    def forward(self, x):
        res = torch.clone(x)

        x = self.encoder(x)[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = res + x

        return x


class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class LearnablePositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query/key tensors shaped (B, H, T, D)."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Precompute cos/sin tables for RoPE (head dimension must be even)."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head dim must be even, got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = int(max_seq_len)

    def forward(self, seq_len: int, device=None, dtype=None) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or self.inv_freq.device
        dtype = dtype or torch.float32
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device=device, dtype=dtype))
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings on Q/K."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must divide num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(seq_len, device=x.device, dtype=x.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(attn)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_mlp: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_mlp)
        self.linear2 = nn.Linear(dim_mlp, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        y = self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x)))))
        return x + self.dropout(y)


class RoPETransformer(nn.Module):
    """Stack of RoPE transformer encoder layers (drop-in for ``Transformer`` when pe_type=rope)."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_mlp: int = 512,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(d_model, num_heads, dim_mlp=dim_mlp)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
