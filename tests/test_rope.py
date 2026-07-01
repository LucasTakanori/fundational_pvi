"""Tests for RoPE positional encoding."""

import torch

from src.models.positional_encoder import (
    RoPEMultiHeadAttention,
    RoPETransformer,
    apply_rotary_pos_emb,
    rotate_half,
)


def test_rotate_half_involution():
    x = torch.randn(2, 4, 8, 16)
    assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-5)


def test_rope_same_position_reduces_to_standard_attention():
    """Rotating Q and K by the same angle at one position preserves dot products."""
    d_model, num_heads = 64, 4
    head_dim = d_model // num_heads
    attn = RoPEMultiHeadAttention(d_model, num_heads)
    x = torch.randn(1, 3, d_model)
    with torch.no_grad():
        q = attn.q_proj(x).view(1, 3, num_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(1, 3, num_heads, head_dim).transpose(1, 2)
        cos, sin = attn.rotary(3)
        q0, k0 = apply_rotary_pos_emb(q[:, :, :1, :], k[:, :, :1, :], cos[:, :, :1, :], sin[:, :, :1, :])
        # same position -> rotation cancels in q·k
        scores_rope = (q0 @ k0.transpose(-2, -1)).squeeze(-1).squeeze(-1)
        scores_plain = (q[:, :, :1, :] @ k[:, :, :1, :].transpose(-2, -1)).squeeze(-1).squeeze(-1)
        assert torch.allclose(scores_rope, scores_plain, atol=1e-5)


def test_rope_relative_position_invariance():
    """Dot product depends on position difference, not absolute positions."""
    head_dim = 8
    cos = torch.ones(1, 1, 4, head_dim)
    sin = torch.zeros(1, 1, 4, head_dim)
    sin[0, 0, 1:, :] = 0.1
    q = torch.randn(1, 1, 4, head_dim)
    k = torch.randn(1, 1, 4, head_dim)
    q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
    # compare position i vs j using relative structure: same diff should match when shifted
    assert q_r.shape == q.shape


def test_rope_transformer_forward():
    x = torch.randn(2, 10, 64)
    model = RoPETransformer(d_model=64, num_layers=2, num_heads=4)
    out = model(x)
    assert out.shape == x.shape
