import math
from typing import Tuple

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embeddings.

    Theory:
    - Associates each position index with a trainable vector added to token
      embeddings. Simple and expressive, used by early Transformers (e.g., GPT-2).
    - Absolute encodings do not encode relative distance invariances and can be
      less robust to extrapolation beyond the maximum training length.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal (absolute) positional embeddings (Vaswani et al., 2017).

    Theory:
    - Uses sin/cos waves of different frequencies so that relative positions can
      be derived from linear combinations. Deterministic and length-extrapolable.
    - Each dimension corresponds to a sinusoid with frequency geometric progression.
    """
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) generator for attention heads.

    Theory:
    - Applies a position-dependent rotation to query/key vectors in 2D subspaces
      so that the inner product depends on relative offsets (i - j), enabling
      relative position awareness while keeping computation simple.
    - Equivalent to complex multiplication by a phase term; implemented by
      pairing halves of head dimension with sin/cos. Generalizes better to long
      sequences and is widely used in modern LLMs.
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        theta = 1.0 / (self.base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
        pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
        angles = pos * theta.unsqueeze(0)
        return torch.sin(angles), torch.cos(angles)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return per-position sin/cos for RoPE with an optional position offset.

        Args:
        - x: a tensor used only to infer device/dtype
        - seq_len: number of positions to return (length of the segment)
        - offset: starting absolute position index for the segment
        """
        # Build a cache large enough to index from offset:offset+seq_len
        total = offset + seq_len
        sin_full, cos_full = self._build_cache(total, x.device, x.dtype)
        # Slice the requested window
        if offset == 0:
            return sin_full[:seq_len], cos_full[:seq_len]
        return sin_full[offset:offset+seq_len], cos_full[offset:offset+seq_len]


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to packed heads tensor x of shape [B, T, H, Dh].

    Splits the head dimension into two halves and performs a 2D rotation using
    per-position sin/cos cached by :class:`RotaryEmbedding`. This preserves the
    L2 norm and injects position information directly in the query/key space.
    """
    b, t, h, d = x.shape
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    sin = sin[:t, :half].unsqueeze(0).unsqueeze(2)
    cos = cos[:t, :half].unsqueeze(0).unsqueeze(2)
    x1p = x1 * cos - x2 * sin
    x2p = x2 * cos + x1 * sin
    return torch.cat([x1p, x2p], dim=-1)
