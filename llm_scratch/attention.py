from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import RotaryEmbedding, apply_rotary


@dataclass
class AttentionConfig:
    """Configuration for grouped multi-head attention.

    - ``n_heads``: total query heads.
    - ``n_kv_heads``: number of distinct key/value heads.
      - MHA: n_kv_heads == n_heads
      - GQA: 1 < n_kv_heads < n_heads (shared k/v across groups of q heads)
      - MQA: n_kv_heads == 1 (single shared k/v across all q heads)
    - ``head_dim``: per-head hidden dimension; if None uses d_model // n_heads.
    - ``rope``: whether to apply Rotary Positional Embeddings to q/k.
    """
    d_model: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    bias: bool = False
    rope: bool = False
    rope_base: float = 10000.0


class GroupedMultiHeadAttention(nn.Module):
    """Scaled dot-product attention supporting MHA, GQA, and MQA in one module.

    Theory:
    - Attention computes softmax(Q K^T / sqrt(d)) V, optionally with additive
      bias for masking (causal/padding/prefix) and positional bias (via RoPE).
    - Grouped-Query Attention (GQA) reduces memory/computation by sharing K/V
      heads across groups of Q heads. Multi-Query Attention (MQA) is the extreme
      case with a single K/V head.
    - This module projects Q with ``n_heads`` and K/V with ``n_kv_heads`` then
      repeats K/V across groups to match Q heads.

    Practical notes:
    - ``rope=True`` applies RoPE to Q and K before attention.
    - ``attn_bias`` expects shape [B, 1, Tq, Tk] or [1, 1, Tq, Tk] and is added
      before softmax to enforce masks.
    """
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads or cfg.n_heads
        self.head_dim = cfg.head_dim or (cfg.d_model // cfg.n_heads)
        assert self.d_model == self.n_heads * self.head_dim
        assert self.n_heads % self.n_kv == 0
        self.group_size = self.n_heads // self.n_kv
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv * self.head_dim, bias=cfg.bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=cfg.bias)
        self.scale = self.head_dim ** -0.5
        self.use_rope = cfg.rope
        self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base) if self.use_rope else None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat K/V heads to match Q heads for GQA/MQA.

        Shapes and semantics:
        - Input ``x``: [B, T, Kv, Dh] where ``Kv = n_kv_heads`` and ``Dh = head_dim``.
        - We have ``H = n_heads`` query heads and ``Kv`` key/value heads with
          ``group_size = H // Kv``. For GQA/MQA, each K/V head is shared by a
          group of ``group_size`` Q heads.
        - Goal: produce K/V with ``H`` heads by logically tiling each K/V head
          across its Q-head group → [B, T, H, Dh].

        Example: H=8, Kv=2 → group_size=4. Input [B,T,2,Dh] → output [B,T,8,Dh].
        """
        b, t, kv, d = x.shape
        # MHA: H == Kv -> group_size == 1, no replication needed.
        if self.group_size == 1:
            return x
        # Insert a new axis for the group and broadcast over it.
        # Before: x shape [B, T, Kv, Dh]
        # Add group axis (size=group_size): [B, T, Kv, 1, Dh]
        # expand(...) creates a broadcasted view (zero-copy) to [B, T, Kv, G, Dh]
        # where G = group_size. No memory is duplicated here.
        x = x[:, :, :, None, :].expand(b, t, kv, self.group_size, d)
        # Merge Kv and G -> H = Kv * G = n_heads, yielding [B, T, H, Dh].
        # This is a reshape of the broadcasted view, still logically repeating K/V.
        return x.reshape(b, t, kv * self.group_size, d)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple] = None,
    ) -> tuple:
        q_inp = x
        kv_inp = x if context is None else context
        bsz, tq, _ = q_inp.shape
        tk = kv_inp.size(1)

        q = self.q_proj(q_inp).view(bsz, tq, self.n_heads, self.head_dim)
        k_new = self.k_proj(kv_inp).view(bsz, tk, self.n_kv, self.head_dim)
        v_new = self.v_proj(kv_inp).view(bsz, tk, self.n_kv, self.head_dim)

        # Always use KV cache semantics
        if context is None:
            # Self-attention: append new keys/values to past
            if self.use_rope:
                offset = past_kv[0].size(1) if past_kv is not None else 0
                sin_q, cos_q = self.rope(q, tq, offset=offset)
                q = apply_rotary(q, sin_q, cos_q)
                sin_k, cos_k = self.rope(k_new, tk, offset=offset)
                k_new = apply_rotary(k_new, sin_k, cos_k)
            if past_kv is not None:
                past_k, past_v = past_kv
                k_all = torch.cat([past_k, k_new], dim=1)
                v_all = torch.cat([past_v, v_new], dim=1)
            else:
                k_all, v_all = k_new, v_new
            present_kv = (k_all, v_all)
        else:
            # Cross-attention: memory is static per segment; cache or reuse provided
            if self.use_rope:
                sin_q, cos_q = self.rope(q, tq, offset=0)
                q = apply_rotary(q, sin_q, cos_q)
                # For memory, use offset=0 (encoder positions)
                sin_k, cos_k = self.rope(k_new, tk, offset=0)
                k_new = apply_rotary(k_new, sin_k, cos_k)
            if past_kv is not None:
                k_all, v_all = past_kv
            else:
                k_all, v_all = k_new, v_new
            present_kv = (k_all, v_all)

        # Repeat K/V heads to match Q heads (for GQA/MQA)
        k = self._repeat_kv(k_all)
        v = self._repeat_kv(v_all)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_bias is not None:
            if attn_bias.dim() == 4:
                att = att + attn_bias
            else:
                att = att + attn_bias[None, None, :, :]
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(bsz, tq, self.n_heads * self.head_dim)
        y = self.out_proj(y)
        y = self.resid_drop(y)
        return y, present_kv
