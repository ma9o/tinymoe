from typing import Optional, Literal

import torch
import torch.nn as nn

from .attention import AttentionConfig, GroupedMultiHeadAttention
from .norms import make_norm
from .feedforward import FeedForward, MoE


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional cross-attention.

    Theory:
    - Pre-norm: Normalization is applied before sublayers, improving gradient
      flow and training stability for deep decoders.
    - Structure: x + SelfAttn(Norm(x)); optional x + CrossAttn(Norm(x)); then
      x + FFN(Norm(x)). Residual connections preserve identity paths.
    - Cross-attention allows conditioning the decoder on an external memory
      (e.g., an encoder output), enabling encoder–decoder architectures.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        norm: Literal["layernorm", "rmsnorm"] = "rmsnorm",
        ffn_hidden_mult: float = 4.0,
        ffn_act: Literal["gelu", "relu", "silu"] = "silu",
        ffn_gated: bool = True,
        use_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 1,
        rope: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.attn = GroupedMultiHeadAttention(
            AttentionConfig(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                bias=bias,
                rope=rope,
            )
        )
        self.norm1 = make_norm(norm, d_model)
        self.norm2 = make_norm(norm, d_model)
        hidden = int(ffn_hidden_mult * d_model)
        if use_moe:
            self.ff = MoE(d_model, hidden, num_experts=num_experts, top_k=moe_top_k, activation=ffn_act, dropout=resid_dropout)
        else:
            self.ff = FeedForward(d_model, hidden, dropout=resid_dropout, activation=ffn_act, gated=ffn_gated)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        cross_bias: Optional[torch.Tensor] = None,
        past_kv_self: Optional[tuple] = None,
        past_kv_cross: Optional[tuple] = None,
    ) -> tuple:
        # Self-attention (always KV cached)
        y, present_self = self.attn(self.norm1(x), attn_bias=attn_bias, past_kv=past_kv_self)
        h = x + y

        # Cross-attention (optional, always KV cached)
        if context is not None:
            y2, present_cross = self.attn(self.norm1(h), attn_bias=cross_bias, context=context, past_kv=past_kv_cross)
            h = h + y2
        else:
            present_cross = None

        h = h + self.ff(self.norm2(h))
        return h, present_self, present_cross


class DecoderBlock(nn.Module):
    """Thin wrapper exposing a decoder-style forward API.

    - ``self_bias``: mask for self-attention (causal/padding/prefix).
    - ``memory`` and ``cross_bias``: enable cross-attention when provided.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.block = TransformerBlock(*args, **kwargs)

    def forward(
        self,
        x,
        self_bias=None,
        memory=None,
        cross_bias=None,
        past_kv_self=None,
        past_kv_cross=None,
    ):
        return self.block(
            x,
            attn_bias=self_bias,
            context=memory,
            cross_bias=cross_bias,
            past_kv_self=past_kv_self,
            past_kv_cross=past_kv_cross,
        )
