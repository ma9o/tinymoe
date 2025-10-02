from typing import Optional, Literal

import torch
import torch.nn as nn

from .norms import make_norm
from .positional import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding
from .masking import (
    make_causal_mask,
    make_padding_bias,
    make_prefix_lm_bias,
    combine_biases,
)
from .blocks import DecoderBlock


class TinyLM(nn.Module):
    """A compact, modular Transformer decoder for experimentation.

    Theory overview:
    - Combines key LLM components: MHA/GQA/MQA attention, (RMS/Layer)Norm,
      FFN or MoE feed-forwards, and several masking strategies.
    - Supports absolute (learned/sinusoidal) and rotary positional encodings.
    - Implements a decoder-only stack with optional cross-attention to an
      external memory, enabling encoder–decoder setups when needed.

    Design notes:
    - Pre-norm residual blocks for stable training.
    - Attention masks are additive biases added before softmax.
    - ``n_kv_heads`` controls K/V sharing pattern (MHA/GQA/MQA).
    - Intended for clarity and teaching; not an optimized training stack.
    """
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        pos_embedding: Literal["learned", "sinusoidal", "none"] = "learned",
        rope_in_attn: bool = False,
        norm: Literal["layernorm", "rmsnorm"] = "rmsnorm",
        ffn_hidden_mult: float = 4.0,
        use_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 1,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.vocab = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos_kind = pos_embedding
        if pos_embedding == "learned":
            self.pos = LearnedPositionalEmbedding(max_len, d_model)
        elif pos_embedding == "sinusoidal":
            self.pos = SinusoidalPositionalEmbedding(max_len, d_model)
        else:
            self.pos = None
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    attn_dropout=attn_dropout,
                    resid_dropout=resid_dropout,
                    norm=norm,
                    ffn_hidden_mult=ffn_hidden_mult,
                    ffn_act="silu",
                    ffn_gated=True,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    moe_top_k=moe_top_k,
                    rope=rope_in_attn,
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = make_norm(norm, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = idx.shape
        h = self.tok(idx)
        if self.pos is not None:
            pos_ids = torch.arange(seqlen, device=idx.device).unsqueeze(0).expand(bsz, seqlen)
            h = h + self.pos(pos_ids)
        return h

    def forward(
        self,
        idx: torch.Tensor,
        attn_strategy: Literal["causal", "none", "padding", "causal_padding", "prefix"] = "causal",
        key_padding: Optional[torch.Tensor] = None,
        prefix_lens: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
        memory_padding: Optional[torch.Tensor] = None,
        past_kv_self: Optional[list] = None,
        past_kv_cross: Optional[list] = None,
    ) -> tuple:
        bsz, seqlen = idx.shape
        h = self.embed(idx)

        if attn_strategy == "none":
            self_bias = None
        elif attn_strategy == "causal":
            past_len = past_kv_self[0][0].size(1) if (past_kv_self is not None and len(past_kv_self) > 0 and past_kv_self[0] is not None) else 0
            base_curr = make_causal_mask(seqlen, seqlen, h.device, h.dtype)
            if past_len > 0:
                left = torch.zeros(seqlen, past_len, device=h.device, dtype=h.dtype)
                base = torch.cat([left, base_curr], dim=1)
            else:
                base = base_curr
            self_bias = base[None, None, :, :]
        elif attn_strategy == "padding":
            past_len = past_kv_self[0][0].size(1) if (past_kv_self is not None and len(past_kv_self) > 0 and past_kv_self[0] is not None) else 0
            curr = make_padding_bias(key_padding, seqlen, h.dtype)
            if curr is None:
                self_bias = torch.zeros(h.size(0), 1, seqlen, past_len, dtype=h.dtype, device=h.device) if past_len > 0 else None
            else:
                if past_len > 0:
                    left = torch.zeros(curr.size(0), 1, seqlen, past_len, dtype=h.dtype, device=h.device)
                    self_bias = torch.cat([left, curr], dim=-1)
                else:
                    self_bias = curr
        elif attn_strategy == "causal_padding":
            past_len = past_kv_self[0][0].size(1) if (past_kv_self is not None and len(past_kv_self) > 0 and past_kv_self[0] is not None) else 0
            base_curr = make_causal_mask(seqlen, seqlen, h.device, h.dtype)
            if past_len > 0:
                left = torch.zeros(seqlen, past_len, device=h.device, dtype=h.dtype)
                base = torch.cat([left, base_curr], dim=1)[None, None, :, :]
            else:
                base = base_curr[None, None, :, :]
            pad_curr = make_padding_bias(key_padding, seqlen, h.dtype)
            if pad_curr is None:
                pad = torch.zeros(h.size(0), 1, seqlen, past_len, dtype=h.dtype, device=h.device) if past_len > 0 else None
            else:
                if past_len > 0:
                    left = torch.zeros(pad_curr.size(0), 1, seqlen, past_len, dtype=h.dtype, device=h.device)
                    pad = torch.cat([left, pad_curr], dim=-1)
                else:
                    pad = pad_curr
            self_bias = combine_biases(base, pad)
        elif attn_strategy == "prefix":
            assert prefix_lens is not None
            # For simplicity, not extending prefix mask with cache (typical usage
            # is full-context training). If needed, this can be extended similarly
            # to the causal case by left-padding with zeros for past_len.
            self_bias = make_prefix_lm_bias(prefix_lens, seqlen, seqlen, h.dtype)
        else:
            raise ValueError(attn_strategy)

        cross_bias = None
        if memory is not None:
            _, memlen, _ = memory.shape
            cross_bias = make_padding_bias(memory_padding, seqlen, h.dtype)

        present_self_list = []
        present_cross_list = []
        for li, blk in enumerate(self.blocks):
            p_self = past_kv_self[li] if (past_kv_self is not None and li < len(past_kv_self)) else None
            p_cross = past_kv_cross[li] if (past_kv_cross is not None and li < len(past_kv_cross)) else None
            h, curr_self, curr_cross = blk(h, self_bias, memory, cross_bias, past_kv_self=p_self, past_kv_cross=p_cross)
            present_self_list.append(curr_self)
            present_cross_list.append(curr_cross)
        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits, present_self_list, present_cross_list
