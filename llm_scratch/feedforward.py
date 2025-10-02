from typing import Literal

import torch
import torch.nn.functional as F
import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise MLP with optional gating (SwiGLU-style).

    Theory:
    - Standard Transformer FFN expands hidden size (e.g., 3-4x), applies a
      nonlinearity, then projects back to model dimension. This increases the
      network's capacity for token-wise transformations.
    - Gated variants (e.g., SwiGLU) compute act(W1 x) ⊙ (Wg x), improving
      expressivity and often convergence in LLMs.
    """
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0, activation: Literal["gelu", "relu", "silu"] = "gelu", gated: bool = False):
        super().__init__()
        self.gated = gated
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc_gate = nn.Linear(d_model, hidden) if gated else None
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
        if activation == "gelu":
            self.act = F.gelu
        elif activation == "relu":
            self.act = F.relu
        elif activation == "silu":
            self.act = F.silu
        else:
            raise ValueError(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            h = self.act(self.fc1(x)) * self.fc_gate(x)
        else:
            h = self.act(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return h


class Expert(nn.Module):
    """A single expert: gated FFN used inside MoE.

    Notes:
    - Uses a SwiGLU-style FFN for richer token-wise transformations.
    - Each expert processes a subset of tokens selected by the router.
    """
    def __init__(self, d_model: int, hidden: int, activation: str = "silu", dropout: float = 0.0):
        super().__init__()
        self.ffn = FeedForward(d_model, hidden, dropout=dropout, activation=activation, gated=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MoE(nn.Module):
    """Mixture-of-Experts with token-level Top-k routing.

    Theory:
    - A router scores each token for each expert; Top-k experts are selected and
      their outputs combined (weighted by softmax probabilities).
    - Increases parameter count without increasing per-token compute proportionally
      (only k experts are active per token).

    Implementation notes:
    - No explicit capacity or load balancing loss is implemented here (kept
      minimal for clarity). In production, auxiliary balancing losses and expert
      capacity constraints are often used to prevent collapse and OOM.
    """
    def __init__(self, d_model: int, hidden: int, num_experts: int = 4, top_k: int = 1, dropout: float = 0.0, activation: str = "silu"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(d_model, hidden, activation=activation, dropout=dropout) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        flat = x.reshape(bsz * seq, dim)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        if self.top_k == 1:
            top_val, top_idx = probs.max(dim=-1)
            y = torch.zeros_like(flat)
            for e in range(self.num_experts):
                mask = (top_idx == e)
                if mask.any():
                    xs = flat[mask]
                    out = self.experts[e](xs)
                    y[mask] = out * top_val[mask].unsqueeze(-1)
        else:
            top_val, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
            y = torch.zeros_like(flat)
            for k in range(self.top_k):
                idx = top_idx[:, k]
                w = top_val[:, k]
                for e in range(self.num_experts):
                    mask = (idx == e)
                    if mask.any():
                        xs = flat[mask]
                        out = self.experts[e](xs)
                        y[mask] += out * w[mask].unsqueeze(-1)
        return y.view(bsz, seq, dim)
