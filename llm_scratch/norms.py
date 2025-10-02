import torch
import torch.nn as nn
from typing import Literal


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    Theory:
    - Normalizes activations by their root-mean-square over the last dimension
      without subtracting the mean, which reduces dependency on batch statistics
      and avoids mean-centering.
    - For a vector x in R^d, RMSNorm computes y = x * g / sqrt(mean(x^2) + eps)
      (+ optional bias), where g is a learned gain.
    - Used in modern LLMs (e.g., LLaMA variants) for training stability and
      inference efficiency, often outperforming LayerNorm in decoder-only stacks.

    Notes:
    - Unlike LayerNorm, there is no mean subtraction; only scale normalization.
    - Applied in "pre-norm" blocks before residual branches to improve gradients.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x = x * rms
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


def make_norm(kind: Literal["layernorm", "rmsnorm"], d_model: int, eps: float = 1e-5) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    if kind == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    raise ValueError(f"Unknown norm kind: {kind}")
