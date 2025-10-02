"""Mask construction utilities (additive attention biases).

Concept:
- We build additive biases for attention scores. Allowed positions get 0.0 and
  disallowed positions get a large negative number ("-inf"), so softmax zeros
  them out. This composes cleanly by simple addition (combine multiple masks).

Shapes:
- Base causal mask: [Tq, Tk]
- Padding bias: [B, 1, Tq, Tk]
- Combined final bias passed to attention: [B or 1, 1, Tq, Tk]

Notation in ASCII diagrams below:
- Rows represent queries (q), columns represent keys (k)
- "1" = allowed (0.0 bias); "." = masked (-inf bias)
"""

from typing import Optional

import torch


def _neg_inf(dtype: torch.dtype) -> float:
    if dtype.is_floating_point:
        return -1e9 if dtype == torch.float32 else -1e4
    return -1e9


def make_causal_mask(q_len: int, k_len: int, device, dtype) -> torch.Tensor:
    """Lower-triangular causal mask as additive bias [Tq, Tk].

    Theory:
    - Enforces autoregressive constraint: token at position i can only attend to
      keys at positions j <= i.

    Example (Tq=Tk=5):
        k: 0 1 2 3 4
      q0: 1 . . . .
      q1: 1 1 . . .
      q2: 1 1 1 . .
      q3: 1 1 1 1 .
      q4: 1 1 1 1 1

    Returns a float tensor with 0.0 for allowed, -inf for masked.
    """
    mask = torch.ones(q_len, k_len, device=device, dtype=torch.bool).tril(diagonal=0)
    bias = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    bias.masked_fill_(~mask, _neg_inf(dtype))
    return bias


def make_padding_bias(key_padding: Optional[torch.Tensor], q_len: int, dtype) -> Optional[torch.Tensor]:
    """Per-batch key padding bias [B, 1, Tq, Tk].

    Args:
    - key_padding: [B, Tk] boolean, True where keys are PAD and must be masked.

    Effect:
    - Masks (sets to -inf) the same key columns for all queries in that batch.

    Example (Tk=6, PAD at cols 4,5):
        key_pad: [0, 0, 0, 0, 1, 1]
        k: 0 1 2 3 4 5
      q*: 1 1 1 1 . .

    Returns None if key_padding is None, else a broadcastable bias.
    """
    if key_padding is None:
        return None
    bsz, k_len = key_padding.shape
    bias = torch.zeros(bsz, 1, q_len, k_len, dtype=dtype, device=key_padding.device)
    bias = bias.masked_fill(key_padding[:, None, None, :], _neg_inf(dtype))
    return bias


def make_prefix_lm_bias(prefix_lens: torch.Tensor, q_len: int, k_len: int, dtype) -> torch.Tensor:
    """Prefix-LM mask [B, 1, Tq, Tk] with bidirectional prefix and causal suffix.

    Semantics for each batch b with prefix length p:
    - Prefix tokens (rows < p) can attend bidirectionally within prefix (cols < p)
      and cannot attend to suffix (cols >= p).
    - Suffix tokens (rows >= p) can attend to all prefix (cols < p) and are causal
      within the suffix (cols in [p, row]).

    Example (Tq=Tk=8, p=3):
        k: 0 1 2 3 4 5 6 7
      q0: 1 1 1 . . . . .
      q1: 1 1 1 . . . . .
      q2: 1 1 1 . . . . .
      q3: 1 1 1 1 . . . .
      q4: 1 1 1 1 1 . . .
      q5: 1 1 1 1 1 1 . .
      q6: 1 1 1 1 1 1 1 .
      q7: 1 1 1 1 1 1 1 1
    """
    bsz = prefix_lens.shape[0]
    bias = torch.zeros(bsz, 1, q_len, k_len, dtype=dtype, device=prefix_lens.device)
    arange_q = torch.arange(q_len, device=prefix_lens.device)
    arange_k = torch.arange(k_len, device=prefix_lens.device)
    causal = arange_q[:, None] >= arange_k[None, :]
    for b in range(bsz):
        p = int(prefix_lens[b].item())
        full = torch.ones(q_len, k_len, dtype=torch.bool, device=bias.device)
        full[:p, :] = torch.cat([
            torch.ones(p, p, dtype=torch.bool, device=bias.device),
            torch.zeros(p, k_len - p, dtype=torch.bool, device=bias.device)
        ], dim=1)
        suffix = torch.zeros(q_len - p, k_len, dtype=torch.bool, device=bias.device)
        suffix[:, :p] = True
        suffix[:, p:] = causal[p:, p:]
        full[p:, :] = suffix
        bias[b, 0].masked_fill_(~full, _neg_inf(dtype))
    return bias


def combine_biases(*biases: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Additive combination of attention biases.

    - Accepts None or broadcastable bias tensors and returns their sum.
    - If any position is masked (-inf) by any component, the sum remains -inf.
    - Typical use: combine causal and padding masks.
    """
    out = None
    for b in biases:
        if b is None:
            continue
        out = b if out is None else out + b
    return out
