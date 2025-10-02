import torch

from ..model import TinyLM


def run_demo_kv_cache():
    torch.manual_seed(0)
    vocab = 64
    d = 64
    n_layers = 2
    n_heads = 4
    t = 8
    bsz = 1

    x = torch.randint(0, vocab, (bsz, t))

    model = TinyLM(
        vocab, max_len=256, d_model=d, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_heads,
        pos_embedding="none", rope_in_attn=True, norm="rmsnorm",
    )

    # Full forward once (no cache), get logits for the last token
    logits_full, _, _ = model(x, attn_strategy="causal")
    last_full = logits_full[:, -1]

    # Incremental forward with cache, feeding one token at a time
    past_self = [None] * n_layers
    for i in range(t):
        xi = x[:, i:i+1]
        logits_step, past_self, _ = model(xi, attn_strategy="causal", past_kv_self=past_self)
    last_cached = logits_step[:, -1]

    # Compare logits equality/close
    max_diff = (last_full - last_cached).abs().max().item()
    print("KV cache demo - last token logits diff:", max_diff)
    print("Shapes: full=", tuple(last_full.shape), "cached=", tuple(last_cached.shape))


if __name__ == "__main__":
    run_demo_kv_cache()
