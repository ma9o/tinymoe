from .common import toy_config
from ..model import TinyLM


def run_demo_gqa_rope():
    cfg = toy_config()
    vocab, d = cfg["vocab"], cfg["d"]
    x = cfg["x"]
    pad = cfg["pad"]

    model = TinyLM(
        vocab, max_len=512, d_model=d, n_layers=2, n_heads=8,
        n_kv_heads=2, pos_embedding="none", rope_in_attn=True, norm="rmsnorm",
        ffn_hidden_mult=3.0, use_moe=False, attn_dropout=0.0, resid_dropout=0.0,
    )
    logits, _, _ = model(x, attn_strategy="causal_padding", key_padding=pad)
    print("GQA logits:", tuple(logits.shape))


if __name__ == "__main__":
    run_demo_gqa_rope()
