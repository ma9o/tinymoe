from .common import toy_config
from ..model import TinyLM


def run_demo_moe_prefix():
    cfg = toy_config()
    vocab, d = cfg["vocab"], cfg["d"]
    x = cfg["x"]
    prefix = cfg["prefix"]

    model = TinyLM(
        vocab, max_len=512, d_model=d, n_layers=2, n_heads=8, n_kv_heads=8,
        pos_embedding="learned", rope_in_attn=False, norm="rmsnorm",
        use_moe=True, num_experts=4, moe_top_k=2,
    )
    logits, _, _ = model(x, attn_strategy="prefix", prefix_lens=prefix)
    print("MoE logits:", tuple(logits.shape))


if __name__ == "__main__":
    run_demo_moe_prefix()
