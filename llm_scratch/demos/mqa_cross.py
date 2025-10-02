from .common import toy_config
from ..model import TinyLM


def run_demo_mqa_cross():
    cfg = toy_config()
    vocab, d = cfg["vocab"], cfg["d"]
    x = cfg["x"]
    mem_idx = cfg["mem_idx"]
    mem_pad = cfg["mem_pad"]

    # Use a tiny encoder to produce memory embeddings
    enc = TinyLM(
        vocab, max_len=512, d_model=d, n_layers=1, n_heads=4, n_kv_heads=4,
        pos_embedding="sinusoidal", rope_in_attn=False, norm="layernorm",
        ffn_hidden_mult=2.0, use_moe=False,
    )
    memory = enc.embed(mem_idx)

    model = TinyLM(
        vocab, max_len=512, d_model=d, n_layers=2, n_heads=8, n_kv_heads=1,
        pos_embedding="sinusoidal", rope_in_attn=False, norm="layernorm",
    )
    logits, _, _ = model(x, attn_strategy="causal", memory=memory, memory_padding=mem_pad)
    print("MQA + cross logits:", tuple(logits.shape))


if __name__ == "__main__":
    run_demo_mqa_cross()
