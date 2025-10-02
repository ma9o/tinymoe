from .common import toy_config
from ..model import TinyLM


def run_demo_cross_attn():
    cfg = toy_config()
    vocab, d = cfg["vocab"], cfg["d"]
    x = cfg["x"]          # decoder input (target tokens)
    mem_idx = cfg["mem_idx"]  # encoder input (source tokens)
    mem_pad = cfg["mem_pad"]

    # Encoder: produce memory features. For clarity, we use embeddings as memory.
    # In a full encoder-decoder, you'd pass the encoder's last hidden states.
    encoder = TinyLM(
        vocab, max_len=512, d_model=d, n_layers=2, n_heads=8, n_kv_heads=8,
        pos_embedding="sinusoidal", rope_in_attn=False, norm="layernorm",
        ffn_hidden_mult=3.0, use_moe=False,
    )
    memory = encoder.embed(mem_idx)  # [B, Tk, D]

    # Decoder: standard MHA, causal self-attn, cross-attends to encoder memory.
    decoder = TinyLM(
        vocab, max_len=512, d_model=d, n_layers=2, n_heads=8, n_kv_heads=8,
        pos_embedding="learned", rope_in_attn=False, norm="rmsnorm",
    )
    logits, _, _ = decoder(x, attn_strategy="causal", memory=memory, memory_padding=mem_pad)
    print("Cross-Attn (MHA) logits:", tuple(logits.shape))


if __name__ == "__main__":
    run_demo_cross_attn()
