from .mha import run_demo_mha
from .gqa_rope import run_demo_gqa_rope
from .mqa_cross import run_demo_mqa_cross
from .moe_prefix import run_demo_moe_prefix
from .cross_attn import run_demo_cross_attn
from .kv_cache import run_demo_kv_cache


def run_demo():
    run_demo_mha()
    run_demo_gqa_rope()
    run_demo_mqa_cross()
    run_demo_moe_prefix()
    run_demo_cross_attn()
    run_demo_kv_cache()


if __name__ == "__main__":
    run_demo()
