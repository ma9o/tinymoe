import torch


def toy_config(vocab: int = 128, bsz: int = 2, t: int = 16, mem_t: int = 10, d: int = 128, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randint(0, vocab, (bsz, t))
    pad = torch.zeros(bsz, t, dtype=torch.bool)
    prefix = torch.tensor([max(1, t // 3), max(1, (t // 2) + 1)])
    mem_idx = torch.randint(0, vocab, (bsz, mem_t))
    mem_pad = torch.zeros(bsz, mem_t, dtype=torch.bool)
    return {
        "vocab": vocab,
        "bsz": bsz,
        "t": t,
        "mem_t": mem_t,
        "d": d,
        "x": x,
        "pad": pad,
        "prefix": prefix,
        "mem_idx": mem_idx,
        "mem_pad": mem_pad,
    }

