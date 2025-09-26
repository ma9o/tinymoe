import math, torch, torch.nn as nn, torch.nn.functional as F

class ExpertMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MoeMLP(nn.Module):
    """DeepSeek-style MoE with top-k routing and capacity limit.
       Simplicity/clarity > raw speed. Good for <10M param models."""
    def __init__(self, d_model, d_ff_expert, num_experts=4, k=2, capacity_factor=1.25):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(d_model, d_ff_expert) for _ in range(num_experts)])

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)
        logits = self.router(x_flat)           # [N, E]
        gate = F.softmax(logits, dim=-1)       # [N, E]
        topv, topi = gate.topk(self.k, dim=-1) # [N, k]
        capacity = int(self.capacity_factor * (N / self.num_experts)) + 1

        # Greedy packing into expert capacity
        out = torch.zeros_like(x_flat)
        usage = gate.mean(0)  # for load balancing

        # Two-phase assignment (first choice, then second) is simple and stable for small models
        counts = [0 for _ in range(self.num_experts)]
        for choice in range(self.k):
            e_sel = topi[:, choice]         # [N]
            w_sel = topv[:, choice]         # [N]
            for e in range(self.num_experts):
                mask = (e_sel == e).nonzero(as_tuple=False).flatten()
                if mask.numel() == 0: 
                    continue
                take = mask[: max(0, capacity - counts[e])]
                if take.numel() == 0: 
                    continue
                y = self.experts[e](x_flat[take])                 # [Ne, D]
                out.index_add_(0, take, y * w_sel[take].unsqueeze(-1))
                counts[e] += take.numel()

        denom = (topv.sum(-1, keepdim=True) + 1e-9)               # [N,1]
        out = out / denom
        return out.reshape(B, T, D), usage, logits

def moe_aux_loss(usage):
    if usage is None:
        return torch.tensor(0.0)
    E = usage.numel()
    return (usage * (E * usage + 1e-9).log()).sum()

def router_z_loss(logits):
    if logits is None:
        return torch.tensor(0.0)
    return logits.float().pow(2).mean()

class MlpDense(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = mlp

        self._aux = (None, None)  # (usage, logits)

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        m = self.mlp(h)
        if isinstance(m, tuple):
            m, usage, logits = m
            self._aux = (usage, logits)
        else:
            self._aux = (None, None)
        return x + m

    def aux(self):
        return self._aux

class TinyMoeGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        V, T, L, D, H = cfg.vocab_size, cfg.seq_len, cfg.n_layers, cfg.d_model, cfg.n_heads
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(T, D)
        self.blocks = nn.ModuleList()
        for _ in range(L):
            if cfg.mlp_type == "moe":
                mlp = MoeMLP(D, cfg.d_ff_expert, cfg.num_experts, cfg.k, cfg.capacity_factor)
            else:
                mlp = MlpDense(D, 4 * D, dropout=cfg.dropout)
            self.blocks.append(Block(D, H, mlp, dropout=cfg.dropout))
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight
        self.seq_len = T

    @staticmethod
    def causal_mask(T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        mask = self.causal_mask(T, x.device)
        aux = []
        for blk in self.blocks:
            h = blk(h, attn_mask=mask)
            aux.append(blk.aux())
        h = self.ln_f(h)
        return self.head(h), aux

