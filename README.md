# TinyStories MoE (DeepSeek-style) — <10M params

This repo is a tiny, Apple‑silicon–friendly training stack to reproduce a decoder-only GPT on **TinyStories**, swapping the dense MLP for a **DeepSeek-style MoE** (top‑k routing with capacity). The default config is **MOE_7_4M** (~7.4M params) so you can iterate quickly on an M1 Max (64 GB).

## Features
- Decoder-only GPT with **MoE MLP** (Top‑k=2, capacity factor=1.25)
- **<10M** parameters (default ~7.4M) with weight tying
- HF **datasets** loader for `roneneldan/TinyStories`
- One-file **tokenizer trainer** → 10k BPE (`tokenizer.json`)
- MPS‑aware training loop (fp16/bf16 autocast), cosine LR, grad clipping
- Simple sampler with nucleus sampling

---

## Install

```bash
# (optional) conda create -n moe python=3.10 && conda activate moe
pip install -r requirements.txt
```

> Apple Silicon: `pip install torch --index-url https://download.pytorch.org/whl/cpu` works, but prefer the official instructions for MPS if needed.

---

## 1) Train a 10k tokenizer

```bash
python scripts/build_tokenizer.py --dataset roneneldan/TinyStories --out tokenizer.json --vocab_size 10000 --limit 200000
```

- This quickly trains a small BPE over a subset (`--limit`) of TinyStories.  
- You can remove `--limit` to use the full dataset (slower).

---

## 2) Train the MOE_7_4M model

Default config is defined in `models/config.py` (6 layers, d_model=256, heads=8, MoE: 4 experts, d_ff_expert=256).

```bash
python scripts/train.py   --tokenizer tokenizer.json   --out_dir checkpoints   --batch_size 8   --grad_accum_steps 1   --max_steps 2000   --eval_every 200
```

**Tips**
- On M1 Max, `device` auto‑selects **mps**. You can force CPU/CUDA by editing `Config.device`.
- Increase `batch_size` and/or `grad_accum_steps` until memory is comfy.
- Adjust `Config.lr/warmup` for your token budget; defaults are conservative.

---

## 3) Sample from a checkpoint

```bash
python scripts/sample.py --ckpt checkpoints/moe_final_step2000.pt --tokenizer tokenizer.json --prompt "Once upon a time,"
```

---

## Config summary (MOE_7_4M)

- `vocab_size=10_000`, `seq_len=512`, `n_layers=6`, `d_model=256`, `n_heads=8`  
- MoE: `num_experts=4`, `k=2`, `d_ff_expert=256`, `capacity_factor=1.25`  
- Loss = CE + `1e-2 * aux_balance` + `1e-4 * router_z`  
- Optim: AdamW (β1=0.9, β2=0.95, wd=0.1), `lr=3e-4`, cosine with warmup

---

## Notes & gotchas

- **Tokenizer/vocab size controls params** (weight tying keeps the head size = embedding size). Stick to ~10k to stay under 10M.
- The MoE router here is **simple & readable**. For speed on multi‑GPU, you’d implement expert parallel + all‑to‑all.
- Overflow handling is greedy + capacity, which is fine in tiny regimes. Increase `capacity_factor` if too many tokens drop.
- Evaluate with perplexity on a held‑out slice; for qualitative checks, use `scripts/sample.py`.

---

## Layout

```
tinymoe/
├── models/
│   ├── config.py
│   └── model.py
├── scripts/
│   ├── build_tokenizer.py
│   ├── train.py
│   └── sample.py
├── requirements.txt
└── README.md
```

---

## License

MIT
