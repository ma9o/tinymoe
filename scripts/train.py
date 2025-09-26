import os, math, argparse, time, json, sys
from contextlib import nullcontext
import _bootstrap; _bootstrap.bootstrap()
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from models.config import Config
from models.model import TinyMoeGPT, moe_aux_loss, router_z_loss
from utils.data_cache import prepare_packed_datasets

def get_device(cfg: Config):
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS device not available. Please use an Apple Silicon Mac with MPS enabled (PYTORCH_ENABLE_MPS_FALLBACK=1).")
    return "mps"

# (Packing/caching moved to utils.data_cache)

def cosine_lr(step, max_steps, max_lr, min_lr=1e-5, warmup=1000):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5*(max_lr-min_lr)*(1 + math.cos(math.pi*progress))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--train_limit", type=int, default=200_000, help="limit number of train rows (-1 for all)")
    ap.add_argument("--val_limit", type=int, default=5_000, help="limit number of validation rows (-1 for all)")
    ap.add_argument("--cache_dir", default=None, help="directory to cache packed tensors (default: <out_dir>/cache)")
    ap.add_argument("--rebuild_cache", action="store_true", help="force rebuild tokenized cache")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default=None, help="override precision (default from Config)")
    ap.add_argument("--num_workers", type=int, default=None, help="override DataLoader workers (macOS often fastest with 0)")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor when num_workers>0")
    ap.add_argument("--mlp_type", choices=["moe", "dense"], default=None, help="override MLP type")
    args = ap.parse_args()

    cfg = Config()
    cfg.grad_accum_steps = args.grad_accum_steps
    cfg.max_steps = args.max_steps
    if args.precision is not None:
        cfg.precision = args.precision
    if args.mlp_type is not None:
        cfg.mlp_type = args.mlp_type
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    elif sys.platform == "darwin":
        # On macOS, tensor datasets often run fastest with single-threaded loading.
        cfg.num_workers = 0

    os.makedirs(args.out_dir, exist_ok=True)
    metrics_path = os.path.join(args.out_dir, "train_metrics.jsonl")

    device = get_device(cfg)
    use_amp = cfg.precision in ("fp16", "bf16")
    dtype = (torch.float16 if cfg.precision == "fp16" else (torch.bfloat16 if cfg.precision == "bf16" else torch.float32))
    print(f"[setup] device={device} precision={cfg.precision} dtype={dtype}")

    

    # Tokenizer
    tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    tok.pad_token = "[PAD]"; tok.unk_token="[UNK]"; tok.bos_token="[BOS]"; tok.eos_token="[EOS]"
    try:
        vocab_size = tok.vocab_size if hasattr(tok, "vocab_size") else len(tok.get_vocab())
    except Exception:
        vocab_size = None
    print(f"[tokenizer] path={args.tokenizer} vocab_size={vocab_size}")
    if vocab_size is not None and vocab_size != cfg.vocab_size:
        cfg.vocab_size = int(vocab_size)
        print(f"[config] adjusted cfg.vocab_size -> {cfg.vocab_size}")

    cache_dir = args.cache_dir or os.path.join(args.out_dir, 'cache')
    train_set, val_set = prepare_packed_datasets(
        cfg, tok,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        cache_dir=cache_dir,
        rebuild_cache=args.rebuild_cache,
        tokenizer_path=os.path.abspath(args.tokenizer),
    )

    dl_common = {"batch_size": args.batch_size}
    train_dl_kwargs = dict(dl_common)
    val_dl_kwargs = dict(dl_common)
    train_dl_kwargs.update({"shuffle": True, "num_workers": cfg.num_workers})
    val_dl_kwargs.update({"shuffle": False, "num_workers": cfg.num_workers})
    if cfg.num_workers and cfg.num_workers > 0:
        train_dl_kwargs.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
        val_dl_kwargs.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    train_loader = DataLoader(train_set, **train_dl_kwargs)
    val_loader = DataLoader(val_set, **val_dl_kwargs)
    tokens_per_step = args.batch_size * cfg.seq_len
    eff_tokens_per_opt = tokens_per_step * cfg.grad_accum_steps
    print(f"[loader] batch_size={args.batch_size} grad_accum={cfg.grad_accum_steps} tokens/step={tokens_per_step} tokens/opt_step={eff_tokens_per_opt}")

    # Model
    model = TinyMoeGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,} trainable={n_trainable:,} (~{n_params/1e6:.2f}M)")

    global_step = 0
    best_val = float("inf")
    last_log_time = time.time()
    tokens_since_last = 0
    steps_since_last = 0
    data_time_accum = 0.0
    compute_time_accum = 0.0
    last_gnorm = None
    last_iter_end = time.time()

    pbar = tqdm(train_loader, total=cfg.max_steps)
    for step, batch in enumerate(pbar):
        if global_step >= cfg.max_steps:
            break
        model.train()
        t_fetch_start = time.time()
        data_wait = t_fetch_start - last_iter_end
        x = batch.to(device, non_blocking=True)
        t_after_x = time.time()

        lr = cosine_lr(global_step, cfg.max_steps, cfg.lr, warmup=cfg.warmup_steps)
        for pg in opt.param_groups: pg["lr"] = lr

        amp_ctx = torch.autocast(device_type="mps", dtype=dtype) if use_amp else nullcontext()
        t_compute_start = t_after_x
        with amp_ctx:
            logits, aux = model(x)
            V = logits.size(-1)
            ce = F.cross_entropy(logits[:, :-1].contiguous().view(-1, V), x[:, 1:].contiguous().view(-1))

            aux_loss = 0.0; z_loss = 0.0
            for usage, z in aux:
                if usage is not None:
                    aux_loss = aux_loss + moe_aux_loss(usage)
                if z is not None:
                    z_loss = z_loss + router_z_loss(z)

            loss = ce + cfg.aux_loss_coeff*aux_loss + cfg.z_loss_coeff*z_loss

        (loss / cfg.grad_accum_steps).backward()

        if (global_step + 1) % cfg.grad_accum_steps == 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            last_gnorm = float(gnorm) if hasattr(gnorm, 'item') else float(gnorm)
            opt.step()
            opt.zero_grad(set_to_none=True)

        t_step_end = time.time()
        data_time_accum += (data_wait + (t_after_x - t_fetch_start))
        compute_time_accum += (t_step_end - t_compute_start)
        last_iter_end = t_step_end

        tokens_since_last += tokens_per_step
        steps_since_last += 1
        if (global_step % cfg.log_every) == 0:
            try:
                torch.mps.synchronize()
            except Exception:
                pass
            now = time.time()
            dt = max(1e-6, now - last_log_time)
            tps = tokens_since_last / dt
            avg_data = data_time_accum / max(1, steps_since_last)
            avg_comp = compute_time_accum / max(1, steps_since_last)
            frac_data = (data_time_accum / max(1e-9, data_time_accum + compute_time_accum))
            msg = (
                f"step {global_step} | loss {loss.item():.4f} | ce {ce.item():.4f} | "
                f"aux {float(aux_loss):.4f} | z {float(z_loss):.4f} | lr {lr:.2e} | "
                f"toks/s {tps:,.0f} | data {avg_data*1000:.1f}ms | comp {avg_comp*1000:.1f}ms | data% {frac_data*100:.1f} | "
                f"gnorm {('-' if last_gnorm is None else f'{last_gnorm:.2f}')}"
            )
            print(msg, flush=True)
            # tqdm postfix
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'ce': f"{ce.item():.3f}",
                'lr': f"{lr:.2e}",
                't/s': f"{tps:,.0f}",
                'data%': f"{frac_data*100:.0f}"
            })
            # persist metrics
            try:
                with open(metrics_path, 'a') as f:
                    f.write(json.dumps({
                        'event': 'train',
                        'step': int(global_step),
                        'loss': float(loss.item()),
                        'ce': float(ce.item()),
                        'aux': float(aux_loss),
                        'z': float(z_loss),
                        'lr': float(lr),
                        'tokens_per_sec': float(tps),
                        'avg_data_time_s': float(avg_data),
                        'avg_compute_time_s': float(avg_comp),
                        'data_fraction': float(frac_data),
                        'grad_norm': (None if last_gnorm is None else float(last_gnorm)),
                    }) + "\n")
            except Exception:
                pass
            last_log_time = now
            tokens_since_last = 0
            steps_since_last = 0
            data_time_accum = 0.0
            compute_time_accum = 0.0

        if (global_step % args.eval_every) == 0 and global_step > 0:
            val_ppl = evaluate(model, val_loader, device, dtype)
            print(f"[eval] step {global_step} | ppl {val_ppl:.2f}")
            try:
                with open(metrics_path, 'a') as f:
                    f.write(json.dumps({
                        'event': 'eval',
                        'step': int(global_step),
                        'ppl': float(val_ppl)
                    }) + "\n")
            except Exception:
                pass
            if val_ppl < best_val:
                best_val = val_ppl
                ckpt = os.path.join(args.out_dir, f"moe_step{global_step}_ppl{val_ppl:.2f}.pt")
                torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, ckpt)
                print(f"saved {ckpt}")

        global_step += 1

    # final save
    ckpt = os.path.join(args.out_dir, f"moe_final_step{global_step}.pt")
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, ckpt)
    print(f"saved {ckpt}")

def evaluate(model, loader, device, dtype):
    model.eval()
    losses = []
    use_amp = (dtype in (torch.float16, torch.bfloat16))
    amp_ctx = torch.autocast(device_type="mps", dtype=dtype) if use_amp else nullcontext()
    with torch.no_grad(), amp_ctx:
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            logits, _ = model(x)
            V = logits.size(-1)
            ce = F.cross_entropy(logits[:, :-1].contiguous().view(-1, V), x[:, 1:].contiguous().view(-1))
            losses.append(ce.item())
    import math
    return math.exp(sum(losses)/len(losses))

if __name__ == "__main__":
    main()
