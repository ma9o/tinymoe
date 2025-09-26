import os, math, argparse, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from models.config import Config
from models.model import TinyMoeGPT, moe_aux_loss, router_z_loss

def get_device(cfg: Config):
    if cfg.device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device

class PackedDataset(Dataset):
    def __init__(self, texts, tok, seq_len):
        ids = []
        for t in texts:
            enc = tok.encode(t)
            ids.extend(enc.ids)
        # pack into fixed blocks
        # drop remainder for simplicity
        n = (len(ids) // seq_len) * seq_len
        ids = ids[:n]
        self.data = torch.tensor(ids, dtype=torch.long).view(-1, seq_len)
    def __len__(self):
        return self.data.size(0)
    def __getitem__(self, idx):
        x = self.data[idx]
        return x

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
    args = ap.parse_args()

    cfg = Config()
    cfg.grad_accum_steps = args.grad_accum_steps
    cfg.max_steps = args.max_steps

    os.makedirs(args.out_dir, exist_ok=True)

    device = get_device(cfg)
    dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16

    # Tokenizer
    tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    tok.pad_token = "[PAD]"; tok.unk_token="[UNK]"; tok.bos_token="[BOS]"; tok.eos_token="[EOS]"

    # Data
    train_ds = load_dataset(cfg.dataset_name, split="train")
    val_ds = load_dataset(cfg.dataset_name, split="validation")

    train_texts = train_ds[cfg.text_field]
    val_texts = val_ds[cfg.text_field][:5000]  # small eval slice

    train_set = PackedDataset(train_texts, tok, cfg.seq_len)
    val_set = PackedDataset(val_texts, tok, cfg.seq_len)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TinyMoeGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda" and cfg.precision=="fp16"))

    global_step = 0
    best_val = float("inf")

    for step, batch in enumerate(tqdm(train_loader, total=cfg.max_steps)):
        if global_step >= cfg.max_steps:
            break
        model.train()
        x = batch.to(device)

        lr = cosine_lr(global_step, cfg.max_steps, cfg.lr, warmup=cfg.warmup_steps)
        for pg in opt.param_groups: pg["lr"] = lr

        with torch.autocast(device_type=("cuda" if device=="cuda" else "cpu" if device=="cpu" else "mps"), dtype=dtype):
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

        if scaler.is_enabled():
            scaler.scale(loss / cfg.grad_accum_steps).backward()
        else:
            (loss / cfg.grad_accum_steps).backward()

        if (global_step + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        if (global_step % cfg.log_every) == 0:
            print(f"step {global_step} | loss {loss.item():.4f} | ce {ce.item():.4f} | aux {float(aux_loss):.4f} | z {float(z_loss):.4f} | lr {lr:.2e}")

        if (global_step % args.eval_every) == 0 and global_step > 0:
            val_ppl = evaluate(model, val_loader, device, dtype)
            print(f"[eval] step {global_step} | ppl {val_ppl:.2f}")
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
    with torch.no_grad(), torch.autocast(device_type=("cuda" if device=="cuda" else "cpu" if device=="cpu" else "mps"), dtype=dtype):
        for batch in loader:
            x = batch.to(device)
            logits, _ = model(x)
            V = logits.size(-1)
            ce = F.cross_entropy(logits[:, :-1].contiguous().view(-1, V), x[:, 1:].contiguous().view(-1))
            losses.append(ce.item())
    import math
    return math.exp(sum(losses)/len(losses))

if __name__ == "__main__":
    main()

