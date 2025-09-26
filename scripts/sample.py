import argparse, torch
import sys, os
# Ensure project root is on PYTHONPATH when running as a script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import PreTrainedTokenizerFast
from models.config import Config
from models.model import TinyMoeGPT

def load_ckpt(path, device):
    state = torch.load(path, map_location=device)
    cfg = Config(**state["cfg"])
    model = TinyMoeGPT(cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, cfg

def generate(model, tok, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9, device="mps"):
    ids = tok.encode(prompt)
    if hasattr(ids, "ids"):
        ids = ids.ids
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(x[:, -model.seq_len:])
            logits = logits[:, -1, :]
            logits = logits / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)

            # nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return tok.decode(x[0].tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--prompt", default="Once upon a time,")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS device not available. Please run on Apple Silicon with MPS enabled.")
    device = "mps"
    tok = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
    tok.pad_token = "[PAD]"; tok.unk_token="[UNK]"; tok.bos_token="[BOS]"; tok.eos_token="[EOS]"

    model, cfg = load_ckpt(args.ckpt, device)
    out = generate(model, tok, args.prompt, args.max_new_tokens, device=device)
    print(out)

if __name__ == "__main__":
    main()
