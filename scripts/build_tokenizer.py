"""
Train a 10k BPE tokenizer on TinyStories and save as tokenizer.json

Usage:
  python scripts/build_tokenizer.py --dataset roneneldan/TinyStories --out tokenizer.json --vocab_size 10000
"""
import argparse, os
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="roneneldan/TinyStories")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--out", default="tokenizer.json")
    ap.add_argument("--vocab_size", type=int, default=10_000)
    ap.add_argument("--limit", type=int, default=200_000, help="limit number of rows for quick training (set -1 for all)")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=False)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    def text_iter():
        for ex in ds:
            yield ex[args.text_field]

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

    trainer = BpeTrainer(vocab_size=args.vocab_size, min_frequency=2, special_tokens=["[PAD]","[UNK]","[BOS]","[EOS]"])
    tokenizer.train_from_iterator(text_iter(), trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")),
                        ("[EOS]", tokenizer.token_to_id("[EOS]"))],
    )
    tokenizer.save(args.out)
    print(f"Saved tokenizer to {args.out}")

if __name__ == "__main__":
    main()

