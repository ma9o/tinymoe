import os
import json
import hashlib
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm


class PackedTensorDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor):
        self.data = data_tensor

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


def _sha1_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha1()
        with open(path, 'rb') as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def _cache_paths(base: str, tag: str) -> Tuple[str, str]:
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"{tag}_data.pt"), os.path.join(base, f"{tag}_meta.json")


def _load_cached(base: str, tag: str, expect_meta: dict) -> Optional[torch.Tensor]:
    data_path, meta_path = _cache_paths(base, tag)
    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        keys = [
            'tokenizer_sha1', 'tokenizer_path', 'dataset_name', 'text_field',
            'seq_len', 'train_limit', 'val_limit', 'vocab_size'
        ]
        for k in keys:
            if meta.get(k) != expect_meta.get(k):
                return None
        t = torch.load(data_path, map_location='cpu')
        if not isinstance(t, torch.Tensor) or t.dtype != torch.long:
            return None
        return t
    except Exception:
        return None


def _save_cached(base: str, tag: str, tensor: torch.Tensor, meta: dict) -> None:
    data_path, meta_path = _cache_paths(base, tag)
    tmp_data = data_path + ".tmp"
    tmp_meta = meta_path + ".tmp"
    torch.save(tensor, tmp_data)
    with open(tmp_meta, 'w') as f:
        json.dump(meta, f)
    os.replace(tmp_data, data_path)
    os.replace(tmp_meta, meta_path)


def _encode_pack_to_tensor(texts, tok, seq_len: int, progress_desc: Optional[str] = None) -> torch.Tensor:
    ids = []
    iterator = tqdm(texts, desc=progress_desc) if progress_desc else texts
    for t in iterator:
        enc = tok.encode(t)
        enc_ids = enc.ids if hasattr(enc, "ids") else enc
        ids.extend(enc_ids)
    n = (len(ids) // seq_len) * seq_len
    ids = ids[:n]
    return torch.tensor(ids, dtype=torch.long).view(-1, seq_len)


def prepare_packed_datasets(cfg, tok, *, train_limit: int, val_limit: int, cache_dir: str,
                            rebuild_cache: bool = False, tokenizer_path: Optional[str] = None):
    # Prefer explicit tokenizer path for stable caching/fingerprint
    tok_path = tokenizer_path or getattr(tok, 'tokenizer_file', None) or getattr(tok, 'name_or_path', None) or ''
    tok_sha1 = _sha1_file(tok_path)

    meta_common = {
        'tokenizer_sha1': tok_sha1,
        'tokenizer_path': tok_path,
        'dataset_name': cfg.dataset_name,
        'text_field': cfg.text_field,
        'seq_len': int(cfg.seq_len),
        'train_limit': int(train_limit),
        'val_limit': int(val_limit),
        'vocab_size': int(cfg.vocab_size),
    }

    train_tensor = None if rebuild_cache else _load_cached(cache_dir, 'train', meta_common)
    val_tensor = None if rebuild_cache else _load_cached(cache_dir, 'val', meta_common)

    if train_tensor is None or val_tensor is None:
        print(f"[data] loading dataset={cfg.dataset_name}")
        train_ds = load_dataset(cfg.dataset_name, split="train")
        val_ds = load_dataset(cfg.dataset_name, split="validation")

        if train_limit > 0:
            try:
                n = min(train_limit, len(train_ds))
                train_ds = train_ds.select(range(n))
            except Exception:
                pass
        if val_limit > 0:
            try:
                n = min(val_limit, len(val_ds))
                val_ds = val_ds.select(range(n))
            except Exception:
                pass

        train_texts = train_ds[cfg.text_field]
        val_texts = val_ds[cfg.text_field]

        print(f"[data] dataset={cfg.dataset_name} field={cfg.text_field}")
        print(f"[data] train_texts={len(train_texts)} val_texts={len(val_texts)}")

        train_tensor = _encode_pack_to_tensor(train_texts, tok, cfg.seq_len, progress_desc="[pack] train")
        val_tensor = _encode_pack_to_tensor(val_texts, tok, cfg.seq_len, progress_desc="[pack] val")
        print(f"[data] packed: train_seqs={train_tensor.size(0)} val_seqs={val_tensor.size(0)} seq_len={cfg.seq_len}")

        try:
            _save_cached(cache_dir, 'train', train_tensor, meta_common)
            _save_cached(cache_dir, 'val', val_tensor, meta_common)
            print(f"[cache] saved to {cache_dir}")
        except Exception as e:
            print(f"[cache] save failed: {e}")
    else:
        print(f"[cache] loaded pre-tokenized tensors from {cache_dir}")

    return PackedTensorDataset(train_tensor), PackedTensorDataset(val_tensor)
