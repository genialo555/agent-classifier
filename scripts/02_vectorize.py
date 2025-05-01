# AI-Assisted (2025-04-18): vectorise taxonomy labels to embeddings. Review focus: model selection & batching.

"""02_vectorize.py

Usage (default paths):
    python scripts/02_vectorize.py --input data/taxo_raw.csv --model gemma --batch-size 64

This script reads the normalised taxonomy CSV, encodes each row as a
single string, computes embeddings with the chosen encoder-only LLM and
stores them in NumPy format alongside a metadata CSV with node ⇔ row mapping.

Supported --model values:
    • gemma      → loads google/gemma-2b-it as encoder (CLS pooling)
    • deepseek   → loads deepseek-ai/deepseek-r1-3b (encoder side)
    • miniLM     → sentence-transformers/all-MiniLM-L6-v2 (default)

Add new models in _MODEL_REGISTRY below.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_DEFAULT_OUTPUT_DIR = Path("data")


# ---------------------------------------------------------------------------
# Helper utils
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_st_model(model_id: str, device: torch.device) -> Callable[[List[str]], torch.Tensor]:
    """Return an encode(texts) → (N, dim) function for *model_id* SentenceTransformer model."""

    st_model = SentenceTransformer(model_id, device=str(device))

    def _encode(list_strings: List[str]) -> torch.Tensor:
        emb = st_model.encode(list_strings, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
        return emb

    return _encode


def _load_hf_encoder(model_id: str, device: torch.device) -> Callable[[List[str]], torch.Tensor]:
    """Return encode function using HF AutoModel + CLS pooling."""

    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    mdl.eval()

    def _encode(list_strings: List[str]) -> torch.Tensor:
        with torch.no_grad():
            outs = []
            bs = 16
            for i in tqdm(range(0, len(list_strings), bs), desc="hf-encode", unit="batch", ncols=80):
                sub = list_strings[i : i + bs]
                batch = tok(sub, truncation=True, padding=True, return_tensors="pt").to(device)
                hidden = mdl(**batch).last_hidden_state  # (B, L, D)
                cls = hidden[:, 0]  # take CLS token
                outs.append(cls.cpu())
            return torch.cat(outs, dim=0)

    return _encode


_MODEL_REGISTRY = {
    "miniLM": lambda device: _load_st_model("sentence-transformers/all-MiniLM-L6-v2", device),
    "gemma": lambda device: _load_hf_encoder("google/gemma-2b-it", device),
    "deepseek": lambda device: _load_hf_encoder("deepseek-ai/deepseek-r1-3b", device),
}


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def read_taxonomy_csv(path: Path, pattern: str | None = None) -> List[str]:
    """Return list of concatenated taxonomy labels, optionally filtered by *pattern* (regex)."""

    df = pd.read_csv(path, sep=",", header=None, dtype=str, keep_default_na=False)
    # Concatenate non-empty hierarchy levels with " | "
    joined: List[str] = [
        " | ".join([c for c in row if isinstance(c, str) and c.strip()]) for row in df.values.tolist()
    ]

    if pattern:
        rx = re.compile(pattern, flags=re.IGNORECASE)
        joined = [s for s in joined if rx.search(s)]

    return joined


def save_embeddings(out_dir: Path, emb: torch.Tensor) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "embeddings.npy"
    np.save(npy_path, emb.cpu().numpy())
    print(f"[vectorize] saved {npy_path} (shape={tuple(emb.shape)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorise taxonomy labels to embeddings.")
    parser.add_argument("--input", default="data/taxo_raw.csv", help="Normalised taxonomy CSV.")
    parser.add_argument("--model", choices=list(_MODEL_REGISTRY.keys()), default="miniLM", help="Which encoder model to use.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for ST models (ignored for HF).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--filter", help="Regex to select only taxonomy rows whose concatenated label matches. Case-insensitive.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vectorize] using device={device}, model={args.model}")

    texts = read_taxonomy_csv(Path(args.input), args.filter)

    encode_fn = _MODEL_REGISTRY[args.model](device)
    embeddings = encode_fn(texts)

    if args.dry_run:
        print(embeddings[:5])
        return

    save_embeddings(_DEFAULT_OUTPUT_DIR, embeddings)

    # Record minimal metadata
    meta = {"input": args.input, "model": args.model, "dim": embeddings.shape[1], "filter": args.filter}
    with open(_DEFAULT_OUTPUT_DIR / "embeddings_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main() 