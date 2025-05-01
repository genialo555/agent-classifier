# AI-Assisted (2025-04-18): first pass implementation of Rule Engine for Sprint 1.
# Review focus : performance & correctness of Rule 5 (exact dup) and Rule 1 (JW+cosine).

"""03_apply_rules.py — Apply the Cursor rules defined in `.cursor/rules/sprint1.xml`

Currently implemented rules
---------------------------
Rule 5 (dup-merge-5):  exact duplicates after normalisation (Unicode NFKC, accents stripped, lower-case).
Rule 1 (sim-merge-1):  lexical Jaro-Winkler ≥ 0.80 **and** cosine embedding similarity ≥ 0.85.

Outputs
-------
* `data/merge_pairs.csv` — two-column file (source_id,target_id) indicating that *source_id* must be merged into *target_id*.
* `logs/cot.log` — appends a Chain-of-Thought line per triggered rule.

Usage
-----
$ python scripts/03_apply_rules.py --taxo data/taxo_raw.csv --emb data/embeddings.npy

Use `--limit 1000` during development to cap the number of rows (faster).
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import torch
from text_unidecode import unidecode
from tqdm import tqdm  # local import to avoid mandatory dep if not installed

# ---------------------------------------------------------------------------
# Similarity function (prefer rapidfuzz & C-extensions for speed)
# ---------------------------------------------------------------------------

try:
    from rapidfuzz.distance import JaroWinkler as _JW  # type: ignore

    def jw_sim(s1: str, s2: str) -> float:  # noqa: N802 – keep camelCase for clarity
        """Return Jaro-Winkler similarity in [0,1] using rapidfuzz's fast C backend."""

        return _JW.normalized_similarity(s1, s2)

except ImportError:
    try:
        from jaro import jaro_winkler_metric as jw_sim  # type: ignore
    except ImportError:
        from jellyfish import jaro_winkler_similarity as jw_sim  # type: ignore  # noqa: F401

    # rapidfuzz not available; jw_sim is provided by fallback implementation.

# ---------------------------------------------------------------------------
# Constants (extracted from sprint1.xml but hard-coded for now)
# ---------------------------------------------------------------------------
LEXICAL_THRESHOLD = 0.80  # Jaro-Winkler
EMBED_THRESHOLD = 0.85    # cosine similarity

LOG_PATH = Path("logs/cot.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_MERGE_PAIRS = Path("data/merge_pairs.csv")

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    return text.lower()


def load_taxonomy(path: Path, limit: int | None = None) -> Tuple[List[str], pd.DataFrame]:
    """Return (labels, df). Labels are concatenated strings used for similarity."""
    df = pd.read_csv(path, sep=",", header=None, dtype=str, keep_default_na=False)
    if limit:
        df = df.head(limit)
    labels = [" | ".join([c for c in row if isinstance(c, str) and c.strip()]) for row in df.values.tolist()]
    return labels, df


# ---------------------------------------------------------------------------
# Rule 5 — exact duplicates after normalisation
# ---------------------------------------------------------------------------

def rule5_exact_duplicates(labels: List[str]) -> List[Tuple[int, int]]:
    """Return (src,target) merge pairs for exact duplicates after normalization, buffering logs for performance."""
    mapping: Dict[str, int] = {}
    merges: List[Tuple[int, int]] = []
    # buffer duplicate logs to avoid per-iteration file opens
    fh = LOG_PATH.open("a", encoding="utf-8")
    for idx, lbl in enumerate(labels):
        key = normalise(lbl)
        if key in mapping:
            merges.append((idx, mapping[key]))  # merge current into first occurrence
            fh.write(f"rule=dup-merge-5 src={idx} tgt={mapping[key]} norm='{key}'\n")
        else:
            mapping[key] = idx
    fh.close()
    return merges


# ---------------------------------------------------------------------------
# Rule 1 — lexical ≥ 0.80 & embedding ≥ 0.85
# ---------------------------------------------------------------------------

def rule1_similar_merge(labels: List[str], embeddings: np.ndarray, block_size: int | None = None) -> List[Tuple[int, int]]:
    """Return list of merge pairs according to Rule 1 (JW & cosine).
    Use Faiss for approximate neighbor search if available to avoid O(n²), else fallback to block-wise computation."""

    n = len(labels)
    if n == 0:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tensor float16 on CUDA to reduce memory/temps; float32 on CPU
    dt = torch.float16 if device.type == "cuda" else torch.float32
    emb_t = torch.from_numpy(embeddings).to(device=device, dtype=dt)
    emb_t = torch.nn.functional.normalize(emb_t, p=2, dim=1)
    n, dim = emb_t.shape
    # Attempt Faiss-based approximate search
    try:
        import faiss
        use_faiss = True
    except ImportError:
        use_faiss = False
    if use_faiss:
        # Move embeddings to CPU float32 for Faiss
        emb_cpu = emb_t.float().cpu().numpy()
        # Build FAISS index for inner product (cosine similarity)
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(emb_cpu)
        index.add(emb_cpu)
        # Search each vector's top-k neighbors (heuristic k=64)
        k = min(n, 64)
        D, I = index.search(emb_cpu, k)
        merges: List[Tuple[int, int]] = []
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            for i in range(n):
                for score, j in zip(D[i], I[i]):
                    if j <= i or score < EMBED_THRESHOLD:
                        continue
                    lex_sim = jw_sim(labels[i], labels[j])
                    if lex_sim >= LEXICAL_THRESHOLD:
                        merges.append((j, i))
                        fh.write(f"rule=sim-merge-1 src={j} tgt={i} jw={lex_sim:.3f}, cos={score:.3f}\n")
        return merges

    # Fallback: block-wise cosine computation
    if block_size is None or n <= block_size:
        cos_mat = emb_t @ emb_t.T  # (n, n)
        cos_mat = cos_mat.float().cpu().numpy()
        rows, cols = np.where(np.triu(cos_mat, k=1) >= EMBED_THRESHOLD)
        cos_lookup = cos_mat
    else:
        rows_list: List[int] = []
        cols_list: List[int] = []
        cos_scores: Dict[Tuple[int, int], float] = {}
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = emb_t[start:end]  # (b, d)
            cos_part = block @ emb_t.T  # (b, n)
            cos_part = cos_part.float().cpu().numpy()

            # For indices in upper triangle only
            for i_b, i_global in enumerate(range(start, end)):
                row = cos_part[i_b]
                cols = np.where(row >= EMBED_THRESHOLD)[0]
                for j in cols:
                    if j <= i_global:
                        continue  # ensure upper triangle
                    rows_list.append(i_global)
                    cols_list.append(j)
                    cos_scores[(i_global, j)] = float(row[j])

        rows = np.array(rows_list, dtype=int)
        cols = np.array(cols_list, dtype=int)

        def _cos(i: int, j: int) -> float:
            return cos_scores[(i, j)]

        cos_lookup = _cos  # type: ignore

    # buffer similarity logs to reduce I/O overhead
    merges: List[Tuple[int, int]] = []
    fh = LOG_PATH.open("a", encoding="utf-8")
    log_buffer: List[str] = []
    for i, j in tqdm(zip(rows, cols), total=len(rows), desc="rule1", disable=len(rows) < 1000):
        lex_sim = jw_sim(labels[i], labels[j])
        if lex_sim >= LEXICAL_THRESHOLD:
            merges.append((j, i))  # keep left-most (smaller index = i)
            if callable(cos_lookup):
                cos_val = cos_lookup(i, j)
            else:
                cos_val = cos_lookup[i, j]
            log_buffer.append(f"rule=sim-merge-1 src={j} tgt={i} jw={lex_sim:.3f}, cos={cos_val:.3f}\n")
            if len(log_buffer) >= 1000:
                fh.write("".join(log_buffer))
                log_buffer.clear()
    # flush remaining logs and close file
    if log_buffer:
        fh.write("".join(log_buffer))
    fh.close()

    # Free GPU memory ASAP
    if torch.cuda.is_available():
        del emb_t
        if 'cos_mat' in locals(): del cos_mat
        torch.cuda.empty_cache()

    return merges


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_cot(src_id: int, tgt_id: int, rule_id: str, details: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"rule={rule_id} src={src_id} tgt={tgt_id} {details}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Cursor rules to taxonomy.")
    parser.add_argument("--taxo", default="data/taxo_raw.csv", help="Path to normalised taxonomy CSV.")
    parser.add_argument("--emb", default="data/embeddings.npy", help="Embeddings 🡒 NumPy .npy file.")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of rows for speed (dev only).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--block-size", type=int, help="Compute cosine similarity by blocks of this size to reduce peak memory (GPU/CPU).")
    parser.add_argument("--filter", help="Regex to select only labels to process (case-insensitive).")
    args = parser.parse_args()

    # Validate regex filter early
    if args.filter:
        try:
            re.compile(args.filter, flags=re.IGNORECASE)
        except re.error as e:
            parser.error(f"Invalid regex for --filter: {e}")

    labels, df = load_taxonomy(Path(args.taxo), args.limit)
    embeddings = np.load(args.emb)
    if args.limit:
        embeddings = embeddings[: args.limit]

    # Optional regex-based filtering
    if args.filter:
        rx = re.compile(args.filter, flags=re.IGNORECASE)
        keep_idx = [i for i, lbl in enumerate(labels) if rx.search(lbl)]
        if not keep_idx:
            print(f"[apply_rules] no label matches pattern '{args.filter}'. Exiting.")
            return

        labels = [labels[i] for i in keep_idx]
        df = df.iloc[keep_idx].reset_index(drop=True)
        embeddings = embeddings[keep_idx]
        print(f"[apply_rules] filter='{args.filter}' → {len(labels)} rows remaining")

    # Rule application with progress metrics
    # Rule 5: exact duplicates
    print(f"[apply_rules] Starting Rule 5 (exact duplicates) on {len(labels)} labels")
    merges5 = rule5_exact_duplicates(labels)
    print(f"[apply_rules] Rule 5 completed: {len(merges5)} merges (exact duplicates)")
    # Rule 1: lexical + embedding similarity
    print(f"[apply_rules] Starting Rule 1 (Jaro-Winkler ≥ {LEXICAL_THRESHOLD} & cosine ≥ {EMBED_THRESHOLD}) on {len(labels)} labels with block size={args.block_size}")
    merges1 = rule1_similar_merge(labels, embeddings, block_size=args.block_size)
    print(f"[apply_rules] Rule 1 completed: {len(merges1)} merges (similarity)")
    # Combine results and report overall metrics
    merges = merges5 + merges1
    total = len(merges)
    pct = (total / len(labels) * 100) if labels else 0
    print(f"[apply_rules] Total merges: {total} ({pct:.2f}% of {len(labels)} labels)")

    if args.dry_run:
        print(f"[apply_rules] would create {len(merges)} merges (dry run). Example: {merges[:5]}")
        return

    # Sort for deterministic ordering (by target, then source)
    if merges:
        merges_sorted = sorted(merges, key=lambda x: (x[1], x[0]))
    else:
        merges_sorted = merges

    OUTPUT_MERGE_PAIRS.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(merges_sorted, columns=["source_id", "target_id"]).to_csv(OUTPUT_MERGE_PAIRS, index=False)
    print(f"[apply_rules] wrote {OUTPUT_MERGE_PAIRS} with {len(merges_sorted)} merges")


if __name__ == "__main__":
    # basic console log config
    logging.basicConfig(level=logging.INFO)
    main() 