#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Group taxonomy labels by semantic similarity to a seed topic.

"""11_topic_grouping.py — Given a seed taxonomy label, find all labels whose embeddings are similar above a threshold."""
import argparse
import unicodedata
from text_unidecode import unidecode
from pathlib import Path
import numpy as np
import pandas as pd


def normalize(text: str) -> str:
    """Normalize text to Unicode NFKC, strip accents, and lowercase."""
    t = unicodedata.normalize("NFKC", text)
    t = unidecode(t)
    return t.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Group taxonomy labels by embedding similarity to a seed topic."
    )
    parser.add_argument(
        "--taxo", default="data/taxo_raw.csv",
        help="Path to the normalized taxonomy CSV (single-column labels)."
    )
    parser.add_argument(
        "--emb", default="data/embeddings.npy",
        help="Path to the embeddings numpy file produced by vectorization."
    )
    parser.add_argument(
        "--subject", required=True,
        help="Seed taxonomy label to group around (exact match in taxonomy labels)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.80,
        help="Cosine similarity threshold (0-1) to include labels in the group."
    )
    parser.add_argument(
        "--output", default="data/topic_group.csv",
        help="Path to write the topic grouping CSV."
    )
    args = parser.parse_args()

    # Load labels
    df_taxo = pd.read_csv(args.taxo, header=None, dtype=str)
    labels = df_taxo.iloc[:, 0].tolist()

    # Find seed index by normalized exact match
    seed_norm = normalize(args.subject)
    matches = [i for i, lbl in enumerate(labels) if normalize(lbl) == seed_norm]
    if not matches:
        raise ValueError(f"Subject '{args.subject}' not found exactly in taxonomy labels.")
    seed_idx = matches[0]

    # Load embeddings
    emb = np.load(args.emb)
    if seed_idx >= emb.shape[0]:
        raise ValueError(f"Seed index {seed_idx} out of bounds for embeddings of shape {emb.shape}.")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb_norm = emb / np.clip(norms, a_min=1e-12, a_max=None)
    seed_emb = emb_norm[seed_idx]

    # Compute cosine similarity
    sims = emb_norm.dot(seed_emb)

    # Select indices above threshold
    selected = np.where(sims >= args.threshold)[0]
    # Build output DataFrame
    df_out = pd.DataFrame({
        'member_id': selected,
        'member_label': [labels[i] for i in selected],
        'similarity': sims[selected]
    })
    # Sort by descending similarity
    df_out = df_out.sort_values(by='similarity', ascending=False)

    # Ensure output directory
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    df_out.to_csv(out_path, index=False)
    print(f"[topic_grouping] wrote {len(df_out)} labels grouped around '{args.subject}' to {args.output}")


if __name__ == '__main__':
    main() 