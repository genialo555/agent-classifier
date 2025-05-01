#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Group taxonomy labels for all canonical seeds using the agent classifier.

"""13_classify_all_groups.py — For each canonical group seed, use the trained classifier to group taxonomy labels and output a combined CSV."""
import argparse
import unicodedata
from text_unidecode import unidecode
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm


def jw_sim(s1: str, s2: str) -> float:
    try:
        from rapidfuzz.distance import JaroWinkler as _JW
        return _JW.normalized_similarity(s1, s2)
    except ImportError:
        # fallback to jaro_winkler package
        from jaro_winkler import jaro_winkler_similarity as jw
        return jw(s1, s2)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = unidecode(t)
    return t.lower()


def main():
    parser = argparse.ArgumentParser(description="Group taxonomy labels for all canonical seeds using classifier.")
    parser.add_argument("--taxo", default="data/taxo_raw.csv", help="Path to taxonomy CSV (raw labels).")
    parser.add_argument("--emb", default="data/embeddings.npy", help="Path to embeddings numpy file.")
    parser.add_argument("--merges", default="data/merge_pairs.csv", help="Path to merge pairs CSV (source->target).")
    parser.add_argument("--model", default="models/agent_classifier.joblib", help="Path to trained classifier model.")
    parser.add_argument("--prob-threshold", type=float, default=0.5, help="Probability threshold for inclusion.")
    parser.add_argument("--output", default="data/classifier_all_groups.csv", help="Path to write combined groups CSV.")
    args = parser.parse_args()

    # Load taxonomy and embeddings
    df_taxo = pd.read_csv(args.taxo, header=None, dtype=str)
    labels = df_taxo.iloc[:, 0].tolist()
    embs = np.load(args.emb)

    # Load merge pairs to identify unique seeds (targets)
    df_merges = pd.read_csv(args.merges)
    seeds = sorted(df_merges['target_id'].unique().tolist())

    # Prepare classifier model
    clf = joblib.load(args.model)

    # Pre-normalize labels
    norm_labels = [normalize(lbl) for lbl in labels]

    # Normalize embeddings for cosine
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / np.clip(norms, a_min=1e-12, a_max=None)

    rows = []
    # Iterate over each canonical seed with a progress bar
    for seed_idx in tqdm(seeds, desc="Processing seeds"):
        seed_label = labels[seed_idx]
        # Print current seed group being processed for visibility
        print(f"[classifier_all_groups] Processing seed_id={seed_idx}, seed_label='{seed_label}'")
        seed_emb = embs_norm[seed_idx]
        # Compute features and predictions for all labels
        for idx, lbl in enumerate(labels):
            lex = jw_sim(seed_label, lbl)
            cos = float(np.dot(embs_norm[idx], seed_emb))
            prob = clf.predict_proba([[lex, cos]])[0][1]
            include = prob >= args.prob_threshold
            if include:
                rows.append({
                    'seed_id': seed_idx,
                    'seed_label': seed_label,
                    'member_id': idx,
                    'member_label': lbl,
                    'lexical_similarity': lex,
                    'cosine_similarity': cos,
                    'merge_probability': prob
                })

    # Create DataFrame and sort by seed and probability
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(by=['seed_id', 'merge_probability'], ascending=[True, False])

    # Ensure output dir exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df_out.to_csv(out_path, index=False)
    print(f"[classifier_all_groups] wrote {len(df_out)} grouped labels across {len(seeds)} seeds to {out_path}")


if __name__ == '__main__':
    main() 