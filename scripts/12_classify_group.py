#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Group taxonomy labels around a seed subject using the agent classifier.

"""12_classify_group.py — Use trained classifier to group taxonomy labels by merge probability with a seed subject."""
import argparse
import unicodedata
from text_unidecode import unidecode
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def jw_sim(s1: str, s2: str) -> float:
    """Return Jaro-Winkler similarity in [0,1]."""
    try:
        from rapidfuzz.distance import JaroWinkler as _JW
        return _JW.normalized_similarity(s1, s2)
    except ImportError:
        try:
            from jaro import jaro_winkler_metric as jw
        except ImportError:
            from jellyfish import jaro_winkler_similarity as jw
        return jw(s1, s2)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = unidecode(t)
    return t.lower()


def main():
    parser = argparse.ArgumentParser(description="Group taxonomy labels by classifier around a seed subject.")
    parser.add_argument("--taxo", default="data/taxo_raw.csv", help="Path to taxonomy CSV (raw labels).")
    parser.add_argument("--emb", default="data/embeddings.npy", help="Path to embeddings numpy file.")
    parser.add_argument("--model", default="models/agent_classifier.joblib", help="Path to trained classifier model.")
    parser.add_argument("--subject", required=True, help="Seed taxonomy label to group around.")
    parser.add_argument("--output", default="data/classifier_group.csv", help="Path to write grouped CSV.")
    parser.add_argument("--prob-threshold", type=float, default=0.5, help="Probability threshold for inclusion.")
    args = parser.parse_args()

    # Load data
    df_taxo = pd.read_csv(args.taxo, header=None, dtype=str)
    labels = df_taxo.iloc[:, 0].tolist()
    embs = np.load(args.emb)

    # Normalize labels for matching
    norm_labels = [normalize(lbl) for lbl in labels]
    seed_norm = normalize(args.subject)
    if seed_norm in norm_labels:
        seed_idx = norm_labels.index(seed_norm)
    else:
        # fallback: find label with highest lexical similarity
        sims = [jw_sim(args.subject, lbl) for lbl in labels]
        seed_idx = int(np.argmax(sims))
        best = labels[seed_idx]
        print(f"Warning: exact match not found. Using closest label '{best}' (JW={sims[seed_idx]:.3f}) as seed.")
        seed_norm = normalize(best)

    # Prepare classifier
    clf = joblib.load(args.model)

    # Normalize embeddings
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / np.clip(norms, a_min=1e-12, a_max=None)
    seed_emb = embs_norm[seed_idx]

    # For each label, compute features and predict
    rows = []
    for idx, lbl in enumerate(labels):
        lex = jw_sim(labels[seed_idx], lbl)
        cos = float(np.dot(embs_norm[idx], seed_emb))
        prob = clf.predict_proba([[lex, cos]])[0][1]
        pred = 1 if prob >= args.prob_threshold else 0
        rows.append({
            'seed_id': seed_idx,
            'seed_label': labels[seed_idx],
            'member_id': idx,
            'member_label': lbl,
            'lexical_similarity': lex,
            'cosine_similarity': cos,
            'merge_probability': prob,
            'include': bool(pred)
        })

    df_out = pd.DataFrame(rows)
    # Filter included
    df_incl = df_out[df_out['include']].sort_values(by='merge_probability', ascending=False)

    # Ensure output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_incl.to_csv(out_path, index=False)
    print(f"[classifier_group] wrote {len(df_incl)} labels grouped around '{labels[seed_idx]}' to {out_path}")

if __name__ == '__main__':
    main() 