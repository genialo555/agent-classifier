#!/usr/bin/env python3
# AI-Assisted (2025-04-18): train a binary classifier agent for merge decisions.

"""07_train_agent_classifier.py — Train a classifier for merge vs non-merge pairs."""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

# Compute lexical similarity
def jw_sim(s1: str, s2: str) -> float:
    try:
        from rapidfuzz.distance import JaroWinkler as _JW
        return _JW.normalized_similarity(s1, s2)
    except ImportError:
        try:
            from jaro import jaro_winkler_metric as jw
        except ImportError:
            from jellyfish import jaro_winkler_similarity as jw
        return jw(s1, s2)


def compute_features(labels, embeddings, pairs):
    """Return feature matrix X and label vector y for given pairs."""
    X = []
    y = []
    for (i, j, label) in pairs:
        lex = jw_sim(labels[i], labels[j])
        emb1 = embeddings[i]
        emb2 = embeddings[j]
        cos = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        X.append([lex, cos])
        y.append(label)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Train agent classifier for merge decisions.")
    parser.add_argument("--taxo", default="data/taxo_raw.csv", help="Normalized taxonomy CSV.")
    parser.add_argument("--emb", default="data/embeddings.npy", help="Embeddings numpy file.")
    parser.add_argument("--merges", default="data/merge_pairs.csv", help="Positive merge pairs CSV.")
    parser.add_argument("--neg-sample", type=int, default=10000, help="Number of negative random pairs.")
    parser.add_argument("--model-out", default="models/agent_classifier.joblib", help="Path to save trained model.")
    args = parser.parse_args()

    # Load data
    labels = pd.read_csv(args.taxo, header=None).astype(str).agg(" | ".join, axis=1).tolist()
    embeddings = np.load(args.emb)
    df_pos = pd.read_csv(args.merges)

    n = len(labels)
    # Positive examples
    pos_pairs = [(r.source_id, r.target_id, 1) for r in df_pos.itertuples(index=False)]

    # Negative sampling: random non-merge pairs
    all_indices = set((i, j) for i in range(n) for j in range(n) if i < j)
    pos_set = set((i, j) for i, j, _ in pos_pairs)
    candidates = list(all_indices - pos_set)
    neg_samples = random.sample(candidates, min(len(candidates), args.neg_sample))
    neg_pairs = [(i, j, 0) for i, j in neg_samples]

    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)

    # Compute features and labels
    X, y = compute_features(labels, embeddings, pairs)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    # Save model
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(f"[agent_classifier] saved trained model to {args.model_out}")

if __name__ == "__main__":
    main() 