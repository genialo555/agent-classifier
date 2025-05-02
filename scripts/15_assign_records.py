#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Assign each record title to taxonomy labels using the agent classifier and embeddings.

"""15_assign_records.py — For each record title, compute features against taxonomy labels and assign the best match using the trained classifier. Export results to Excel."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import unicodedata
from text_unidecode import unidecode
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)
    return text.lower()


def jw_sim(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity with fallbacks."""
    # 1. Try rapidfuzz
    try:
        from rapidfuzz.distance import JaroWinkler as _JW
        return _JW.normalized_similarity(s1, s2)
    except ImportError:
        pass
    # 2. Try jaro_winkler package
    try:
        from jaro_winkler import jaro_winkler_similarity as jw
        return jw(s1, s2)
    except ImportError:
        pass
    # 3. Try jellyfish
    try:
        from jellyfish import jaro_winkler_similarity as jw
        return jw(s1, s2)
    except ImportError:
        # 4. Last resort: character overlap ratio
        set1, set2 = set(s1), set(s2)
        return len(set1 & set2) / max(len(set1 | set2), 1)


# Add utility to sniff delimiter
def sniff_sep(file_path: str) -> str:
    """Detect delimiter by comparing tab and comma counts in the header line."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
    return '\t' if header.count('\t') > header.count(',') else ','


def main():
    parser = argparse.ArgumentParser(description="Assign each record to a taxonomy label.")
    parser.add_argument("--records", default="data/sorted_data.csv", help="Path to CSV of records with titles.")
    parser.add_argument("--title-col", default=0, help="Index or name of the title column in records CSV.")
    parser.add_argument("--taxo", default="data/taxo_raw.csv", help="Path to taxonomy CSV (labels).")
    parser.add_argument("--taxo-emb", default="data/embeddings.npy", help="Path to taxonomy embeddings numpy file.")
    parser.add_argument("--model", default="models/agent_classifier.joblib", help="Path to trained classifier model.")
    parser.add_argument("--emb-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model for record embeddings.")
    parser.add_argument("--output", default="data/record_classification.xlsx", help="Output Excel file path.")
    args = parser.parse_args()

    # Detect delimiter and load records
    sep = sniff_sep(args.records)
    df_rec = pd.read_csv(args.records, sep=sep, dtype=str, engine='python')
    # Determine title series
    if isinstance(args.title_col, str) and args.title_col in df_rec.columns:
        titles = df_rec[args.title_col].astype(str).tolist()
    else:
        idx = int(args.title_col)
        titles = df_rec.iloc[:, idx].astype(str).tolist()

    # Load taxonomy labels and embeddings
    df_taxo = pd.read_csv(args.taxo, header=None, dtype=str)
    taxo_labels = df_taxo.iloc[:, 0].tolist()
    taxo_embs = np.load(args.taxo_emb)

    # Normalize taxonomy embeddings
    norms = np.linalg.norm(taxo_embs, axis=1, keepdims=True)
    taxo_norm = taxo_embs / np.clip(norms, a_min=1e-12, a_max=None)

    # Load classifier
    clf = joblib.load(args.model)

    # Compute record embeddings
    model = SentenceTransformer(args.emb_model)
    rec_embs = model.encode(titles, show_progress_bar=True, convert_to_numpy=True)
    rec_norm = rec_embs / np.clip(np.linalg.norm(rec_embs, axis=1, keepdims=True), a_min=1e-12, a_max=None)

    # Iterate and assign
    results = []
    for i, title in enumerate(tqdm(titles, desc="Classifying records")):
        best_prob = -1.0
        best_j = None
        lex_norm = normalize(title)
        for j, lab in enumerate(taxo_labels):
            lex_sim = jw_sim(title, lab)
            cos_sim = float(np.dot(rec_norm[i], taxo_norm[j]))
            prob = clf.predict_proba([[lex_sim, cos_sim]])[0][1]
            if prob > best_prob:
                best_prob = prob
                best_j = j
        # Append result
        results.append({
            **df_rec.iloc[i].to_dict(),
            'assigned_taxo_id': best_j,
            'assigned_taxo_label': taxo_labels[best_j],
            'merge_probability': best_prob
        })

    # Create DataFrame
    df_out = pd.DataFrame(results)

    # Export to Excel
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df_out.to_excel(writer, sheet_name='RecordClassification', index=False)

    print(f"[assign_records] wrote classification for {len(df_out)} records to {out_path}")

if __name__ == '__main__':
    main() 