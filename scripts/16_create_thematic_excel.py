#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Automatically group taxonomy into thematic sheets in Excel.

"""16_create_thematic_excel.py — Automatically identify major themes in taxonomy and generate Excel with one sheet per theme."""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict
from tqdm import tqdm
import unicodedata
from text_unidecode import unidecode

def normalize(text: str) -> str:
    """Normalize text to Unicode NFKC, strip accents, and lowercase."""
    t = unicodedata.normalize("NFKC", text)
    t = unidecode(t)
    return t.lower()

def get_top_keywords(data, clusters, labels, n_terms=5):
    """Extract most representative words for each cluster."""
    # This is a simple approach - could be enhanced with TF-IDF
    topics = {}
    for i in range(len(np.unique(clusters))):
        tokens = []
        # Get all words from the cluster
        for j in np.where(clusters == i)[0]:
            tokens.extend(normalize(labels[j]).split())
        
        # Count word frequencies
        word_counts = defaultdict(int)
        for word in tokens:
            if len(word) > 3:  # Skip short words
                word_counts[word] += 1
        
        # Get top words by frequency
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:n_terms]
        topics[i] = [word for word, _ in top_words]
    
    return topics

def name_cluster(top_words, labels_in_cluster):
    """Generate a meaningful name for the cluster based on keywords and labels."""
    # Start with the most frequent words
    theme_name = " / ".join(top_words[:2])
    
    # If we have very few words, add the most representative label
    if len(top_words) < 2:
        # Find the most central label in the cluster
        if labels_in_cluster:
            theme_name = labels_in_cluster[0]
    
    return theme_name.capitalize()

def main():
    parser = argparse.ArgumentParser(description="Create thematic Excel workbook from taxonomy.")
    parser.add_argument(
        "--taxo", default="data/taxo_raw.csv",
        help="Path to the taxonomy CSV (raw labels)."
    )
    parser.add_argument(
        "--emb", default="data/embeddings.npy",
        help="Path to embeddings numpy file."
    )
    parser.add_argument(
        "--n-themes", type=int, default=12,
        help="Number of themes/clusters to identify (default: 12)"
    )
    parser.add_argument(
        "--output", default="data/thematic_taxonomy.xlsx",
        help="Path to output Excel workbook."
    )
    args = parser.parse_args()

    # Load taxonomy and embeddings
    print(f"[thematic_excel] Loading taxonomy from {args.taxo}")
    df_taxo = pd.read_csv(args.taxo, header=None, dtype=str)
    labels = df_taxo.iloc[:, 0].tolist()
    
    print(f"[thematic_excel] Loading embeddings from {args.emb}")
    embeddings = np.load(args.emb)
    
    # Ensure we have matching dimensions
    if len(labels) != embeddings.shape[0]:
        raise ValueError(f"Mismatch between number of labels ({len(labels)}) and embeddings ({embeddings.shape[0]})")

    # Perform clustering
    print(f"[thematic_excel] Clustering into {args.n_themes} themes...")
    kmeans = KMeans(n_clusters=args.n_themes, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Identify keywords for each cluster
    print("[thematic_excel] Extracting representative keywords for themes...")
    topics = get_top_keywords(embeddings, clusters, labels)
    
    # Create theme names
    theme_names = {}
    cluster_labels = {i: [] for i in range(args.n_themes)}
    
    # Collect labels per cluster
    for i, cluster_id in enumerate(clusters):
        cluster_labels[cluster_id].append(labels[i])
    
    # Create meaningful names for each cluster
    for cluster_id, top_words in topics.items():
        theme_names[cluster_id] = name_cluster(top_words, cluster_labels[cluster_id][:10])
    
    # Create a mapping from original to normalized forms for exact matching of duplicates
    norm_to_orig = {}
    for label in labels:
        norm = normalize(label)
        if norm not in norm_to_orig:
            norm_to_orig[norm] = []
        norm_to_orig[norm].append(label)
    
    # Create dataframes for each theme
    theme_dfs = {}
    for cluster_id in range(args.n_themes):
        # Get indices of all items in this cluster
        indices = np.where(clusters == cluster_id)[0]
        
        # Create dataframe with these items
        theme_df = pd.DataFrame({
            'label_id': indices,
            'label': [labels[i] for i in indices],
            'theme': theme_names[cluster_id]
        })
        
        # Sort by label for better readability
        theme_df = theme_df.sort_values('label')
        
        theme_dfs[theme_names[cluster_id]] = theme_df
    
    # Write to Excel workbook
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[thematic_excel] Writing Excel workbook to {output_path}...")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Add a summary sheet
        summary_rows = []
        for theme, df in theme_dfs.items():
            summary_rows.append({
                'Theme': theme,
                'Count': len(df),
                'Examples': ', '.join(df['label'].head(3).tolist())
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Add each theme as a sheet
        for theme, df in theme_dfs.items():
            # Make sheet name Excel-compatible (max 31 chars, no special chars)
            sheet_name = "".join(c for c in theme[:31] if c.isalnum() or c in " _").strip()
            if not sheet_name:
                sheet_name = f"Theme_{list(theme_dfs.keys()).index(theme)}"
            
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"[thematic_excel] Successfully wrote {args.n_themes} thematic sheets to {output_path}")

if __name__ == '__main__':
    main() 