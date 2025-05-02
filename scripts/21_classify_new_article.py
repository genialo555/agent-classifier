#!/usr/bin/env python3
# AI-Assisted (2025-05-02): Classify new articles automatically into themes and veille types.

"""21_classify_new_article.py — Classify new articles automatically into the most appropriate theme and type de veille."""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import joblib
from text_unidecode import unidecode
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def normalize(text: str) -> str:
    """Normalize text to Unicode NFKC, strip accents, and lowercase."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = unidecode(t)
    return t.lower()

def jw_sim(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity with fallbacks."""
    # Try rapidfuzz
    try:
        from rapidfuzz.distance import JaroWinkler as _JW
        return _JW.normalized_similarity(s1, s2)
    except ImportError:
        pass
    # Try jaro_winkler package
    try:
        from jaro_winkler import jaro_winkler_similarity as jw
        return jw(s1, s2)
    except ImportError:
        pass
    # Try jellyfish
    try:
        from jellyfish import jaro_winkler_similarity as jw
        return jw(s1, s2)
    except ImportError:
        # Last resort: character overlap ratio
        set1, set2 = set(s1), set(s2)
        return len(set1 & set2) / max(len(set1 | set2), 1)

def get_article_text(input_path=None, text=None):
    """Get article text from file or direct input."""
    if input_path:
        with open(input_path, 'r', encoding='utf-8') as f:
            return f.read()
    return text

def extract_keywords(text, top_n=15):
    """Extract important keywords from the text."""
    # Simple implementation based on frequency
    words = normalize(text).split()
    # Filter out short words and common stop words
    stop_words = set(['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'en', 'de', 'du', 'au', 'aux', 
                      'ce', 'ces', 'cette', 'sur', 'pour', 'par', 'dans', 'que', 'qui', 'avec', 'est',
                      'sont', 'ont', 'sera', 'plus', 'leur', 'leurs', 'tous', 'tout', 'toute', 'toutes',
                      'a', 'à', 'au', 'aux', 'se', 'si', 'son', 'sa', 'ses', 'je', 'tu', 'il', 'elle',
                      'nous', 'vous', 'ils', 'elles', 'non', 'oui', 'mais', 'car', 'donc', 'or', 'ni',
                      'ne', 'pas', 'aucun', 'aucune', 'autre', 'autres', 'même', 'mêmes'])
    filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
    
    # Count word frequency
    word_counts = {}
    for word in filtered_words:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
    
    # Get top N most frequent words
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_n]]

def extract_key_phrases(text, max_phrases=5, max_words=10):
    """Extract key phrases from the text to help with classification."""
    # Split into sentences and normalize
    import re
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    # Score sentences based on keyword presence
    keywords = extract_keywords(text)
    scores = []
    
    for s in sentences:
        # Count keywords in sentence
        normalized = normalize(s)
        keyword_count = sum(1 for kw in keywords if kw in normalized.split())
        # Score is keyword count divided by sentence length (in words)
        score = keyword_count / (len(normalized.split()) + 1)  # +1 to avoid division by zero
        scores.append((s, score))
    
    # Get top scored sentences
    top_sentences = sorted(scores, key=lambda x: x[1], reverse=True)[:max_phrases]
    
    # Truncate sentences to max_words if needed
    result = []
    for sent, _ in top_sentences:
        words = sent.split()
        if len(words) > max_words:
            result.append(' '.join(words[:max_words]) + '...')
        else:
            result.append(sent)
    
    return result

def plot_classification_results(theme_scores, veille_scores, output_path=None):
    """Create visualizations of classification results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot theme scores
    theme_df = pd.DataFrame(theme_scores).sort_values('score', ascending=False).head(5)
    bars1 = ax1.barh(theme_df['theme'], theme_df['score'], color='skyblue')
    ax1.set_xlabel('Score de similarité')
    ax1.set_ylabel('Thématique')
    ax1.set_title('Top 5 thématiques correspondantes')
    
    # Add score text
    for bar in bars1:
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                va='center', fontsize=9)
    
    # Plot veille type scores
    veille_df = pd.DataFrame(veille_scores).sort_values('score', ascending=False).head(5)
    bars2 = ax2.barh(veille_df['veille_type'], veille_df['score'], color='lightgreen')
    ax2.set_xlabel('Score de similarité')
    ax2.set_ylabel('Type de veille')
    ax2.set_title('Top 5 types de veille correspondants')
    
    # Add score text
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Classify a new article into themes and veille types.")
    parser.add_argument(
        "--input", 
        help="Path to article text file to classify."
    )
    parser.add_argument(
        "--text", 
        help="Direct article text to classify (alternative to --input)."
    )
    parser.add_argument(
        "--thematic-excel", default="data/thematic_taxonomy_custom.xlsx",
        help="Path to thematic Excel for theme classification."
    )
    parser.add_argument(
        "--veille-excel", default="data/veille_types.xlsx",
        help="Path to veille types Excel for type de veille classification."
    )
    parser.add_argument(
        "--emb-model", default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for embeddings."
    )
    parser.add_argument(
        "--classifier", default="models/agent_classifier.joblib",
        help="Path to trained classifier model."
    )
    parser.add_argument(
        "--output", default="data/article_classification.xlsx",
        help="Path to output Excel with classification results."
    )
    parser.add_argument(
        "--viz", default="data/article_classification.png",
        help="Path to output visualization of classification results."
    )
    args = parser.parse_args()

    if not args.input and not args.text:
        parser.error("Either --input or --text must be provided")
    
    # Get article text
    article_text = get_article_text(args.input, args.text)
    
    if not article_text:
        print("[classify_article] Error: Empty article text")
        return
    
    print(f"[classify_article] Article length: {len(article_text)} characters")
    
    # Extract keywords and key phrases
    keywords = extract_keywords(article_text)
    key_phrases = extract_key_phrases(article_text)
    
    print("[classify_article] Keywords:", ", ".join(keywords))
    print("[classify_article] Key phrases:")
    for i, phrase in enumerate(key_phrases, 1):
        print(f"  {i}. {phrase}")
    
    # Load models
    print(f"[classify_article] Loading embedding model: {args.emb_model}")
    model = SentenceTransformer(args.emb_model)
    
    print(f"[classify_article] Loading classifier: {args.classifier}")
    clf = joblib.load(args.classifier) if Path(args.classifier).exists() else None
    
    # Embed article - both full text and key components
    print("[classify_article] Embedding article...")
    article_full_embedding = model.encode([article_text])[0]
    keywords_embedding = model.encode([" ".join(keywords)])[0]
    phrases_embedding = model.encode([" ".join(key_phrases)])[0]
    
    # Create a composite embedding (weighted average)
    article_embedding = 0.4 * article_full_embedding + 0.3 * keywords_embedding + 0.3 * phrases_embedding
    
    # Load thematic data
    print(f"[classify_article] Loading thematic data from {args.thematic_excel}")
    xls_theme = pd.ExcelFile(args.thematic_excel)
    
    # Process each theme sheet
    theme_scores = []
    
    print("[classify_article] Analyzing themes...")
    for sheet in tqdm(xls_theme.sheet_names):
        if sheet == 'Summary':
            continue
            
        # Load this theme's data
        theme_df = pd.read_excel(xls_theme, sheet)
        
        if len(theme_df) == 0:
            continue
            
        # Get all entries
        all_entries = theme_df['label'].astype(str).tolist()
        
        # Embed all entries for accurate comparison
        # But limit to max 50 entries to prevent memory issues
        sample_size = min(50, len(all_entries))
        sample_entries = all_entries[:sample_size]
        sample_embeddings = model.encode(sample_entries)
        
        # Compute similarities
        similarities = cosine_similarity([article_embedding], sample_embeddings)[0]
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        # Find top 3 most similar entries for this theme
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_entries = [sample_entries[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        # Compute lexical similarity between keywords and theme name
        theme_kw_sim = jw_sim(" ".join(keywords), normalize(sheet))
        
        # Combined score (weighted) - prioritizing content similarity
        combined_score = 0.15 * avg_similarity + 0.65 * max_similarity + 0.2 * theme_kw_sim
        
        theme_scores.append({
            'theme': sheet,
            'score': combined_score,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'top_entries': ", ".join(top_entries),
            'top_scores': ", ".join([f"{s:.3f}" for s in top_scores]),
            'keyword_similarity': theme_kw_sim
        })
    
    # Sort and identify best theme
    theme_scores = sorted(theme_scores, key=lambda x: x['score'], reverse=True)
    best_theme = theme_scores[0]['theme'] if theme_scores else None
    
    # Load veille types data
    print(f"[classify_article] Loading veille types data from {args.veille_excel}")
    xls_veille = pd.ExcelFile(args.veille_excel)
    
    # Process each veille type sheet
    veille_scores = []
    
    print("[classify_article] Analyzing veille types...")
    # First get all sheet names
    sheets = [sheet for sheet in xls_veille.sheet_names if sheet != 'Summary']
    
    # Load summary to get type count info
    summary_df = pd.read_excel(xls_veille, 'Summary')
    
    # Process each veille type
    for sheet in tqdm(sheets):
        # Load this veille type's data
        veille_df = pd.read_excel(xls_veille, sheet)
        
        if len(veille_df) == 0:
            continue
            
        # Get original veille type name from summary
        sheet_info = summary_df[summary_df['Type'].str.contains(sheet, regex=False, na=False)]
        if not sheet_info.empty:
            veille_type = sheet_info['Type'].iloc[0]
        else:
            veille_type = sheet
        
        # Sample entries from different columns for more representative content
        sample_entries = []
        
        # Try to find columns with text content 
        text_cols = []
        for col in veille_df.columns:
            # Skip columns that are likely IDs or metadata
            if any(skip in normalize(col) for skip in ['id', 'code', 'date', 'typedeveille']):
                continue
            
            # Check if this column has text data
            col_sample = veille_df[col].astype(str).iloc[0]
            if len(col_sample) > 5:  # Simple heuristic for text data
                text_cols.append(col)
        
        # If we found text columns, sample from each
        if text_cols:
            for col in text_cols[:2]:  # Use at most 2 columns
                samples = veille_df[col].astype(str).head(15).tolist()
                sample_entries.extend(samples)
        
        # Fallback: use any column if no text columns identified
        if not sample_entries and len(veille_df.columns) > 1:
            col = veille_df.columns[1]  # Second column, assuming first might be an ID
            sample_entries = veille_df[col].astype(str).head(30).tolist()
        
        # Ensure we have entries
        if not sample_entries:
            continue
            
        # Filter out very short entries
        sample_entries = [entry for entry in sample_entries if len(entry) > 5]
        
        if not sample_entries:
            continue
        
        # Embed sample entries
        sample_embeddings = model.encode(sample_entries)
        
        # Compute similarities
        similarities = cosine_similarity([article_embedding], sample_embeddings)[0]
        avg_similarity = np.mean(similarities)
        max_similarity = np.max(similarities)
        
        # Find top 3 most similar entries for this veille type
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_entries = [sample_entries[i] for i in top_indices]
        top_scores = [similarities[i] for i in top_indices]
        
        # Compute keyword similarity
        veille_kw_sim = jw_sim(" ".join(keywords), normalize(veille_type))
        
        # Get the weight of this category (based on entry count)
        sheet_count = int(sheet_info['Count'].iloc[0]) if not sheet_info.empty else 0
        weight = min(1.0, sheet_count / 1000) * 0.1  # Small adjustment based on category size
        
        # Combined score (weighted) - prioritizing content similarity
        combined_score = (0.15 * avg_similarity + 0.65 * max_similarity + 0.2 * veille_kw_sim) + weight
        
        veille_scores.append({
            'veille_type': veille_type,
            'score': combined_score,
            'avg_similarity': avg_similarity,
            'max_similarity': max_similarity,
            'top_entries': ", ".join(top_entries[:3]),
            'top_scores': ", ".join([f"{s:.3f}" for s in top_scores[:3]]),
            'keyword_similarity': veille_kw_sim
        })
    
    # Sort and identify best veille type
    veille_scores = sorted(veille_scores, key=lambda x: x['score'], reverse=True)
    best_veille = veille_scores[0]['veille_type'] if veille_scores else None
    
    # Print results
    print("\n=== CLASSIFICATION RESULTS ===")
    print(f"Best Theme: {best_theme} (Score: {theme_scores[0]['score']:.3f})" if best_theme else "No theme found")
    if best_theme and len(theme_scores) > 1:
        print(f"Second Best Theme: {theme_scores[1]['theme']} (Score: {theme_scores[1]['score']:.3f})")
    
    print(f"Best Type de Veille: {best_veille} (Score: {veille_scores[0]['score']:.3f})" if best_veille else "No veille type found")
    if best_veille and len(veille_scores) > 1:
        print(f"Second Best Type de Veille: {veille_scores[1]['veille_type']} (Score: {veille_scores[1]['score']:.3f})")
    
    # Create results dataframe
    results = {
        'article_text': [article_text[:500] + "..." if len(article_text) > 500 else article_text],
        'keywords': [", ".join(keywords)],
        'key_phrases': [" | ".join(key_phrases)],
        'best_theme': [best_theme],
        'theme_score': [theme_scores[0]['score'] if theme_scores else None],
        'best_veille_type': [best_veille],
        'veille_score': [veille_scores[0]['score'] if veille_scores else None]
    }
    
    # Save results to Excel
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main results
        pd.DataFrame(results).to_excel(writer, sheet_name='Results', index=False)
        
        # Theme scores
        pd.DataFrame(theme_scores).to_excel(writer, sheet_name='Theme_Scores', index=False)
        
        # Veille scores
        pd.DataFrame(veille_scores).to_excel(writer, sheet_name='Veille_Scores', index=False)
    
    print(f"[classify_article] Classification results saved to {output_path}")
    
    # Plot results
    plot_classification_results(theme_scores, veille_scores, args.viz)
    print(f"[classify_article] Visualization saved to {args.viz}")


if __name__ == '__main__':
    main() 