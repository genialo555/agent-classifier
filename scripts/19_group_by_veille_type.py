#!/usr/bin/env python3
# AI-Assisted (2025-05-02): Group taxonomy entries by their "type de veille" and generate Excel with one sheet per type.

"""19_group_by_veille_type.py — Group entries by their type de veille and create Excel workbook with one sheet per type."""
import argparse
import pandas as pd
from pathlib import Path
import unicodedata
from text_unidecode import unidecode

def normalize(text: str) -> str:
    """Normalize text to Unicode NFKC, strip accents, and lowercase."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = unidecode(t)
    return t.lower()

def sniff_sep(file_path: str) -> str:
    """Detect delimiter by comparing tab and comma counts in the header line."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
    return '\t' if header.count('\t') > header.count(',') else ','

def main():
    parser = argparse.ArgumentParser(description="Group entries by type de veille into Excel workbook.")
    parser.add_argument(
        "--input", default="data.csv",
        help="Path to the original CSV with 'typedeveille' column."
    )
    parser.add_argument(
        "--output", default="data/veille_types.xlsx",
        help="Path to output Excel workbook."
    )
    args = parser.parse_args()

    # Load original data with type de veille column
    print(f"[group_by_veille] Loading data from {args.input}")
    separator = sniff_sep(args.input)
    df = pd.read_csv(args.input, sep=separator, dtype=str, engine='python')
    
    # Check if "typedeveille" column exists
    col_names = df.columns.tolist()
    veille_col = None
    for col in col_names:
        if normalize(col) == "typedeveille":
            veille_col = col
            break
    
    if not veille_col:
        raise ValueError("Column 'typedeveille' not found in input file.")
    
    # Group by type de veille
    print(f"[group_by_veille] Grouping entries by {veille_col}")
    
    # Clean and normalize type de veille values
    df['type_normalized'] = df[veille_col].fillna("unknown").apply(normalize)
    
    # Get unique types
    unique_types = df['type_normalized'].dropna().unique().tolist()
    print(f"[group_by_veille] Found {len(unique_types)} unique types de veille")
    
    # Prepare the Excel workbook
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[group_by_veille] Writing Excel workbook to {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Create summary sheet first
        summary_rows = []
        for type_name in unique_types:
            # Skip empty types
            if not type_name or str(type_name).strip() == '':
                continue
                
            group_df = df[df['type_normalized'] == type_name]
            type_display = df.loc[df['type_normalized'] == type_name, veille_col].iloc[0]
            
            summary_rows.append({
                'Type': type_display,
                'Count': len(group_df),
                'Examples': ', '.join(group_df.iloc[:3, 1].astype(str).tolist())  # Use second column as examples
            })
            
        # Create and write summary DataFrame
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('Count', ascending=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Create a sheet for each type de veille
        for type_name in unique_types:
            # Skip empty types
            if not type_name or str(type_name).strip() == '':
                continue
                
            # Get display name from original data
            type_display = df.loc[df['type_normalized'] == type_name, veille_col].iloc[0]
            
            # Filter data for this type
            group_df = df[df['type_normalized'] == type_name].drop(columns=['type_normalized'])
            
            # Make sheet name Excel-compatible (max 31 chars, no special chars)
            sheet_name = "".join(c for c in type_display[:31] if c.isalnum() or c in " _").strip()
            if not sheet_name:
                sheet_name = f"Type_{unique_types.index(type_name)}"
                
            # Write sheet
            group_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"[group_by_veille] Successfully wrote {len(unique_types)} type de veille sheets to {output_path}")

if __name__ == '__main__':
    main() 