#!/usr/bin/env python3
# AI-Assisted (2025-05-02): Group taxonomy entries by their "type de veille" and generate Excel with one sheet per type.

"""19_group_by_veille_type.py — Group entries by their type de veille and create Excel workbook with one sheet per type."""
import argparse
import pandas as pd
from pathlib import Path
import unicodedata
from text_unidecode import unidecode
from openpyxl.utils import get_column_letter

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
    
    # Identify domain and subdomain columns
    col_names = df.columns.tolist()
    domain_col = next((col for col in col_names if normalize(col) == 'domaine'), None)
    subdomain_col = next((col for col in col_names if normalize(col) == 'sousdomaine'), None)
    if not domain_col or not subdomain_col:
        raise ValueError("Columns 'DOMAINE' or 'SOUSDOMAINE' not found in input file.")
    
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
    
    # Define QSE terms
    qse_terms = {"qualite", "securite", "environnement"}

    print(f"[group_by_veille] Writing Excel workbook to {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Create summary sheet first
        summary_rows = []
        for type_name in unique_types:
            if not type_name or str(type_name).strip() == '':
                continue
            group_df = df[df['type_normalized'] == type_name]
            type_display = group_df[veille_col].iloc[0]
            # Determine if any domain or subdomain matches QSE terms
            all_domains = group_df[domain_col].fillna('').apply(normalize)
            all_subs = group_df[subdomain_col].fillna('').apply(normalize)
            is_qse = any(any(term in text for term in qse_terms) for text in pd.concat([all_domains, all_subs]))

            summary_rows.append({
                'Type': type_display,
                'Count': len(group_df),
                'QSE': 'Yes' if is_qse else 'No'
            })

        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values('Count', ascending=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Create a sheet for each type de veille listing domains and subdomains
        for type_name in unique_types:
            if not type_name or str(type_name).strip() == '':
                continue
            group_df = df[df['type_normalized'] == type_name]
            type_display = group_df[veille_col].iloc[0]
            # Extract unique domain/subdomain combos
            dom_sub_df = group_df[[domain_col, subdomain_col]].drop_duplicates().fillna('')
            # Determine sheet-level QSE flag: if any domaine or sous-domaine contains QSE terms
            sheet_is_qse = any(
                any(term in normalize(val) for term in qse_terms)
                for val in list(dom_sub_df[domain_col]) + list(dom_sub_df[subdomain_col])
            )
            # Apply QSE flag uniformly per sheet
            dom_sub_df['QSE'] = 'Yes' if sheet_is_qse else 'No'
            sheet_df = dom_sub_df
            # Make sheet name Excel-compatible
            sheet_name = "".join(c for c in type_display[:31] if c.isalnum() or c in " _").strip()
            if not sheet_name:
                sheet_name = f"Type_{unique_types.index(type_name)}"
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            # Make subdomain columns accessible: apply auto-filter and freeze row
            ws = writer.sheets[sheet_name]
            max_row, max_col = sheet_df.shape
            col_letter = get_column_letter(max_col)
            # Set filter over all columns
            ws.auto_filter.ref = f"A1:{col_letter}{max_row+1}"
            # Freeze the header row
            ws.freeze_panes = "A2"
    print(f"[group_by_veille] Successfully wrote {len(unique_types)} type de veille sheets to {output_path}")

if __name__ == '__main__':
    main() 