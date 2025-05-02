#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Export grouping CSVs to a single Excel workbook.

"""14_export_to_excel.py — Combine compartments and classifier groups into an Excel workbook."""
import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Export groupings to Excel workbook.")
    parser.add_argument(
        "--compartments", default="data/compartments.csv",
        help="Path to the compartments CSV file."
    )
    parser.add_argument(
        "--classifier_groups", default="data/classifier_all_groups.csv",
        help="Path to the classifier all groups CSV file."
    )
    parser.add_argument(
        "--output", default="data/taxonomy_groups.xlsx",
        help="Path to the output Excel workbook."
    )
    args = parser.parse_args()

    # Load data
    df_comp = pd.read_csv(args.compartments)
    df_cls = pd.read_csv(args.classifier_groups)

    # Write to Excel workbook
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_comp.to_excel(writer, sheet_name='Compartments', index=False)
        df_cls.to_excel(writer, sheet_name='ClassifierGroups', index=False)

    print(f"[export_to_excel] wrote Excel workbook to {output_path}")

if __name__ == '__main__':
    main() 