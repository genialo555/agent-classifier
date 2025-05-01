#!/usr/bin/env python3
# AI-Assisted (2025-04-17): export script for merge pipeline.
# Review focus: applying merges to normalized taxonomy.

"""06_export.py — Export the aligned master taxonomy after merging."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Export aligned taxonomy.")
    parser.add_argument("--taxo", default="data/taxo_raw.csv", help="Path to normalized taxonomy CSV.")
    parser.add_argument("--merges", default="data/merge_pairs.csv", help="Path to merge pairs CSV.")
    parser.add_argument("--output", default="data/taxonomy_aligned.csv", help="Where to write aligned taxonomy.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.taxo, header=None)
    df_merges = pd.read_csv(args.merges)

    # Identify merged source rows
    merged_sources = set(df_merges['source_id'].tolist())

    # Apply merges by dropping merged rows
    print(f"[export] original rows: {len(df)}")
    print(f"[export] dropping {len(merged_sources)} merged rows")
    df_aligned = df.drop(index=merged_sources).reset_index(drop=True)
    print(f"[export] aligned taxonomy rows: {len(df_aligned)}")

    # Save aligned taxonomy
    df_aligned.to_csv(args.output, index=False, header=False)
    print(f"[export] wrote aligned taxonomy to {args.output}")


if __name__ == "__main__":
    main() 