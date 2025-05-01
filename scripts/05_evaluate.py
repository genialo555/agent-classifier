#!/usr/bin/env python3
# AI-Assisted (2025-04-17): evaluation script for merge pipeline.
# Review focus: metrics reporting.

"""05_evaluate.py — Evaluate merge results."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Evaluate merge results.")
    parser.add_argument("--taxo", default="data/taxo_raw.csv", help="Normalized taxonomy CSV.")
    parser.add_argument("--merges", default="data/merge_pairs.csv", help="Merge pairs CSV.")
    args = parser.parse_args()

    # Load data
    df_taxo = pd.read_csv(args.taxo, header=None)
    df_merges = pd.read_csv(args.merges)

    # Compute metrics: total pairs vs unique source merges
    original_count = len(df_taxo)
    pair_count = len(df_merges)
    unique_sources = df_merges['source_id'].nunique()
    aligned_count = original_count - unique_sources
    reduction = unique_sources

    # Print summary
    print(f"[evaluate] original labels: {original_count}")
    print(f"[evaluate] merge pairs: {pair_count}")
    print(f"[evaluate] unique sources merged: {unique_sources}")
    print(f"[evaluate] aligned labels: {aligned_count}")
    print(f"[evaluate] reduction: {reduction} labels ({reduction/original_count*100:.2f}%)")


if __name__ == "__main__":
    main() 