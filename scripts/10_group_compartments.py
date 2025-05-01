#!/usr/bin/env python3
# AI-Assisted (2025-04-18): Group aligned taxonomy entries into compartments (archivist role).

"""10_group_compartments.py — After taxonomy alignment, group taxonomy entries into compartments based on merge pairs."""
import argparse
import pandas as pd
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Group taxonomy entries into compartments based on merge pairs.")
    parser.add_argument(
        "--taxo", default="data/taxo_raw.csv",
        help="Path to the normalized taxonomy CSV (raw labels)."
    )
    parser.add_argument(
        "--merges", default="data/merge_pairs.csv",
        help="Path to the merge pairs CSV."
    )
    parser.add_argument(
        "--output", default="data/compartments.csv",
        help="Path to write the compartments CSV."
    )
    args = parser.parse_args()

    # Load raw taxonomy labels
    df_taxo = pd.read_csv(args.taxo, header=None)
    labels = df_taxo.iloc[:, 0].astype(str).tolist()

    # Load merge pairs (source_id -> target_id)
    df_merges = pd.read_csv(args.merges)

    # Build grouping: map each target to its list of sources
    groups = defaultdict(list)
    for row in df_merges.itertuples(index=False):
        src, tgt = int(row.source_id), int(row.target_id)
        groups[tgt].append(src)

    # Prepare rows: each compartment lists canonical label and its members
    rows = []
    for tgt, srcs in sorted(groups.items(), key=lambda x: x[0]):
        # include the canonical itself
        member_ids = [tgt] + srcs
        for mid in sorted(member_ids):
            rows.append({
                'canonical_id': tgt,
                'canonical_label': labels[tgt],
                'member_id': mid,
                'member_label': labels[mid],
            })

    # Export to CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.output, index=False)
    print(f"[group_compartments] wrote compartments to {args.output}")


if __name__ == '__main__':
    main() 