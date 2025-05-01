#!/usr/bin/env python3
# AI-Assisted (2025-04-18): sort the original data.csv by GPT enrichment match flag.

"""08_sort_data.py — Sort the original data.csv by the `isMatch` field in the enrichment JSON."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def sniff_sep(file_path: str) -> str:
    """Detect delimiter by comparing tab and comma counts in the header line."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def extract_is_match(x: str) -> bool:
    """Parse the JSON or fallback regex to extract the isMatch boolean."""
    if not isinstance(x, str):
        return False
    # Try JSON parse
    try:
        d = json.loads(x)
        return bool(d.get("isMatch", False))
    except Exception:
        # Fallback regex
        m = re.search(r'isMatch\s*:\s*(true|false)', x, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"
        return False


def main():
    parser = argparse.ArgumentParser(description="Sort data.csv by enrichment isMatch flag.")
    parser.add_argument("--input", default="data.csv", help="Original data CSV with enrichment column.")
    parser.add_argument("--output", default="data/sorted_data.csv", help="Path to write sorted CSV.")
    args = parser.parse_args()

    sep = sniff_sep(args.input)
    # Read with Python engine to respect custom delimiter
    df = pd.read_csv(args.input, sep=sep, quotechar='"', engine='python')

    # Identify enrichment column (last column)
    enrichment_col = df.columns[-1]

    tqdm.pandas(desc="Extracting isMatch")
    df['isMatch'] = df[enrichment_col].progress_apply(extract_is_match)

    # Sort by isMatch descending
    df_sorted = df.sort_values(by='isMatch', ascending=False).reset_index(drop=True)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_sorted.to_csv(args.output, sep=sep, index=False)
    print(f"[sort_data] wrote sorted data to {args.output}")


if __name__ == "__main__":
    main() 