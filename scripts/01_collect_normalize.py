# AI-Assisted (2025-04-18): minimal stub for data collection & normalisation. Review focus: CLI, Unicode handling.

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
from text_unidecode import unidecode
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Return NFKC-normalized, accent-stripped, lower-cased version of *text*."""
    if not isinstance(text, str):
        return text  # leave e.g. NaN untouched
    text = unicodedata.normalize("NFKC", text)
    text = unidecode(text)  # strip accents
    return text.lower()


def collect_csv(sources) -> pd.DataFrame:
    frames = []
    for src in tqdm(sources, desc="collecting sources", unit="file"):
        # Sniff delimiter from header line: prefer tab if more tabs than commas
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            header_line = f.readline()
        sep = "\t" if header_line.count("\t") > header_line.count(",") else ","
        df = pd.read_csv(src, sep=sep, header=None, dtype=str, keep_default_na=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect & normalise raw taxonomy files.")
    parser.add_argument("--sources", nargs="*", required=True, help="Path pattern(s) to source CSV/TSV files.")
    parser.add_argument("--output", default="data/taxo_raw.csv", help="Where to write the aggregated file.")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing file (for CI speed).")
    args = parser.parse_args()

    # enable pandas progress_apply with tqdm
    tqdm.pandas()

    df = collect_csv(args.sources)

    # Apply normalization column-wise with progress bar
    df = df.progress_applymap(normalize_text)

    if not args.dry_run:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False, header=False)
        print(f"[collect_normalize] wrote {args.output} (rows={len(df)})")
    else:
        print(df.head())


if __name__ == "__main__":
    main() 