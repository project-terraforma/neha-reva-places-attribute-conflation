#!/usr/bin/env python3
"""
Preview tool: load parquet, adapt rows, write JSONL.

Usage:
  python -m agentic_approach.validate --input data/project_a_samples.parquet --limit 20 --out out/rows_preview.jsonl

From project root.
"""

import argparse
import json
from pathlib import Path

from .io import load_parquet, iter_rows
from .adapter import adapt_row


def main():
    parser = argparse.ArgumentParser(
        description="Load parquet, adapt rows to canonical format, write JSONL preview."
    )
    parser.add_argument(
        "--input",
        default="data/project_a_samples.parquet",
        help="Input parquet path (default: data/project_a_samples.parquet)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max rows to output (default: 20)",
    )
    parser.add_argument(
        "--out",
        default="out/rows_preview.jsonl",
        help="Output JSONL path (default: out/rows_preview.jsonl)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_parquet(input_path)
    adapted = []
    for raw_row in iter_rows(df, limit=args.limit):
        adapted.append(adapt_row(raw_row))

    with open(out_path, "w", encoding="utf-8") as f:
        for row in adapted:
            f.write(json.dumps(row, default=str) + "\n")

    print(f"Wrote {len(adapted)} adapted rows to {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
