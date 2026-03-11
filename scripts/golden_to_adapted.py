#!/usr/bin/env python3
"""
Convert golden_dataset.json to rows_adapted.jsonl format for flow input.

Golden JSON is created via scripts/create_golden_dataset.py from parquet.
Annotate label, base_score, alt_score, and scores.*.winner manually.
Golden uses: names, addresses, websites, categories (plural).
Adapted uses: name, address, website, category (singular).
"""
import argparse
import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "analysis/inspection/golden/golden_dataset.json"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "rows_adapted_from_golden.jsonl"


def golden_to_adapted(golden_path: Path, out_path: Path, limit: int | None = None) -> int:
    """Convert golden JSON to adapted JSONL. limit=None means all records."""
    with open(golden_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    key_map = {"names": "name", "addresses": "address", "websites": "website", "categories": "category"}
    if limit is not None:
        records = records[:limit]
    rows = []
    for r in records:
        data = r.get("data", {})
        base_raw = data.get("base", {})
        other_raw = data.get("current", {})
        base = {key_map.get(k, k): v for k, v in base_raw.items()}
        other = {key_map.get(k, k): v for k, v in other_raw.items()}
        rows.append({
            "id": r.get("id", ""),
            "base_id": r.get("base_id", r.get("id", "")),
            "base": base,
            "other": other,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert golden_dataset.json to rows_adapted_from_golden.jsonl")
    parser.add_argument("--golden", type=Path, default=GOLDEN, help="Input golden JSON path")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Max rows (default: all)")
    args = parser.parse_args()
    n = golden_to_adapted(args.golden, args.out, limit=args.limit)
    print(f"Wrote {n} rows to {args.out}")
