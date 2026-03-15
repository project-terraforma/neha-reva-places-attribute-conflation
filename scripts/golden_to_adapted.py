#!/usr/bin/env python3
"""
Convert golden_dataset.json to agentic_input.jsonl format for flow input.
For the adapted output required by the downstream "flow", these plural keys are mapped to their singular forms (name, address, website, category).
This conversion script outputs agentic_input.jsonl, with each line containing a simplified record in the adapted format.
"""
import argparse
import json
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "inspection/golden/golden_dataset.json"
OUT_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "agentic_input.jsonl"


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
