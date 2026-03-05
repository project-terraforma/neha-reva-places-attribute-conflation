#!/usr/bin/env python3
"""
Convert golden_dataset.json to rows_adapted.jsonl format for flow input.
Golden uses: names, addresses, websites, categories (plural).
Adapted uses: name, address, website, category (singular).
"""
import json
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "analysis/inspection/golden/golden_dataset.json"


def golden_to_adapted(golden_path: Path, out_path: Path, limit: int = 10):
    with open(golden_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    key_map = {"names": "name", "addresses": "address", "websites": "website", "categories": "category"}
    rows = []
    for r in records[:limit]:
        data = r.get("data", {})
        base_raw = data.get("base", {})
        other_raw = data.get("current", {})
        base = {key_map.get(k, k): v for k, v in base_raw.items()}
        other = {key_map.get(k, k): v for k, v in other_raw.items()}
        rows.append({
            "id": r.get("id", ""),
            "base_id": r.get("id", ""),
            "base": base,
            "other": other,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    return len(rows)


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = Path(__file__).resolve().parents[1] / "data" / "rows_adapted_from_golden.jsonl"
    n = golden_to_adapted(GOLDEN, out, limit=limit)
    print(f"Wrote {n} rows to {out}")
