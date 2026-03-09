#!/usr/bin/env python3
"""
Create golden_dataset.json from parquet for manual annotation.

Outputs a JSON file with entry information (base + current/other) and empty
fields for label, base_score, alt_score, and per-attribute scores (category winners).
Fill these in manually to create your golden dataset.
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PARQUET_DEFAULT = PROJECT / "data/project_a_samples.parquet"
GOLDEN_DEFAULT = PROJECT / "analysis/inspection/golden/golden_dataset.json"

# Parquet column -> golden data.base / data.current key
BASE_COLS = {
    "base_names": "names",
    "base_addresses": "addresses",
    "base_phones": "phones",
    "base_websites": "websites",
    "base_categories": "categories",
    "base_brand": "brand",
    "base_socials": "socials",
    "base_emails": "emails",
}
OTHER_COLS = {
    "names": "names",
    "addresses": "addresses",
    "phones": "phones",
    "websites": "websites",
    "categories": "categories",
    "brand": "brand",
    "socials": "socials",
    "emails": "emails",
}
# Parquet may have confidence columns
BASE_CONF = "base_confidence"
OTHER_CONF = "confidence"

SCORE_ATTRS = ["name", "phones", "website", "address", "category"]


def _safe_val(val):
    """Handle nulls and ensure JSON-serializable."""
    if val is None:
        return None
    if hasattr(val, "__iter__") and not isinstance(val, (str, dict)):
        try:
            return list(val)
        except (TypeError, ValueError):
            return val
    try:
        import pandas as pd
        if pd.isna(val):
            return None
    except ImportError:
        pass
    return val


def create_golden_dataset(parquet_path: Path, json_path: Path, limit: int | None = None) -> int:
    """Create golden_dataset.json from parquet with empty annotation fields."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    records = []
    for idx, row in df.iterrows():
        if limit is not None and len(records) >= limit:
            break
        raw = row.to_dict()
        base = {}
        for pcol, gkey in BASE_COLS.items():
            v = _safe_val(raw.get(pcol))
            if v is not None and str(v).strip():
                base[gkey] = v
        conf = _safe_val(raw.get(BASE_CONF))
        if conf is not None:
            base["confidence"] = conf

        current = {}
        for pcol, gkey in OTHER_COLS.items():
            v = _safe_val(raw.get(pcol))
            if v is not None and str(v).strip():
                current[gkey] = v
        conf = _safe_val(raw.get(OTHER_CONF))
        if conf is not None:
            current["confidence"] = conf

        records.append({
            "id": str(raw.get("id", "")),
            "base_id": str(raw.get("base_id", "")),
            "record_index": len(records),
            "label": None,
            "base_score": None,
            "alt_score": None,
            "method": "manual",
            "data": {"base": base, "current": current},
            "scores": {attr: {"winner": None} for attr in SCORE_ATTRS},
        })
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    return len(records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create golden_dataset.json from parquet")
    parser.add_argument("--input", type=Path, default=PARQUET_DEFAULT, help="Input parquet path")
    parser.add_argument("--out", type=Path, default=GOLDEN_DEFAULT, help="Output golden JSON path")
    parser.add_argument("--limit", type=int, default=None, help="Max records (default: all)")
    args = parser.parse_args()
    n = create_golden_dataset(args.input, args.out, limit=args.limit)
    print(f"Wrote {n} records to {args.out}. Fill in label, base_score, alt_score, and scores.*.winner manually.")
