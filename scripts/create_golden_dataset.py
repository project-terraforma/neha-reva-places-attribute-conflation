"""
Create a golden dataset of 200 values for manual labeling.
Biases toward "interesting" rows: conflicts or one-sided presence.

Run from project root: python scripts/create_golden_dataset.py

Outputs: analysis/inspection/golden/golden_dataset.json
"""
import json
import duckdb
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "project_a_samples.parquet"
OUT_DIR = PROJECT_ROOT / "analysis" / "inspection" / "golden"

GOLDEN_SAMPLE_N = 200


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")

    existing = set(con.execute(f"DESCRIBE SELECT * FROM '{PARQUET_PATH}'").fetchdf()["column_name"])
    driver = "phones" if "phones" in existing and "base_phones" in existing else None

    if driver:
        query = f"""
            SELECT id, base_id, names, base_names, phones, base_phones, websites, base_websites,
                   addresses, base_addresses, categories, base_categories, confidence, base_confidence
            FROM '{PARQUET_PATH}'
            WHERE
                ({driver} IS NOT NULL OR base_{driver} IS NOT NULL)
            ORDER BY
                CASE
                    WHEN {driver} IS NOT NULL AND base_{driver} IS NOT NULL AND {driver} != base_{driver} THEN 0
                    WHEN {driver} IS NOT NULL AND base_{driver} IS NULL THEN 1
                    WHEN {driver} IS NULL AND base_{driver} IS NOT NULL THEN 2
                    ELSE 3
                END,
                random()
            LIMIT {GOLDEN_SAMPLE_N}
        """
    else:
        query = f"""
            SELECT id, base_id, names, base_names, phones, base_phones, websites, base_websites,
                   addresses, base_addresses, categories, base_categories, confidence, base_confidence
            FROM '{PARQUET_PATH}'
            USING SAMPLE {GOLDEN_SAMPLE_N} ROWS
        """

    df = con.execute(query).fetchdf()

    def to_entry_val(v, key):
        """Convert to JSON-serializable value; use [null] for empty phones/websites."""
        if pd.isna(v) or v is None:
            return "[null]" if key in ("phones", "websites") else None
        if isinstance(v, (int, float)):
            return v
        return str(v)

    records = []
    for record_index, (_, row) in enumerate(df.iterrows()):
        current = {
            "names": to_entry_val(row.get("names"), "names"),
            "phones": to_entry_val(row.get("phones"), "phones"),
            "websites": to_entry_val(row.get("websites"), "websites"),
            "addresses": to_entry_val(row.get("addresses"), "addresses"),
            "categories": to_entry_val(row.get("categories"), "categories"),
            "confidence": to_entry_val(row.get("confidence"), "confidence"),
        }
        base = {
            "names": to_entry_val(row.get("base_names"), "names"),
            "phones": to_entry_val(row.get("base_phones"), "phones"),
            "websites": to_entry_val(row.get("base_websites"), "websites"),
            "addresses": to_entry_val(row.get("base_addresses"), "addresses"),
            "categories": to_entry_val(row.get("base_categories"), "categories"),
            "confidence": to_entry_val(row.get("base_confidence"), "confidence"),
        }

        records.append({
            "id": str(row["id"]),
            "record_index": record_index,
            "label": "",
            "method": "manual_review (manual)",
            "data": {"current": current, "base": base},
        })

    out_path = OUT_DIR / "golden_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"Golden labeling sample saved to: {out_path} ({len(records)} records)")
    con.close()


if __name__ == "__main__":
    main()
