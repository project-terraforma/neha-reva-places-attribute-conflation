"""
Create a golden dataset of 200 values for manual labeling.
Biases toward "interesting" rows: conflicts or one-sided presence.

Run from project root: python scripts/create_golden_dataset.py

Outputs: analysis/inspection/golden/golden_dataset.csv
        analysis/inspection/golden/golden_dataset.json  # [CHANGE: JSON with scores area - undo to remove]
"""
import json
import duckdb
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "project_a_samples.parquet"
OUT_DIR = PROJECT_ROOT / "analysis" / "inspection" / "golden"

GOLDEN_SAMPLE_N = 200

# [CHANGE: placeholder for attribute scores - undo to remove]
def _empty_scores():
    """Placeholder scores structure for JSON output."""
    return {
        "phones": {"winner": None, "base_q": None, "alt_q": None},
        "websites": {"winner": None, "base_q": None, "alt_q": None},
        "addresses": {"winner": None, "base_q": None, "alt_q": None},
        "categories": {"winner": None, "base_q": None, "alt_q": None},
        "base_score": None,
        "alt_score": None,
    }


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
    out = df.copy()

    out["label"] = ""
    out["notes"] = ""

    out_path = OUT_DIR / "golden_dataset.csv"
    out.to_csv(out_path, index=False)
    print(f"Golden labeling sample saved to: {out_path} ({len(out)} records)")

    # [CHANGE: create JSON file with scores area - undo to remove]
    json_records = []
    for _, row in out.iterrows():
        rec = {k: (None if pd.isna(v) else v) for k, v in row.items()}
        rec["label"] = None  # placeholder for auto_label to fill
        rec["scores"] = _empty_scores()  # placeholder for attribute scores
        json_records.append(rec)
    json_path = OUT_DIR / "golden_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2, default=str)
    print(f"Golden dataset JSON saved to: {json_path} ({len(json_records)} records)")
    # [END CHANGE]

    con.close()


if __name__ == "__main__":
    main()
