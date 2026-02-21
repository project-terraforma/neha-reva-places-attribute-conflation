"""
Shared logic for attribute-pair inspection (base_X vs X).
Used by inspect_categories.py, inspect_addresses.py, inspect_phones.py, inspect_websites.py.
"""
import json
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "project_a_samples.parquet"
OUT_DIR = PROJECT_ROOT / "analysis" / "inspection" / "attributes"

SAMPLE_N = 50
DISAGREEMENT_SAMPLE_N = 20


def run_attr_analysis(attr: str) -> None:
    """
    Analyze a single attribute pair (base_{attr} vs {attr}).
    Prints stats, examples, disagreement samples; exports JSON only.
    """
    base_attr = f"base_{attr}"
    con = duckdb.connect(database=":memory:")

    # Check columns exist
    existing = set(con.execute(f"DESCRIBE SELECT * FROM '{PARQUET_PATH}'").fetchdf()["column_name"])
    if attr not in existing or base_attr not in existing:
        print(f"[error] Missing {attr} or {base_attr} in dataset.")
        con.close()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = con.execute(f"SELECT COUNT(*) FROM '{PARQUET_PATH}'").fetchone()[0]
    row = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE {attr} IS NOT NULL) AS has_attr,
            COUNT(*) FILTER (WHERE {base_attr} IS NOT NULL) AS has_base,
            COUNT(*) FILTER (WHERE {attr} IS NOT NULL AND {base_attr} IS NOT NULL) AS comparable,
            COUNT(*) FILTER (WHERE {attr} IS NOT NULL AND {base_attr} IS NOT NULL AND {attr}::VARCHAR != {base_attr}::VARCHAR) AS disagree
        FROM '{PARQUET_PATH}'
    """).fetchone()

    has_attr, has_base, comparable, disagree = row
    agree = comparable - disagree
    disagree_pct = 100 * disagree / comparable if comparable > 0 else 0

    print("=" * 60)
    print(f"ATTRIBUTE PAIR: base_{attr} vs {attr}")
    print("=" * 60)
    print(f"\nTotal rows:           {total:,}")
    print(f"Rows with {attr}:         {has_attr:,} ({100*has_attr/total:.1f}%)")
    print(f"Rows with base_{attr}:  {has_base:,} ({100*has_base/total:.1f}%)")
    print(f"Comparable (both):    {comparable:,}")
    print(f"  Agree:              {agree:,}")
    print(f"  Disagree:           {disagree:,} ({disagree_pct:.1f}%)")
    print("-" * 60)

    # Value examples
    print(f"\n=== Sample values (first 8 comparable rows) ===")
    df_ex = con.execute(f"""
        SELECT id, base_id, {base_attr}, {attr}
        FROM '{PARQUET_PATH}'
        WHERE {attr} IS NOT NULL AND {base_attr} IS NOT NULL
        LIMIT 8
    """).fetchdf()
    for i, r in df_ex.iterrows():
        print(f"\n  id={r['id']}")
        print(f"    base_{attr}: {_trunc(str(r[base_attr]), 100)}")
        print(f"    {attr}:      {_trunc(str(r[attr]), 100)}")

    # Disagreement examples
    if disagree > 0:
        print(f"\n=== Disagreement examples (first {DISAGREEMENT_SAMPLE_N}) ===")
        df_dis = con.execute(f"""
            SELECT id, base_id, {base_attr}, {attr}
            FROM '{PARQUET_PATH}'
            WHERE {attr} IS NOT NULL AND {base_attr} IS NOT NULL AND {attr}::VARCHAR != {base_attr}::VARCHAR
            LIMIT {DISAGREEMENT_SAMPLE_N}
        """).fetchdf()
        for i, r in df_dis.iterrows():
            print(f"\n  id={r['id']}")
            print(f"    base_{attr}: {_trunc(str(r[base_attr]), 80)}")
            print(f"    {attr}:      {_trunc(str(r[attr]), 80)}")

    # Export full sample for this attribute (JSON only)
    df_export = con.execute(f"""
        SELECT id, base_id, {base_attr}, {attr}
        FROM '{PARQUET_PATH}'
        WHERE {attr} IS NOT NULL OR {base_attr} IS NOT NULL
        LIMIT {SAMPLE_N}
    """).fetchdf()

    json_path = OUT_DIR / f"{attr}_pair_sample.json"
    records = []
    for _, row in df_export.iterrows():
        obj = {k: row[k] for k in df_export.columns}
        records.append(obj)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"\nExported: {json_path}")
    print("\n" + "=" * 60)
    con.close()


def _trunc(s: str, max_len: int) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s
