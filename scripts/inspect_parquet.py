"""
Inspect places attribute conflation Parquet data: schema, stats, samples, and golden labeling.

Run from project root: python scripts/inspect_parquet.py

Outputs:
  - Console: schema, stats, side-by-side samples, value examples
  - analysis/inspection/golden/: golden_labeling_sample.json
  - analysis/inspection/side_by_side/: side_by_side_sample.csv, .jsonl
  - analysis/inspection/attributes/: {attr}_pair_sample.csv, .jsonl (from attribute scripts)
"""
import json
import duckdb
import pandas as pd
from pathlib import Path

# Paths relative to project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
PARQUET_PATH = PROJECT_ROOT / "data" / "project_a_samples.parquet"
OUT_DIR = PROJECT_ROOT / "analysis" / "inspection"
OUT_DIR_GOLDEN = OUT_DIR / "golden"
OUT_DIR_SIDE_BY_SIDE = OUT_DIR / "side_by_side"
DATA_PATH = Path(__file__).parent.parent / "data" / "project_a_samples.parquet"

SAMPLE_N = 30          # small sample to print + export
GOLDEN_SAMPLE_N = 200  # for labeling

CORE_ATTRS = ["addresses", "categories", "phones", "websites", "names", "emails", "socials"]
ID_COLS = ["id", "base_id"]


def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR_GOLDEN.mkdir(parents=True, exist_ok=True)
    OUT_DIR_SIDE_BY_SIDE.mkdir(parents=True, exist_ok=True)


def pretty_print_schema(con):
    print("\n=== DuckDB DESCRIBE (columns + types) ===")
    schema_df = con.execute(f"DESCRIBE SELECT * FROM '{PARQUET_PATH}'").fetchdf()
    print(schema_df.to_string(index=False))


def print_quick_stats(con):
    print("\n=== Basic stats ===")
    total = con.execute(f"SELECT COUNT(*) AS n FROM '{PARQUET_PATH}'").fetchone()[0]
    print(f"Rows: {total}")

    cols = con.execute(f"DESCRIBE SELECT * FROM '{PARQUET_PATH}'").fetchdf()["column_name"].tolist()
    print(f"Columns ({len(cols)}): {cols}")


def sample_rows_side_by_side(con, n=SAMPLE_N):
    """
    Pull a few rows with core attrs shown as base vs alt columns.
    This makes it easy to see what values look like.
    """
    select_cols = ID_COLS[:]
    for a in CORE_ATTRS:
        select_cols.append(f"base_{a}")
        select_cols.append(a)

    # Some datasets may not include every attribute listed above; filter to existing columns
    existing = set(con.execute(f"DESCRIBE SELECT * FROM '{PARQUET_PATH}'").fetchdf()["column_name"])
    select_cols = [c for c in select_cols if c in existing]

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM '{PARQUET_PATH}'
        USING SAMPLE {n} ROWS
    """
    df = con.execute(query).fetchdf()
    return df


def export_sample_readable(df, prefix="sample", out_dir=None):
    """
    Export:
      - CSV for quick viewing
      - JSONL for nested structures that CSV mangles
    """
    out_dir = out_dir or OUT_DIR
    csv_path = out_dir / f"{prefix}.csv"
    jsonl_path = out_dir / f"{prefix}.jsonl"

    df.to_csv(csv_path, index=False)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # Convert pandas objects to JSON-serializable
            obj = {k: row[k] for k in df.columns}
            f.write(json.dumps(obj, default=str) + "\n")

    print(f"\nWrote:\n- {csv_path}\n- {jsonl_path}")


def show_value_examples(con, attr="phones", k=10):
    """
    Show a few example values for an attribute + its base_ version.
    Useful to understand whether it's a list, string, struct, etc.
    """
    base_attr = f"base_{attr}"

    existing = set(con.execute(f"DESCRIBE SELECT * FROM '{PARQUET_PATH}'").fetchdf()["column_name"])
    if attr not in existing or base_attr not in existing:
        print(f"\n[skip] Missing {attr} or {base_attr} in dataset.")
        return

    print(f"\n=== Examples for {attr} vs {base_attr} (first {k} non-null pairs) ===")
    df = con.execute(f"""
        SELECT id, {base_attr}, {attr}
        FROM '{PARQUET_PATH}'
        WHERE {attr} IS NOT NULL OR {base_attr} IS NOT NULL
        LIMIT {k}
    """).fetchdf()

    # Print row by row so nested values are easier to see
    for i, r in df.iterrows():
        print(f"\nRow {i} id={r['id']}")
        print(f"  {base_attr}: {r[base_attr]}")
        print(f"  {attr}:      {r[attr]}")


def make_golden_labeling_sample(con, n=GOLDEN_SAMPLE_N):
    """
    Make a reproducible golden dataset of 200 values in JSON format.
    Biases toward "interesting" rows: conflicts or one-sided presence.
    Format matches other teams: id, record_index, label, method, data{current, base}.
    """
    existing = set(con.execute(f"DESCRIBE SELECT * FROM '{PARQUET_PATH}'").fetchdf()["column_name"])

    # Pick one core attribute to drive "interestingness" if available
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
            LIMIT {n}
        """
    else:
        query = f"""
            SELECT id, base_id, names, base_names, phones, base_phones, websites, base_websites,
                   addresses, base_addresses, categories, base_categories, confidence, base_confidence
            FROM '{PARQUET_PATH}'
            USING SAMPLE {n} ROWS
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

    out_path = OUT_DIR_GOLDEN / "golden_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"\nGolden labeling sample saved to: {out_path} ({len(records)} records)")


def analyze_disagreement_rates(con):
    """
    Analyze disagreement rates between base and conflated attributes.
    For each attribute (addresses, websites, categories, phones), computes:
    - How many rows have both values present (comparable)
    - How many of those disagree (values differ)
    - Disagreement rate as % of comparable pairs
    """
    attrs = ["addresses", "websites", "categories", "phones"]
    existing = set(con.execute(f"DESCRIBE SELECT * FROM '{PARQUET_PATH}'").fetchdf()["column_name"])

    total = con.execute(f"SELECT COUNT(*) FROM '{PARQUET_PATH}'").fetchone()[0]

    print("\n=== Disagreement rates (base vs conflated) ===")
    print("-" * 60)
    print(f"{'Attribute':<12} {'Comparable':>10} {'Disagree':>10} {'Agree':>10} {'Disagree %':>10}")
    print("-" * 60)

    for attr in attrs:
        base_attr = f"base_{attr}"
        if attr not in existing or base_attr not in existing:
            print(f"  {attr:<12} (column missing)")
            continue

        row = con.execute(f"""
            SELECT
                COUNT(*) FILTER (WHERE {attr} IS NOT NULL AND {base_attr} IS NOT NULL) AS comparable,
                COUNT(*) FILTER (WHERE {attr} IS NOT NULL AND {base_attr} IS NOT NULL AND {attr}::VARCHAR != {base_attr}::VARCHAR) AS disagree
            FROM '{PARQUET_PATH}'
        """).fetchone()

        comparable, disagree = row[0], row[1]
        agree = comparable - disagree
        disagree_pct = 100 * disagree / comparable if comparable > 0 else 0

        print(f"  {attr:<12} {comparable:>10,} {disagree:>10,} {agree:>10,} {disagree_pct:>9.1f}%")

    print("-" * 60)
    print(f"  Total rows: {total:,}")
    print("  (Comparable = rows where both base_X and X are non-null)")


def main():
    ensure_out_dir()
    con = duckdb.connect(database=":memory:")

    print("=" * 60)
    print("PLACES ATTRIBUTE CONFLATION — Dataset Overview")
    print("=" * 60)

    # --- Row count ---
    row_count = con.execute(f"SELECT COUNT(*) FROM '{DATA_PATH}'").fetchone()[0]
    # print(f"\nROW COUNT: {row_count:,}")

    print_quick_stats(con)

    # --- Schema ---
    # print("\nSCHEMA (columns & types)")
    # print("-" * 40)
    schema = con.execute(f"DESCRIBE SELECT * FROM '{DATA_PATH}'").fetchdf()
    # for _, row in schema.iterrows():
    #     print(f"  {row['column_name']:<20} {row['column_type']}")

    pretty_print_schema(con)

    # --- Null/missing stats ---
    print("\nNULL COUNTS per column")
    print("-" * 40)
    null_counts = []
    for col in schema["column_name"]:
        n = con.execute(f"SELECT COUNT(*) FROM '{DATA_PATH}' WHERE {col} IS NULL").fetchone()[0]
        pct = 100 * n / row_count
        null_counts.append((col, n, pct))
    for col, n, pct in sorted(null_counts, key=lambda x: -x[1]):
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {col:<20} {n:>5} ({pct:5.1f}%) {bar}")

    # --- Disagreement rates ---
    analyze_disagreement_rates(con)

    # --- Confidence distribution ---
    print("\nCONFIDENCE distribution (conflated vs base)")
    print("-" * 40)
    conf_stats = con.execute(f"""
        SELECT
            ROUND(confidence, 2) AS conf_bin,
            COUNT(*) AS cnt
        FROM '{DATA_PATH}'
        WHERE confidence IS NOT NULL
        GROUP BY conf_bin
        ORDER BY conf_bin DESC
        LIMIT 10
    """).fetchdf()
    print(conf_stats.to_string(index=False))
    base_conf = con.execute(f"""
        SELECT
            MIN(base_confidence) AS min_base,
            MAX(base_confidence) AS max_base,
            AVG(base_confidence) AS avg_base
        FROM '{DATA_PATH}'
        WHERE base_confidence IS NOT NULL
    """).fetchone()
    print(f"\n  base_confidence: min={base_conf[0]:.2f}, max={base_conf[1]:.2f}, avg={base_conf[2]:.2f}")

    # --- Sample of key attributes ---
    print("\nSAMPLE ROWS (key attributes)")
    print("-" * 40)
    sample = con.execute(f"""
        SELECT id, base_id, names, categories, confidence,
            base_names, base_categories, base_confidence
        FROM '{DATA_PATH}'
        LIMIT 5
    """).fetchdf()
    print(sample.to_string())

    # --- Uniqueness ---
    print("\nUNIQUENESS")
    print("-" * 40)
    unique_id = con.execute(f"SELECT COUNT(DISTINCT id) FROM '{DATA_PATH}'").fetchone()[0]
    unique_base = con.execute(f"SELECT COUNT(DISTINCT base_id) FROM '{DATA_PATH}'").fetchone()[0]
    print(f"  Unique id:       {unique_id:,}")
    print(f"  Unique base_id:  {unique_base:,}")
    if unique_base < row_count:
        dupes = con.execute(f"""
            SELECT base_id, COUNT(*) AS n
            FROM '{DATA_PATH}'
            GROUP BY base_id
            HAVING COUNT(*) > 1
            ORDER BY n DESC
            LIMIT 5
        """).fetchdf()
        print(f"  (Multiple conflated records per base_id; top duplicate bases:)")
        print(dupes.to_string(index=False))

    # --- Full sample (all columns) ---
    print("\nFULL SAMPLE (first row, all columns)")
    print("-" * 40)
    full_row = con.execute(f"SELECT * FROM '{DATA_PATH}' LIMIT 1").fetchdf()
    for col in full_row.columns:
        val = full_row[col].iloc[0]
        val_str = str(val)[:80] + "..." if val is not None and len(str(val)) > 80 else str(val)
        print(f"  {col}: {val_str}")

    
    

    # Side-by-side sample
    df_sample = sample_rows_side_by_side(con, n=SAMPLE_N)
    # print("\n=== Side-by-side sample (first 10 rows) ===")
    # print(df_sample.head(10).to_string(index=False))
    export_sample_readable(df_sample, prefix="side_by_side_sample", out_dir=OUT_DIR_SIDE_BY_SIDE)

    # Show a few example values per attribute (helps interpret nested fields)
    for a in ["phones", "websites", "addresses", "categories", "names", "emails", "sources"]:
        show_value_examples(con, attr=a, k=8)

    # Create labeling sample for golden dataset
    make_golden_labeling_sample(con, n=GOLDEN_SAMPLE_N)

    con.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
