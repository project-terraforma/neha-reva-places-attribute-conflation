#!/usr/bin/env python3
"""
Analyze debug output: compare our labels vs golden labels for accuracy.

Run standalone: python scripts/analyze_debug_output.py [--debug PATH] [--golden PATH]
Also invoked by flow --debug.
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEBUG_DEFAULT = PROJECT / "out/agentic_labels_debug.jsonl"
GOLDEN_DEFAULT = PROJECT / "analysis/inspection/golden/golden_dataset.json"


def run_analysis(debug_path: Path, golden_path: Path) -> None:
    """Print accuracy analysis. Used by flow --debug and when run standalone."""
    if not debug_path.exists():
        print(f"Debug file not found: {debug_path}")
        return
    with open(debug_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return
    if not golden_path.exists():
        print("(Golden dataset not found, skipping accuracy comparison)")
        return
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = {r["id"]: r for r in json.load(f)}

    print("=" * 80)
    print("DEBUG OUTPUT ANALYSIS - Accuracy vs Golden Labels")
    print("=" * 80)
    matches = 0
    annotated_count = 0
    for i, row in enumerate(rows):
        rid = row.get("id", "")
        g = golden.get(rid, {})
        golden_label = g.get("label")
        our_label = row.get("label")
        is_annotated = golden_label is not None
        if is_annotated:
            annotated_count += 1
            if our_label == golden_label:
                matches += 1
        match = "✓" if (our_label == golden_label and is_annotated) else ("—" if not is_annotated else "✗")
        print(f"\n--- Row {i+1}: {rid[:16]}... ---")
        print(f"  Our label: {our_label} (0=base, 1=alt, 2=abstain)")
        print(f"  Golden label: {golden_label or '(not annotated)'}")
        print(f"  Match: {match}")
        print(f"  base_score={row.get('base_score')} alt_score={row.get('alt_score')}")
        print(f"  attr_winners: {row.get('attr_winners')}")
        dbg = row.get("debug", {})
        if dbg:
            print(f"  website_reason: {dbg.get('website_reason')}")
            print(f"  phones_reason: {dbg.get('phones_reason')}")
            print(f"  address_reason: {dbg.get('address_reason')}")
            print(f"  category_reason: {dbg.get('category_reason')}")
            print(f"  label_reason: {dbg.get('label_reason')}")
    print("\n" + "=" * 80)
    if annotated_count > 0:
        print(f"Accuracy (annotated only): {matches}/{annotated_count} = {100*matches/annotated_count:.1f}%")
    else:
        print("Accuracy: No annotated golden labels found. Fill in label in golden_dataset.json.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare agentic labels vs golden labels")
    parser.add_argument("--debug", type=Path, default=DEBUG_DEFAULT, help="Debug JSONL path")
    parser.add_argument("--golden", type=Path, default=GOLDEN_DEFAULT, help="Golden dataset JSON path")
    args = parser.parse_args()
    run_analysis(args.debug, args.golden)


if __name__ == "__main__":
    main()
