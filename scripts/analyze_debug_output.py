#!/usr/bin/env python3
"""
Analyze debug output: compare our labels vs golden labels for accuracy.

Run standalone: python scripts/analyze_debug_output.py [--debug PATH] [--golden PATH]
Also invoked by flow --debug.

Reports overall label accuracy and per-attribute (name, website, category, phones, address)
accuracy.
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEBUG_DEFAULT = PROJECT / "out/agentic_labels_debug.jsonl"
GOLDEN_DEFAULT = PROJECT / "inspection/golden/golden_dataset.json"

# Golden winner: L=Left/Base, R=Right/Alt, B=Both, N=Neither. Our: 1=base, -1=alt, 2=both, 0=abstain
GOLDEN_TO_OUR = {"L": 1, "R": -1, "B": 2, "N": 0}
ATTR_KEYS = ["name", "website", "category", "phones", "address"]  # website not "website" in golden scores
GOLDEN_ATTR_MAP = {"name": "name", "website": "website", "category": "category", "phones": "phones", "address": "address"}


def _golden_winner_to_our(winner: str | None) -> int | None:
    if winner is None:
        return None
    w = (winner or "").strip().upper()
    if w not in GOLDEN_TO_OUR:
        return None
    return GOLDEN_TO_OUR[w]


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
    # Per-attribute: our winner list, golden winner list (only for annotated rows)
    attr_our = {a: [] for a in ATTR_KEYS}
    attr_golden = {a: [] for a in ATTR_KEYS}
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
            # Collect per-attribute for annotated rows
            scores_g = g.get("scores") or {}
            winners_our = row.get("attr_winners") or {}
            for attr in ATTR_KEYS:
                g_winner = (scores_g.get(attr) or {}).get("winner")
                o_val = winners_our.get(attr)
                g_val = _golden_winner_to_our(g_winner)
                if g_val is not None:
                    attr_golden[attr].append(g_val)
                    attr_our[attr].append(o_val if o_val is not None else 0)
        match = "✓" if (our_label == golden_label and is_annotated) else ("—" if not is_annotated else "✗")
        golden_display = golden_label if golden_label is not None else "(not annotated)"
        print(f"\n--- Row {i+1}: {rid[:16]}... ---")
        print(f"  Our label: {our_label} (0=base, 1=alt, 2=abstain)")
        print(f"  Golden label: {golden_display}")
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
        print(f"Overall label accuracy: {matches}/{annotated_count} = {100*matches/annotated_count:.1f}%")
        print()
        print("Per-attribute (vs golden scores.*.winner: L=base, R=alt, B=both, N=neither):")
        print("-" * 60)
        for attr in ATTR_KEYS:
            our_list = attr_our[attr]
            golden_list = attr_golden[attr]
            if not golden_list:
                print(f"  {attr}: no golden annotations")
                continue
            n = len(golden_list)
            correct = sum(1 for o, g in zip(our_list, golden_list) if o == g)
            acc = 100 * correct / n
            print(f"  {attr}: accuracy = {correct}/{n} = {acc:.1f}%")
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
