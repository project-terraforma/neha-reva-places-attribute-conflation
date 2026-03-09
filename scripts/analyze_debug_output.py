#!/usr/bin/env python3
"""
Analyze debug output: compare our labels vs golden labels for accuracy and F1.

Run standalone: python scripts/analyze_debug_output.py [--debug PATH] [--golden PATH]
Also invoked by flow --debug.
"""
import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEBUG_DEFAULT = PROJECT / "out/agentic_labels_debug.jsonl"
GOLDEN_DEFAULT = PROJECT / "analysis/inspection/golden/golden_dataset.json"


def _f1_score(y_true: list[int], y_pred: list[int], labels: tuple[int, ...] = (0, 1, 2)) -> tuple[float, float]:
    """
    Compute macro and weighted F1 for multiclass. Returns (macro_f1, weighted_f1).
    """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    f1_per_class = []
    support_per_class = []
    for k in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == k and p == k)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == k and t != k)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == k and p != k)
        support = tp + fn
        support_per_class.append(support)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
    macro_f1 = sum(f1_per_class) / len(labels) if labels else 0.0
    total_support = sum(support_per_class)
    weighted_f1 = (
        sum(f1 * s for f1, s in zip(f1_per_class, support_per_class)) / total_support
        if total_support > 0 else 0.0
    )
    return macro_f1, weighted_f1


def run_analysis(debug_path: Path, golden_path: Path) -> None:
    """Print accuracy and F1. Used by flow --debug and when run standalone."""
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

    y_true = []
    y_pred = []
    for row in rows:
        rid = row.get("id", "")
        g = golden.get(rid, {})
        golden_label = g.get("label")
        our_label = row.get("label")
        if golden_label is not None:
            y_true.append(int(golden_label))
            y_pred.append(int(our_label) if our_label is not None else 2)

    if not y_true:
        print("Accuracy: No annotated golden labels found. Fill in label in golden_dataset.json.")
        return

    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = matches / len(y_true)
    macro_f1, weighted_f1 = _f1_score(y_true, y_pred)

    print("=" * 50)
    print("Accuracy & F1 vs Golden Labels")
    print("=" * 50)
    print(f"Accuracy: {matches}/{len(y_true)} = {100*accuracy:.1f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Compare agentic labels vs golden labels")
    parser.add_argument("--debug", type=Path, default=DEBUG_DEFAULT, help="Debug JSONL path")
    parser.add_argument("--golden", type=Path, default=GOLDEN_DEFAULT, help="Golden dataset JSON path")
    args = parser.parse_args()
    run_analysis(args.debug, args.golden)


if __name__ == "__main__":
    main()
