#!/usr/bin/env python3
"""
analyze_deployment.py
=====================
Loads one or more deployment eval CSVs and prints a structured analysis:

  1. Overall accuracy + per-class recall
  2. Confidence calibration table
  3. Extract stats (word count distribution)
  4. Flagged rows for manual review:
       - Wrong predictions with high confidence (≥ 0.85) — calibration failures
       - Correct predictions with low confidence (≤ 0.5) — under-confidence
       - Very short extracts (< 5 words) — possible extraction failures
       - Very long extracts (> 80 words) — possible over-extraction

Usage (from repo root):
    python3 eval/analyze_deployment.py --run-dir deployment/d_v1
    python3 eval/analyze_deployment.py --files eval/results/deployment/d_v1/eval_gpt-5.4_d_v1.csv
"""

import csv
import sys
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABELS = ["divine_teleology", "internal_essence", "junk", "non_divine_teleology"]


def load_csvs(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def calibration_table(rows: list[dict]):
    buckets = [(0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]
    print(f"\n{'Confidence':>15}  {'N':>4}  {'Avg conf':>9}  {'Accuracy':>9}")
    for lo, hi in buckets:
        subset = [r for r in rows if lo <= float(r["confidence"]) < hi]
        if not subset:
            continue
        avg_c = sum(float(r["confidence"]) for r in subset) / len(subset)
        acc   = sum(r["correct"] == "True" for r in subset) / len(subset)
        label = f"{lo:.0%}–{hi:.0%}" if hi < 1.01 else f"{lo:.0%}–100%"
        print(f"  {label:>13}  {len(subset):>4}  {avg_c:>9.2f}  {acc:>9.0%}")


def extract_stats(rows: list[dict]):
    lengths = [len(r["extract"].split()) for r in rows]
    lengths_by_tag = {}
    for r in rows:
        tag = r["predicted_tag"]
        lengths_by_tag.setdefault(tag, []).append(len(r["extract"].split()))

    print(f"\nExtract word-count distribution (all rows, N={len(lengths)}):")
    print(f"  min={min(lengths)}  median={sorted(lengths)[len(lengths)//2]}  max={max(lengths)}")
    print(f"  < 5 words:  {sum(l < 5 for l in lengths)} rows")
    print(f"  > 80 words: {sum(l > 80 for l in lengths)} rows")

    print("\n  Per predicted tag:")
    for tag in LABELS:
        ls = lengths_by_tag.get(tag, [])
        if not ls:
            continue
        print(f"    {tag:<25} N={len(ls):>3}  avg={sum(ls)/len(ls):.0f}w  "
              f"min={min(ls)}w  max={max(ls)}w")


def flag_rows(rows: list[dict]):
    wrong_hc = [r for r in rows if r["correct"] == "False" and float(r["confidence"]) >= 0.85]
    right_lc = [r for r in rows if r["correct"] == "True"  and float(r["confidence"]) <= 0.5]
    short_ex  = [r for r in rows if len(r["extract"].split()) < 5]
    long_ex   = [r for r in rows if len(r["extract"].split()) > 80]

    def show(title, subset, fields):
        if not subset:
            print(f"\n{title}: none")
            return
        print(f"\n{title} ({len(subset)}):")
        for r in subset:
            vals = "  |  ".join(f"{k}={r[k][:60] if len(r.get(k,''))>60 else r.get(k,'')!r}" for k in fields)
            print(f"  {vals}")

    show(
        "Wrong + high confidence (≥0.85) [calibration failures]",
        wrong_hc,
        ["correct_tag", "predicted_tag", "confidence", "extract"],
    )
    show(
        "Correct + low confidence (≤0.5) [under-confidence]",
        right_lc,
        ["correct_tag", "confidence", "extract"],
    )
    show(
        "Short extracts (< 5 words) [extraction failures?]",
        short_ex,
        ["predicted_tag", "extract", "text"],
    )
    show(
        "Long extracts (> 80 words) [over-extraction?]",
        long_ex,
        ["predicted_tag", "extract"],
    )


def print_accuracy(rows: list[dict]):
    correct = sum(r["correct"] == "True" for r in rows)
    total   = len(rows)
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.1%}  (across {total} rows)")

    print("\nPer-class recall:")
    for label in LABELS:
        subset = [r for r in rows if r["correct_tag"] == label]
        if not subset:
            continue
        rec = sum(r["correct"] == "True" for r in subset) / len(subset)
        n_correct = sum(r["correct"] == "True" for r in subset)
        print(f"  {label:<25} {rec:.0%}  ({n_correct}/{len(subset)})")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Subdirectory of eval/results/ — loads all CSVs in it")
    parser.add_argument("--files", nargs="+", type=str, default=None,
                        help="Explicit CSV file paths")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent / "results"

    if args.files:
        paths = [Path(f) for f in args.files]
    elif args.run_dir:
        paths = sorted(base.glob(f"{args.run_dir}/*.csv"))
    else:
        print("Provide --run-dir or --files", file=sys.stderr)
        sys.exit(1)

    if not paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(paths)} file(s): {[p.name for p in paths]}")
    rows = load_csvs(paths)
    print(f"Total rows: {len(rows)}")

    print_accuracy(rows)
    print("\n--- Confidence calibration ---")
    calibration_table(rows)
    print("\n--- Extract stats ---")
    extract_stats(rows)
    print("\n--- Flagged rows for manual review ---")
    flag_rows(rows)


if __name__ == "__main__":
    main()
