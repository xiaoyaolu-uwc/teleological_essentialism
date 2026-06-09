#!/usr/bin/env python3
"""
analyze_errors.py
=================
Aggregates eval result CSVs, counts how often each passage was mislabeled
across runs, and reports the worst offenders.

Usage (from repo root):
    # Single-layer:
    python3 eval/analyze_errors.py --run-dir v7c

    # Two-layer (produces l1_error_analysis.csv + l2_error_analysis.csv):
    python3 eval/analyze_errors.py --run-dir two_layer_l1v1_l2v1 --two-layer

    # Explicit files:
    python3 eval/analyze_errors.py path/to/a.csv path/to/b.csv

Output:
    CLI: aggregate stats + top-N most-mislabeled passages
    File: eval/results/<run-dir>/analysis/error_analysis.csv

Two-layer mode additionally writes:
    eval/results/<run-dir>/analysis/l1_error_analysis.csv  — L1 junk gate accuracy
    eval/results/<run-dir>/analysis/l2_error_analysis.csv  — L2 category accuracy
                                                              (restricted to non-junk gold passages)
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TOP_N       = 10
ALL_LABELS  = ["divine_teleology", "internal_essence", "junk", "non_divine_teleology", "non_junk", "error"]
# ---------------------------------------------------------------------------


def load_results(files: list[Path] = None) -> list[dict]:
    """Read result CSVs and return a flat list of rows."""
    if not files:
        files = [f for f in RESULTS_DIR.glob("*.csv") if f.parent == RESULTS_DIR]
    if not files:
        print(f"No result CSVs found in {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in sorted(files):
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row["_source"] = f.name
                rows.append(row)
    print(f"Loaded {len(files)} result file(s), {len(rows)} total rows.\n")
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Group by text, count mislabellings and per-label frequencies."""
    by_text = defaultdict(lambda: {
        "correct_tag":      "",
        "your_rationale":   "",
        "total_runs":       0,
        "mislabeled_count": 0,
        **{f"{l}_n": 0 for l in ALL_LABELS},
    })

    for row in rows:
        text = row["text"]
        entry = by_text[text]
        entry["correct_tag"]    = row["correct_tag"]
        entry["your_rationale"] = row.get("your_rationale", "")
        entry["total_runs"]    += 1
        label = row["predicted_tag"].strip().lower()
        key = f"{label}_n" if f"{label}_n" in entry else "error_n"
        entry[key] += 1
        if row["correct"] != "True":
            entry["mislabeled_count"] += 1

    return sorted(
        [{"text": t, **v} for t, v in by_text.items()],
        key=lambda r: r["mislabeled_count"],
        reverse=True,
    )


def aggregate_l1(rows: list[dict]) -> list[dict]:
    """Two-layer L1 analysis: treat each passage as junk vs non-junk.

    Gold label is 'junk' if correct_tag == 'junk', else 'non_junk'.
    Predicted label is the l1_tag column.
    """
    L1_LABELS = ["junk", "non_junk", "error"]
    by_text = defaultdict(lambda: {
        "correct_tag":      "",
        "your_rationale":   "",
        "total_runs":       0,
        "mislabeled_count": 0,
        **{f"{l}_n": 0 for l in L1_LABELS},
    })

    for row in rows:
        if "l1_tag" not in row:
            continue
        text = row["text"]
        entry = by_text[text]
        gold_tag = row["correct_tag"]
        entry["correct_tag"]    = gold_tag
        entry["your_rationale"] = row.get("your_rationale", "")
        entry["total_runs"]    += 1

        l1_pred = row["l1_tag"].strip().lower()
        gold_binary = "junk" if gold_tag == "junk" else "non_junk"
        key = f"{l1_pred}_n" if f"{l1_pred}_n" in entry else "error_n"
        entry[key] += 1
        if l1_pred != gold_binary:
            entry["mislabeled_count"] += 1

    return sorted(
        [{"text": t, **v} for t, v in by_text.items()],
        key=lambda r: r["mislabeled_count"],
        reverse=True,
    )


def aggregate_l2(rows: list[dict]) -> list[dict]:
    """Two-layer L2 analysis: category accuracy on non-junk passages only.

    Restricted to passages where correct_tag != 'junk'.
    Uses predicted_tag (the final L2 decision) vs correct_tag.
    """
    L2_LABELS = ["divine_teleology", "internal_essence", "non_divine_teleology", "error"]
    by_text = defaultdict(lambda: {
        "correct_tag":      "",
        "your_rationale":   "",
        "total_runs":       0,
        "mislabeled_count": 0,
        **{f"{l}_n": 0 for l in L2_LABELS},
    })

    for row in rows:
        if row["correct_tag"] == "junk":
            continue  # L2 analysis only covers non-junk passages
        text = row["text"]
        entry = by_text[text]
        entry["correct_tag"]    = row["correct_tag"]
        entry["your_rationale"] = row.get("your_rationale", "")
        entry["total_runs"]    += 1
        label = row["predicted_tag"].strip().lower()
        key = f"{label}_n" if f"{label}_n" in entry else "error_n"
        entry[key] += 1
        if row["correct"] != "True":
            entry["mislabeled_count"] += 1

    return sorted(
        [{"text": t, **v} for t, v in by_text.items()],
        key=lambda r: r["mislabeled_count"],
        reverse=True,
    )


def print_stats(rows: list[dict]):
    """Print aggregate accuracy, per-class F1, and confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = [r["correct_tag"] for r in rows]
    y_pred = [r["predicted_tag"].strip().lower() for r in rows]
    all_labels = sorted(set(y_true) | set(y_pred))

    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    n_runs = len({r["_source"] for r in rows})
    print(f"Aggregate over {n_runs} run(s), {total} total predictions ({total // n_runs} passages each)\n")
    print(f"Overall accuracy: {correct}/{total} = {correct/total:.1%}\n")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    col_width = max(len(l) for l in all_labels) + 2
    print("Confusion matrix (rows=true, cols=predicted):")
    print(" " * col_width + "".join(l.ljust(col_width) for l in all_labels))
    for label, row_vals in zip(all_labels, cm):
        print(label.ljust(col_width) + "".join(str(v).ljust(col_width) for v in row_vals))
    print()


def print_stats_l1(rows: list[dict]):
    """Print L1 (junk/non-junk gate) accuracy stats."""
    from sklearn.metrics import classification_report, confusion_matrix

    l1_rows = [r for r in rows if "l1_tag" in r]
    if not l1_rows:
        print("No l1_tag column found — skipping L1 stats.")
        return

    y_true = ["junk" if r["correct_tag"] == "junk" else "non_junk" for r in l1_rows]
    y_pred = [r["l1_tag"].strip().lower() for r in l1_rows]
    all_labels = ["junk", "non_junk"]

    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    n_runs = len({r["_source"] for r in l1_rows})
    print(f"=== L1 Gate Analysis (junk/non-junk) ===")
    print(f"Aggregate over {n_runs} run(s), {total} predictions\n")
    print(f"L1 accuracy: {correct}/{total} = {correct/total:.1%}\n")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))
    print()


def print_stats_l2(rows: list[dict]):
    """Print L2 (category) accuracy stats, restricted to non-junk passages."""
    from sklearn.metrics import classification_report, confusion_matrix

    l2_rows = [r for r in rows if r["correct_tag"] != "junk"]
    if not l2_rows:
        print("No non-junk passages found — skipping L2 stats.")
        return

    y_true = [r["correct_tag"] for r in l2_rows]
    y_pred = [r["predicted_tag"].strip().lower() for r in l2_rows]
    all_labels = sorted(set(y_true) | set(y_pred))

    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    n_runs = len({r["_source"] for r in l2_rows})
    print(f"=== L2 Category Analysis (non-junk passages only) ===")
    print(f"Aggregate over {n_runs} run(s), {total} predictions ({total // n_runs} passages each)\n")
    print(f"L2 accuracy: {correct}/{total} = {correct/total:.1%}\n")
    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))
    print()


def print_report(records: list[dict], label: str = ""):
    header = f"Top {min(TOP_N, len([r for r in records if r['mislabeled_count'] > 0]))} most-mislabeled passages"
    if label:
        header += f" ({label})"
    top = [r for r in records if r["mislabeled_count"] > 0][:TOP_N]
    if not top:
        print(f"No mislabeled passages found{' (' + label + ')' if label else ''}.")
        return

    print(header + "\n" + "=" * 60)
    for i, r in enumerate(top, 1):
        snippet = r["text"][:120].replace("\n", " ")
        if len(r["text"]) > 120:
            snippet += "..."

        wrong_counts = {
            l: r[f"{l}_n"] for l in ALL_LABELS
            if f"{l}_n" in r and r[f"{l}_n"] > 0 and l != r["correct_tag"]
        }
        wrong_str = ", ".join(f"{l} ×{n}" for l, n in sorted(wrong_counts.items(), key=lambda x: -x[1]))

        print(f"\n{i}. [mislabeled {r['mislabeled_count']}/{r['total_runs']} runs]  correct: {r['correct_tag']}")
        print(f"   \"{snippet}\"")
        if r["your_rationale"]:
            print(f"   rationale: {r['your_rationale']}")
        print(f"   predicted as: {wrong_str or '—'}")

    print()


def write_analysis(records: list[dict], out_dir: Path = None, filename: str = "error_analysis.csv"):
    out_dir = (out_dir or RESULTS_DIR) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    all_cols = ["text", "correct_tag", "your_rationale", "total_runs", "mislabeled_count"] + [f"{l}_n" for l in ALL_LABELS]
    # Only write columns that exist in the records
    sample = records[0] if records else {}
    fieldnames = [c for c in all_cols if c in sample or c in ("text", "correct_tag", "your_rationale", "total_runs", "mislabeled_count")]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"Analysis written to: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate eval result CSVs and report error analysis")
    parser.add_argument("--run-dir",     type=str,  default=None,  help="Subdirectory of eval/results/")
    parser.add_argument("--output-name", type=str,  default="error_analysis.csv")
    parser.add_argument("--no-csv",      action="store_true",      help="Print only, do not write CSV")
    parser.add_argument("--two-layer",   action="store_true",      help="Also produce L1 and L2 separate analyses")
    parser.add_argument("files",         nargs="*", type=Path,     help="Explicit result CSV files")
    args = parser.parse_args()

    run_dir = RESULTS_DIR / args.run_dir if args.run_dir else None

    if args.files:
        rows = load_results(args.files)
    elif run_dir:
        rows = load_results([f for f in run_dir.glob("*.csv") if f.parent == run_dir])
    else:
        rows = load_results(None)

    # --- Overall stats ---
    print_stats(rows)
    records = aggregate(rows)
    print_report(records)
    if not args.no_csv:
        write_analysis(records, out_dir=run_dir, filename=args.output_name)

    # --- Two-layer extra analyses ---
    if args.two_layer:
        has_l1 = any("l1_tag" in r for r in rows)
        if not has_l1:
            print("Warning: --two-layer specified but no l1_tag column found in results.")
        else:
            print("\n" + "=" * 60)
            print_stats_l1(rows)
            l1_records = aggregate_l1(rows)
            print_report(l1_records, label="L1 gate")
            if not args.no_csv:
                write_analysis(l1_records, out_dir=run_dir, filename="l1_error_analysis.csv")

            print("\n" + "=" * 60)
            print_stats_l2(rows)
            l2_records = aggregate_l2(rows)
            print_report(l2_records, label="L2 category")
            if not args.no_csv:
                write_analysis(l2_records, out_dir=run_dir, filename="l2_error_analysis.csv")


if __name__ == "__main__":
    main()
