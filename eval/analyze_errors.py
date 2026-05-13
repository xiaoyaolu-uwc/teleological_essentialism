#!/usr/bin/env python3
"""
analyze_errors.py
=================
Aggregates eval result CSVs, counts how often each passage was mislabeled
across runs, and reports the worst offenders.

Usage (from repo root):
    python3 eval/analyze_errors.py                    # globs all of RESULTS_DIR
    python3 eval/analyze_errors.py path/to/a.csv ...  # explicit files

Output:
    CLI: ranked top-N most-mislabeled passages
    File: eval/results/analysis/error_analysis.csv
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
ALL_LABELS  = ["divine_teleology", "internal_essence", "junk", "mixed", "non_divine_teleology", "error"]
# ---------------------------------------------------------------------------


def load_results(files: list[Path] = None) -> list[dict]:
    """Read result CSVs and return a flat list of rows.

    If files is None or empty, globs all CSVs directly in RESULTS_DIR
    (skipping the analysis subdir). Otherwise reads exactly the given files.
    """
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
        "total_runs":       0,
        "mislabeled_count": 0,
        **{f"{l}_n": 0 for l in ALL_LABELS},
    })

    for row in rows:
        text = row["text"]
        entry = by_text[text]
        entry["correct_tag"]  = row["correct_tag"]
        entry["total_runs"]  += 1
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


def print_report(records: list[dict]):
    top = [r for r in records if r["mislabeled_count"] > 0][:TOP_N]
    if not top:
        print("No mislabeled passages found across all runs.")
        return

    print(f"Top {min(TOP_N, len(top))} most-mislabeled passages\n" + "=" * 60)
    for i, r in enumerate(top, 1):
        snippet = r["text"][:120].replace("\n", " ")
        if len(r["text"]) > 120:
            snippet += "..."

        wrong_counts = {
            l: r[f"{l}_n"] for l in ALL_LABELS
            if r[f"{l}_n"] > 0 and l != r["correct_tag"]
        }
        wrong_str = ", ".join(f"{l} ×{n}" for l, n in sorted(wrong_counts.items(), key=lambda x: -x[1]))

        print(f"\n{i}. [mislabeled {r['mislabeled_count']}/{r['total_runs']} runs]  correct: {r['correct_tag']}")
        print(f"   \"{snippet}\"")
        print(f"   predicted as: {wrong_str or '—'}")

    print()


def write_analysis(records: list[dict]):
    out_dir = RESULTS_DIR / "analysis"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "error_analysis.csv"

    fieldnames = ["text", "correct_tag", "total_runs", "mislabeled_count"] + [f"{l}_n" for l in ALL_LABELS]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Analysis written to: {out_path}")


def main():
    explicit = [Path(a) for a in sys.argv[1:]]
    rows    = load_results(explicit or None)
    records = aggregate(rows)
    print_report(records)
    write_analysis(records)


if __name__ == "__main__":
    main()
