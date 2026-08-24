#!/usr/bin/env python3
"""
calibrate_gate_threshold.py
===========================
Free improvement, no retraining: the gate's decision threshold is currently
0.5 by default. Junk leakage into the kept set is the dominant remaining
source of proportion error (see docs/PROPORTION_EVAL_RESULTS.md), and raising
the threshold trades some recall for purity.

The honest way to measure this: NESTED selection. For each fold, the threshold
is chosen on the OTHER five folds' books and then applied to this fold's books.
Nothing is ever scored at a threshold tuned on itself, so the reported numbers
stay genuinely out-of-sample. Tuning on all 16 books and reporting on the same
16 would be circular and would overstate the gain.

Selection criterion is mean per-book TVD, the same quantity the evaluation
reports -- not accuracy, and not gate precision on its own.

Usage:
    python3 evaluation/calibrate_gate_threshold.py
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

CATS = ["divine_teleology", "non_divine_teleology", "internal_essence"]


def mix(tags):
    c = {k: 0 for k in CATS}
    for t in tags:
        if t in c:
            c[t] += 1
    n = sum(c.values())
    return ({k: c[k] / n for k in CATS} if n else None)


def book_tvd(rows, thr):
    """TVD between true and predicted mix for one book at gate threshold thr."""
    true_mix = mix([r["true_tag"] for r in rows])
    pred = [r["s2_pred"] for r in rows
            if float(r["gate_prob_nonjunk"]) >= thr and r["s2_pred"] != "junk"]
    pred_mix = mix(pred)
    if not true_mix or not pred_mix:
        return None
    return 0.5 * sum(abs(pred_mix[c] - true_mix[c]) for c in CATS)


def mean_tvd(books, thr):
    vals = [t for t in (book_tvd(rs, thr) for rs in books.values()) if t is not None]
    return statistics.fmean(vals) if vals else None


def gate_stats(rows, thr):
    kept = [r for r in rows if float(r["gate_prob_nonjunk"]) >= thr]
    leak = [r for r in kept if r["true_tag"] == "junk"]
    nonjunk = [r for r in rows if r["true_tag"] != "junk"]
    surv = [r for r in nonjunk if float(r["gate_prob_nonjunk"]) >= thr]
    return {"precision": 1 - len(leak) / len(kept) if kept else None,
            "recall": len(surv) / len(nonjunk) if nonjunk else None,
            "kept": len(kept)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-row", default="evaluation/results/proportions/per_row_predictions.csv")
    ap.add_argument("--out", default="evaluation/results/proportions/threshold_calibration.json")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.per_row, newline="", encoding="utf-8")))
    by_fold_book = defaultdict(dict)
    for r in rows:
        by_fold_book[r["fold"]].setdefault(r["work"], []).append(r)
    folds = sorted(by_fold_book)
    grid = [i / 100 for i in range(30, 91, 2)]

    # --- reference: what a single global threshold would look like (in-sample,
    # shown only to make the nested number's honesty visible by contrast) ---
    all_books = {w: rs for f in folds for w, rs in by_fold_book[f].items()}
    curve = [(t, mean_tvd(all_books, t)) for t in grid]
    best_insample = min((c for c in curve if c[1] is not None), key=lambda c: c[1])

    # --- nested: threshold chosen on the other five folds ---------------------
    per_fold, applied = {}, {}
    for f in folds:
        others = {w: rs for g in folds if g != f for w, rs in by_fold_book[g].items()}
        scored = [(t, mean_tvd(others, t)) for t in grid]
        thr = min((s for s in scored if s[1] is not None), key=lambda s: s[1])[0]
        per_fold[f] = thr
        for w, rs in by_fold_book[f].items():
            applied[w] = (thr, book_tvd(rs, thr))

    base = {w: book_tvd(rs, 0.5) for w, rs in all_books.items()}
    base_mean = statistics.fmean(v for v in base.values() if v is not None)
    nested_mean = statistics.fmean(v for _, v in applied.values() if v is not None)

    g05 = gate_stats(rows, 0.5)
    gN = [gate_stats(sum(by_fold_book[f].values(), []), per_fold[f]) for f in folds]

    print("Threshold sweep (all books pooled -- IN-SAMPLE, reference only)")
    print(f"{'thr':>5s} {'meanTVD':>8s}")
    for t, v in curve:
        if v is not None and abs(round(t * 100) % 10) < 1e-9:
            print(f"{t:5.2f} {v:8.4f}")
    print(f"  in-sample best: thr={best_insample[0]:.2f} meanTVD={best_insample[1]:.4f}")

    print("\nNested selection (threshold picked on the other 5 folds)")
    for f in folds:
        print(f"  {f}: thr={per_fold[f]:.2f}")
    print(f"\n  baseline thr=0.50        mean per-book TVD = {base_mean:.4f}   "
          f"gate precision={g05['precision']:.3f} recall={g05['recall']:.3f}")
    print(f"  nested calibrated        mean per-book TVD = {nested_mean:.4f}   "
          f"gate precision={statistics.fmean(g['precision'] for g in gN):.3f} "
          f"recall={statistics.fmean(g['recall'] for g in gN):.3f}")
    delta = base_mean - nested_mean
    print(f"\n  change: {100*delta:+.2f} pp TVD "
          f"({'improvement' if delta > 0 else 'REGRESSION -- do not adopt'})")
    improved = sum(1 for w in base if applied[w][1] is not None and base[w] is not None
                   and applied[w][1] < base[w])
    print(f"  books improved: {improved}/{len(base)}")

    out = {"grid": curve, "in_sample_best": best_insample, "nested_thresholds": per_fold,
           "baseline_mean_tvd": base_mean, "nested_mean_tvd": nested_mean,
           "per_book": {w: {"thr": t, "tvd": v, "baseline_tvd": base[w]}
                        for w, (t, v) in applied.items()}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
