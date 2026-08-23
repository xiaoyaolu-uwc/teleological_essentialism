#!/usr/bin/env python3
"""
evaluate_proportions.py
=======================
The proportion evaluation. Consumes the per-row prediction file written by
eval/run_cascade_folds.py and produces every number the blog post needs.

What is being estimated, precisely: for an unseen text, the share of its
*explanatory* (non-junk) sentences falling in each of DT / NDT / IE. Junk is
excluded from both the true and predicted mix so the two sit on the same
basis; the junk rate is reported separately as a data-quality figure.

Why the unit of analysis is the WORK: each held-out book is one observation
of "how far off is the mix". With 6 folds covering all 16 works, that gives
16 observations. The spread of those 16 signed errors IS the error bar --
no resampling involved. Sampling noise within a book is small next to
classifier bias, and classifier bias does not shrink as rows are added, so
bootstrapping rows would measure the wrong thing (see the plan of record).

Ranking claims are reported as observed rate + Wilson 95% interval. With 16
works we deliberately do NOT claim "95% confident above gap X" -- we report
what was observed and how wide the interval on it is.

Stage attribution, all from one inference pass:
  end_to_end      gate decides; survivors take stage 2's label
  perfect_gate    true non-junk rows take stage 2's label
  perfect_stage2  gate decides; survivors credited to their TRUE label

Usage:
    python3 eval/evaluate_proportions.py \
        --per-row eval/results/proportions/per_row_predictions.csv \
        --out-dir eval/results/proportions
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

CATS = ["divine_teleology", "non_divine_teleology", "internal_essence"]
SHORT = {"divine_teleology": "DT", "non_divine_teleology": "NDT", "internal_essence": "IE"}
GAP_BINS = [(0.0, 0.025), (0.025, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a binomial rate. Used instead of the
    normal approximation because several of these cells have small n or
    rates at the 0/1 boundary, where the normal interval is nonsense."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mix(tags):
    """Category shares among non-junk tags. None if there are no non-junk rows."""
    c = {k: 0 for k in CATS}
    for t in tags:
        if t in c:
            c[t] += 1
    n = sum(c.values())
    return ({k: c[k] / n for k in CATS} if n else None), n


def variant_labels(rows, variant):
    """Per-row label under one stage-attribution variant. None means the row
    is dropped (predicted junk / truly junk), exactly as deployment drops it."""
    out = []
    for r in rows:
        true_tag, gate, s2 = r["true_tag"], r["gate_pred"], r["s2_pred"]
        if variant == "end_to_end":
            # A full4way stage 2 can itself say "junk"; that is the point of
            # giving it a junk option, so honour it.
            out.append(None if gate == "junk" or s2 == "junk" else s2)
        elif variant == "perfect_gate":
            out.append(None if true_tag == "junk" or s2 == "junk" else s2)
        elif variant == "perfect_stage2":
            out.append(None if gate == "junk" or true_tag == "junk" else true_tag)
        else:
            raise ValueError(variant)
    return out


def per_work_errors(by_work, variant):
    """work -> {true_mix, pred_mix, signed_error, ...} for one variant."""
    out = {}
    for work, rows in by_work.items():
        true_mix, true_n = mix([r["true_tag"] for r in rows])
        pred_mix, pred_n = mix([l for l in variant_labels(rows, variant) if l])
        if not true_mix or not pred_mix:
            continue
        signed = {c: pred_mix[c] - true_mix[c] for c in CATS}
        tel_t = true_mix["divine_teleology"] + true_mix["non_divine_teleology"]
        tel_p = pred_mix["divine_teleology"] + pred_mix["non_divine_teleology"]
        out[work] = {
            "year": rows[0]["year"], "n_rows": len(rows),
            "true_n_nonjunk": true_n, "pred_n_kept": pred_n,
            "true_mix": true_mix, "pred_mix": pred_mix, "signed_error": signed,
            "max_abs_error": max(abs(v) for v in signed.values()),
            "tvd": 0.5 * sum(abs(v) for v in signed.values()),
            "teleology_true": tel_t, "teleology_pred": tel_p,
            "teleology_signed_error": tel_p - tel_t,
        }
    return out


def bias_table(pw):
    """Mean (bias) and sd (error bar) of signed error per category, over works."""
    out = {}
    for c in CATS + ["teleology"]:
        key = "teleology_signed_error" if c == "teleology" else None
        vals = [(v[key] if key else v["signed_error"][c]) for v in pw.values()]
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[c] = {
            "mean_signed_error": statistics.fmean(vals),
            "sd": sd,
            "mean_abs_error": statistics.fmean(abs(v) for v in vals),
            "max_abs_error": max(abs(v) for v in vals),
            "n_works": len(vals),
            # Interval for a NEW book, not for the mean -- this is a
            # prediction interval, which is what the blog post needs.
            "predict_90pct": 1.645 * sd,
        }
    return out


def binned(observations):
    """observations: list of (gap, correct_bool) -> per-bin rate + Wilson CI."""
    out = []
    for lo, hi in GAP_BINS:
        sel = [c for g, c in observations if lo <= g < hi]
        if not sel:
            continue
        k, n = sum(sel), len(sel)
        lo_ci, hi_ci = wilson(k, n)
        out.append({"gap_lo": lo, "gap_hi": hi, "n": n, "correct": k,
                    "rate": k / n, "ci_lo": lo_ci, "ci_hi": hi_ci})
    return out


def within_work_ranking(pw):
    """Does the DT/NDT/IE ordering inside a book come out right?"""
    full_ok, pairs = 0, []
    for v in pw.values():
        t, p = v["true_mix"], v["pred_mix"]
        if sorted(CATS, key=lambda c: -t[c]) == sorted(CATS, key=lambda c: -p[c]):
            full_ok += 1
        for a, b in combinations(CATS, 2):
            gap = abs(t[a] - t[b])
            correct = (t[a] > t[b]) == (p[a] > p[b])
            pairs.append((gap, correct))
    k, n = full_ok, len(pw)
    return {"full_order_correct": k, "n_works": n, "rate": k / n if n else None,
            "ci": wilson(k, n), "pairwise_by_gap": binned(pairs),
            "pairwise_overall": {"correct": sum(c for _, c in pairs), "n": len(pairs),
                                 "rate": sum(c for _, c in pairs) / len(pairs),
                                 "ci": wilson(sum(c for _, c in pairs), len(pairs))}}


def across_work_ranking(pw):
    """For every pair of books and every category, is the ordering right?
    This is the evidence for decade-to-decade comparisons, and it is
    CONSERVATIVE: a real decade bucket pools ~10 books, so per-book errors
    partly cancel there but not here."""
    works = sorted(pw)
    per_cat, allobs = {}, []
    for c in CATS + ["teleology"]:
        obs = []
        for a, b in combinations(works, 2):
            ta = pw[a]["teleology_true"] if c == "teleology" else pw[a]["true_mix"][c]
            tb = pw[b]["teleology_true"] if c == "teleology" else pw[b]["true_mix"][c]
            pa = pw[a]["teleology_pred"] if c == "teleology" else pw[a]["pred_mix"][c]
            pb = pw[b]["teleology_pred"] if c == "teleology" else pw[b]["pred_mix"][c]
            if ta == tb:
                continue
            obs.append((abs(ta - tb), (ta > tb) == (pa > pb)))
        k, n = sum(c_ for _, c_ in obs), len(obs)
        per_cat[c] = {"correct": k, "n": n, "rate": k / n if n else None,
                      "ci": wilson(k, n), "by_gap": binned(obs)}
        if c != "teleology":
            allobs += obs
    k, n = sum(c_ for _, c_ in allobs), len(allobs)
    per_cat["_pooled_3cat"] = {"correct": k, "n": n, "rate": k / n if n else None,
                               "ci": wilson(k, n), "by_gap": binned(allobs)}
    return per_cat


def fmt_pct(x):
    return "n/a" if x is None else f"{100*x:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-row", default="eval/results/proportions/per_row_predictions.csv")
    ap.add_argument("--out-dir", default="eval/results/proportions")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.per_row, newline="", encoding="utf-8")))
    by_work = defaultdict(list)
    for r in rows:
        by_work[r["work"]].append(r)

    report = {"n_rows": len(rows), "n_works": len(by_work), "variants": {}}
    for variant in ["end_to_end", "perfect_gate", "perfect_stage2"]:
        pw = per_work_errors(by_work, variant)
        report["variants"][variant] = {
            "per_work": pw,
            "bias": bias_table(pw),
            "mean_tvd": statistics.fmean(v["tvd"] for v in pw.values()),
            "max_tvd": max(v["tvd"] for v in pw.values()),
        }
        if variant == "end_to_end":
            report["variants"][variant]["within_work_ranking"] = within_work_ranking(pw)
            report["variants"][variant]["across_work_ranking"] = across_work_ranking(pw)

    # junk rate, reported separately as data quality rather than folded into the mix
    n_true_junk = sum(1 for r in rows if r["true_tag"] == "junk")
    kept_junk = sum(1 for r in rows if r["true_tag"] == "junk" and r["gate_pred"] == "non_junk"
                    and r["s2_pred"] != "junk")
    kept = sum(1 for r in rows if r["gate_pred"] == "non_junk" and r["s2_pred"] != "junk")
    report["junk"] = {"true_junk_rate": n_true_junk / len(rows),
                      "leakage_into_kept": kept_junk / kept if kept else None,
                      "kept_n": kept}

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out_dir / "proportion_metrics.json", "w"), indent=2)

    # ---------------- console summary ----------------
    e2e = report["variants"]["end_to_end"]
    print("=" * 78)
    print(f"PROPORTION EVALUATION -- {report['n_works']} held-out works, {report['n_rows']} rows")
    print("=" * 78)
    print(f"\nTrue junk rate {100*report['junk']['true_junk_rate']:.1f}%   "
          f"junk leaking into kept set {100*report['junk']['leakage_into_kept']:.1f}%\n")

    print("PER-WORK MIX ERROR (end-to-end, percentage points)")
    print(f"{'work':34s} {'yr':>4s} {'n':>4s} | {'DT t/p':>12s} {'NDT t/p':>12s} {'IE t/p':>12s} | {'maxErr':>6s}")
    for w, v in sorted(e2e["per_work"].items(), key=lambda kv: kv[1]["year"]):
        t, p = v["true_mix"], v["pred_mix"]
        print(f"{w[:34]:34s} {v['year']:>4s} {v['true_n_nonjunk']:4d} | "
              + " ".join(f"{fmt_pct(t[c])}/{fmt_pct(p[c])}" for c in CATS)
              + f" | {100*v['max_abs_error']:5.1f}")

    print("\nBIAS AND ERROR BAR (signed error = predicted - true share)")
    print(f"{'category':14s} {'bias':>7s} {'sd':>7s} {'meanAbs':>8s} {'worst':>7s} {'+/-90%':>7s}")
    for c, m in e2e["bias"].items():
        print(f"{SHORT.get(c,c):14s} {100*m['mean_signed_error']:+7.1f} {100*m['sd']:7.1f} "
              f"{100*m['mean_abs_error']:8.1f} {100*m['max_abs_error']:7.1f} {100*m['predict_90pct']:7.1f}")

    print("\nSTAGE ATTRIBUTION (mean TVD across works, percentage points)")
    for v in ["perfect_gate", "perfect_stage2", "end_to_end"]:
        print(f"  {v:16s} {100*report['variants'][v]['mean_tvd']:5.1f}")

    wr = e2e["within_work_ranking"]
    print(f"\nWITHIN-BOOK RANKING: full DT/NDT/IE order correct in "
          f"{wr['full_order_correct']}/{wr['n_works']} books "
          f"({100*wr['rate']:.0f}%, 95% CI {100*wr['ci'][0]:.0f}-{100*wr['ci'][1]:.0f}%)")
    print(f"  pairwise overall {wr['pairwise_overall']['correct']}/{wr['pairwise_overall']['n']} "
          f"({100*wr['pairwise_overall']['rate']:.0f}%)")
    print(f"  {'true gap':>14s} {'n':>4s} {'correct':>8s} {'rate':>6s}  95% CI")
    for b in wr["pairwise_by_gap"]:
        print(f"  {100*b['gap_lo']:5.1f}-{100*b['gap_hi']:5.1f}pp {b['n']:4d} {b['correct']:8d} "
              f"{100*b['rate']:5.0f}%  {100*b['ci_lo']:.0f}-{100*b['ci_hi']:.0f}%")

    ar = e2e["across_work_ranking"]
    print("\nACROSS-BOOK RANKING (same category, two books -- basis for decade comparisons)")
    for c in CATS + ["teleology", "_pooled_3cat"]:
        m = ar[c]
        print(f"  {SHORT.get(c,c):14s} {m['correct']:4d}/{m['n']:<4d} {100*m['rate']:5.0f}%  "
              f"95% CI {100*m['ci'][0]:.0f}-{100*m['ci'][1]:.0f}%")
    print(f"\n  pooled 3-category, by true gap:")
    print(f"  {'true gap':>14s} {'n':>4s} {'correct':>8s} {'rate':>6s}  95% CI")
    for b in ar["_pooled_3cat"]["by_gap"]:
        print(f"  {100*b['gap_lo']:5.1f}-{100*b['gap_hi']:5.1f}pp {b['n']:4d} {b['correct']:8d} "
              f"{100*b['rate']:5.0f}%  {100*b['ci_lo']:.0f}-{100*b['ci_hi']:.0f}%")
    print(f"\nwrote {out_dir / 'proportion_metrics.json'}")


if __name__ == "__main__":
    main()
