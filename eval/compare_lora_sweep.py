#!/usr/bin/env python3
"""
compare_lora_sweep.py
======================
Aggregates every models/lora_gate/train.py run's metrics.json into one
table. Sorted by the WORST per-category (DT/NDT/IE) recall on held-out text
-- not pooled non_junk recall -- per the project's actual objective: what
matters is that every non-junk category survives the gate at a high AND
even rate (so downstream category proportions come out right), not that
the pooled average looks good while one category (historically IE) is
quietly being thrown away. See models/data_utils.py's gate_proportion_metrics
and LORA_JUNK_GATE_PLAN.md for why.

Runs trained before gate_proportion_metrics existed won't have
holdout_proportion_metrics/golden_proportion_metrics in their metrics.json
-- run eval/backfill_gate_proportions.py first to add it (re-evaluates the
saved checkpoint, no retraining) or those columns show as "N/A" here.

Doesn't train or predict anything itself; each run already wrote its own
metrics.json, this just reads them back and lines them up.

Usage:
    python3 eval/compare_lora_sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS

# BERT's measured baseline on this same holdout (Reign of Law) -- see
# LORA_JUNK_GATE_PLAN.md section 9 and the gate_proportion_metrics run
# against eval/results/bert_cascade/loo_reign_of_law/metrics.json.
BERT_BASELINE = {
    "held_out_non_junk_recall": 0.467,
    "golden_non_junk_recall": 0.600,
    "held_per_category_recall": {"divine_teleology": 0.483, "non_divine_teleology": 0.496, "internal_essence": 0.333},
    "held_recall_evenness": 0.162,
    "golden_per_category_recall": {"divine_teleology": 0.400, "non_divine_teleology": 0.714, "internal_essence": 0.667},
    "golden_recall_evenness": 0.314,
}

CATS = ["divine_teleology", "non_divine_teleology", "internal_essence"]
CAT_ABBR = {"divine_teleology": "DT", "non_divine_teleology": "NDT", "internal_essence": "IE"}


def fmt_cats(per_category_recall):
    # "DT=0.48 NDT=0.48 IE=0.48" is 24 chars -- match that width when missing
    # so table columns stay aligned regardless of which rows have data.
    if per_category_recall is None:
        return "N/A".center(24)
    return " ".join(f"{CAT_ABBR[c]}={per_category_recall[c]:.2f}" for c in CATS)


def main():
    results_dir = PATHS["lora_results_dir"]
    run_dirs = sorted(results_dir.glob("*/metrics.json"))
    if not run_dirs:
        print(f"No runs found under {results_dir}")
        return

    rows = []
    for path in run_dirs:
        d = json.load(open(path))
        held_prop = d.get("holdout_proportion_metrics")
        golden_prop = d.get("golden_proportion_metrics")
        rows.append({
            "run_name": d["run_name"],
            "lr": d["lr"],
            "lora_r": d["lora_r"],
            "lora_alpha": d["lora_alpha"],
            "target_modules": d.get("target_modules", "?"),
            "oversample": d.get("oversample", "?"),
            "prompt_variant": d.get("prompt_variant", "current"),
            "held_per_cat": held_prop["per_category_recall"] if held_prop else None,
            "held_evenness": held_prop["recall_evenness"] if held_prop else None,
            "held_min_recall": min(held_prop["per_category_recall"].values()) if held_prop else -1,
            "held_non_junk_precision": d["holdout_text_metrics"].get("per_class", {}).get("junk", {}).get("precision"),
            "golden_per_cat": golden_prop["per_category_recall"] if golden_prop else None,
            "golden_evenness": golden_prop["recall_evenness"] if golden_prop else None,
        })

    # Primary sort: worst-case per-category recall on held-out, descending --
    # rewards both "high recall" and "evenness" at once (a run can't win this
    # by being high on two categories and abandoning the third).
    rows.sort(key=lambda r: r["held_min_recall"], reverse=True)

    print(f"{'run_name':34s} {'lr':>8s} {'r':>3s} {'tgt':>8s} {'ovspl':>6s} {'prompt':>14s} | "
          f"{'held-out DT/NDT/IE recall':28s} {'even':>5s} | {'golden DT/NDT/IE recall':28s} {'even':>5s}")
    print("-" * 145)
    print(f"{'BERT baseline (Reign of Law)':34s} {'':>8s} {'':>3s} {'':>8s} {'':>6s} {'':>14s} | "
          f"{fmt_cats(BERT_BASELINE['held_per_category_recall']):28s} "
          f"{BERT_BASELINE['held_recall_evenness']:>5.2f} | "
          f"{fmt_cats(BERT_BASELINE['golden_per_category_recall']):28s} "
          f"{BERT_BASELINE['golden_recall_evenness']:>5.2f}")
    print("-" * 145)
    for r in rows:
        held_even = f"{r['held_evenness']:.2f}" if r["held_evenness"] is not None else "N/A"
        gold_even = f"{r['golden_evenness']:.2f}" if r["golden_evenness"] is not None else "N/A"
        print(f"{r['run_name']:34s} {r['lr']:>8.0e} {r['lora_r']:>3d} {r['target_modules']:>8s} "
              f"{str(r['oversample']):>6s} {r['prompt_variant']:>14s} | "
              f"{fmt_cats(r['held_per_cat']):28s} {held_even:>5s} | "
              f"{fmt_cats(r['golden_per_cat']):28s} {gold_even:>5s}")

    scored = [r for r in rows if r["held_min_recall"] >= 0]
    if scored:
        best = scored[0]
        print(f"\nBest by worst-case held-out per-category recall: {best['run_name']} "
              f"(min={best['held_min_recall']:.3f})")
    missing = [r["run_name"] for r in rows if r["held_min_recall"] < 0]
    if missing:
        print(f"\nNo proportion metrics yet (run eval/backfill_gate_proportions.py): {', '.join(missing)}")


if __name__ == "__main__":
    main()
