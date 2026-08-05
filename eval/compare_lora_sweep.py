#!/usr/bin/env python3
"""
compare_lora_sweep.py
======================
Aggregates every models/lora_gate/train.py run's metrics.json into one
table, sorted by held-out non_junk recall (the primary metric for the
sweep -- see LORA_JUNK_GATE_PLAN.md and the discussion of why golden's 30
non-junk rows are too noisy for single-run comparisons). Doesn't train or
predict anything itself; each run already wrote its own metrics.json, this
just reads them back and lines them up so a winner is visible at a glance.

Usage:
    python3 eval/compare_lora_sweep.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS

# BERT's measured baseline on this same holdout (Reign of Law), for reference
# -- see LORA_JUNK_GATE_PLAN.md section 9.
BERT_BASELINE = {"held_out_non_junk_recall": 0.467, "golden_non_junk_recall": 0.600}


def main():
    results_dir = PATHS["lora_results_dir"]
    run_dirs = sorted(results_dir.glob("*/metrics.json"))
    if not run_dirs:
        print(f"No runs found under {results_dir}")
        return

    rows = []
    for path in run_dirs:
        d = json.load(open(path))
        held = d["holdout_text_metrics"]["per_class"]["non_junk"]
        golden = d["golden_eval_metrics"]["per_class"]["non_junk"]
        rows.append({
            "run_name": d["run_name"],
            "lr": d["lr"],
            "lora_r": d["lora_r"],
            "lora_alpha": d["lora_alpha"],
            "target_modules": d.get("target_modules", "?"),
            "oversample": d.get("oversample", "?"),
            "seed": d.get("seed", "?"),
            "held_acc": d["holdout_text_metrics"]["accuracy"],
            "held_macro_f1": d["holdout_text_metrics"]["macro_f1"],
            "held_non_junk_recall": held["recall"],
            "golden_acc": d["golden_eval_metrics"]["accuracy"],
            "golden_macro_f1": d["golden_eval_metrics"]["macro_f1"],
            "golden_non_junk_recall": golden["recall"],
        })

    rows.sort(key=lambda r: r["held_non_junk_recall"], reverse=True)

    header = (f"{'run_name':32s} {'lr':>8s} {'r':>3s} {'alpha':>5s} {'target':>8s} "
              f"{'oversmp':>7s} {'seed':>4s} | {'held_recall':>11s} {'held_f1':>7s} "
              f"{'gold_recall':>11s} {'gold_f1':>7s}")
    print(header)
    print("-" * len(header))
    print(f"{'BERT baseline (Reign of Law)':32s} {'':>8s} {'':>3s} {'':>5s} {'':>8s} "
          f"{'':>7s} {'':>4s} | {BERT_BASELINE['held_out_non_junk_recall']:>11.4f} {'':>7s} "
          f"{BERT_BASELINE['golden_non_junk_recall']:>11.4f} {'':>7s}")
    print("-" * len(header))
    for r in rows:
        print(f"{r['run_name']:32s} {r['lr']:>8.0e} {r['lora_r']:>3d} {r['lora_alpha']:>5d} "
              f"{r['target_modules']:>8s} {str(r['oversample']):>7s} {r['seed']:>4} | "
              f"{r['held_non_junk_recall']:>11.4f} {r['held_macro_f1']:>7.4f} "
              f"{r['golden_non_junk_recall']:>11.4f} {r['golden_macro_f1']:>7.4f}")

    best = rows[0]
    print(f"\nBest by held-out non_junk recall: {best['run_name']} "
          f"({best['held_non_junk_recall']:.4f}, vs BERT's {BERT_BASELINE['held_out_non_junk_recall']:.4f})")


if __name__ == "__main__":
    main()
