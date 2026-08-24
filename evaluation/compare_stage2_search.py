#!/usr/bin/env python3
"""
compare_stage2_search.py
========================
Ranks the stage-2 config-search runs (evaluation/results/lora/s2_*) by the metric
that actually decides the choice: how far the predicted DT/NDT/IE *mix* on
unseen text is from the true mix -- not pooled accuracy, which can look
healthy while the mix is skewed.

Two seeds per config are averaged, and the per-seed spread is printed too,
because this project has already been burned once by selecting a config off
seed noise (see docs/history/lora_prompt_evolution.md's multi-seed reliability pass).

Usage:
    python3 evaluation/compare_stage2_search.py [--results-dir evaluation/results/lora]
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DEV_WORK = "The Reign of Law"


def config_key(run_name):
    """s2_<stage>_r<rank>_<prompt>_seed<n> -> everything but the seed."""
    return run_name.rsplit("_seed", 1)[0]


def collect(results_dir):
    runs = defaultdict(list)
    for d in sorted(Path(results_dir).glob("s2_*")):
        f = d / "metrics.json"
        if not f.exists():
            continue
        r = json.load(open(f))
        if "holdout_mix_metrics" not in r:
            continue
        runs[config_key(d.name)].append(r)
    return runs


def summarize(reports):
    """Fold-pooled and dev-work-only mix error, averaged over seeds."""
    out = {"n_seeds": len(reports)}
    for field, getter in [
        ("fold_tvd", lambda r: r["holdout_mix_metrics"]["tvd"]),
        ("fold_max_err", lambda r: r["holdout_mix_metrics"]["max_abs_error"]),
        ("fold_tel_err", lambda r: abs(r["holdout_mix_metrics"]["teleology_signed_error"])),
        ("acc", lambda r: r["holdout_text_metrics"]["accuracy"]),
        ("macro_f1", lambda r: r["holdout_text_metrics"]["macro_f1"]),
    ]:
        vals = [getter(r) for r in reports]
        out[field] = statistics.fmean(vals)
        out[field + "_spread"] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0

    dev = [r["holdout_per_work_mix"][DEV_WORK] for r in reports
           if DEV_WORK in r.get("holdout_per_work_mix", {})]
    if dev:
        out["dev_tvd"] = statistics.fmean(d["tvd"] for d in dev)
        out["dev_max_err"] = statistics.fmean(d["max_abs_error"] for d in dev)
        out["dev_tvd_spread"] = (max(d["tvd"] for d in dev) - min(d["tvd"] for d in dev)) if len(dev) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="evaluation/results/lora")
    args = ap.parse_args()

    runs = collect(args.results_dir)
    if not runs:
        print(f"no s2_* runs with mix metrics under {args.results_dir}")
        return
    rows = [(k, summarize(v)) for k, v in runs.items()]
    rows.sort(key=lambda kv: kv[1].get("dev_tvd", kv[1]["fold_tvd"]))

    hdr = (f"{'config':46s} {'n':>2s} {'devTVD':>7s} {'+/-':>5s} {'devMax':>7s} "
           f"{'foldTVD':>8s} {'foldMax':>8s} {'telErr':>7s} {'acc':>6s} {'mF1':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for k, m in rows:
        print(f"{k[:46]:46s} {m['n_seeds']:2d} "
              f"{m.get('dev_tvd', float('nan')):7.3f} {m.get('dev_tvd_spread', 0):5.3f} "
              f"{m.get('dev_max_err', float('nan')):7.3f} "
              f"{m['fold_tvd']:8.3f} {m['fold_max_err']:8.3f} {m['fold_tel_err']:7.3f} "
              f"{m['acc']:6.3f} {m['macro_f1']:6.3f}")
    print(f"\nSorted by dev_tvd ({DEV_WORK}). MacBERTh stage-2 baseline for reference:")
    print("  Reign of Law  acc=0.775 macro_f1=0.725 | mix err DT +4.7pp NDT -4.8pp IE 0.0pp -> tvd=0.047")
    print("  Darwiniana    acc=0.863 macro_f1=0.864 | mix err DT +1.3pp NDT -8.2pp IE +6.9pp -> tvd=0.082")


if __name__ == "__main__":
    main()
