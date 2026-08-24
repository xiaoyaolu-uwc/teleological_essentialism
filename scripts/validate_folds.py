#!/usr/bin/env python3
"""
Confirms every fold run finished with the config it was supposed to, and that a
matching checkpoint exists.

Worth running before any inference: the `preempt` partition can requeue a job
after its metrics.json was already written, so the presence of a results file
is NOT proof that a run completed. Check the config fields, not the file.

    python3 scripts/validate_folds.py [--suffix _s7]
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--suffix", default="", help="e.g. _s7 for a second-seed set")
ap.add_argument("--stages", nargs="+", default=["gate", "s2"])
args = ap.parse_args()

folds = json.load(open("evaluation/folds.json"))
expected = {"gate": "junk_gate", "s2": "nonjunk_3way"}
ok = True
for p in args.stages:
    for f in range(6):
        n = f"{p}_fold{f}{args.suffix}"
        mp = f"evaluation/results/lora/{n}/metrics.json"
        cp = f"models/checkpoints/lora/{n}"
        if not os.path.exists(mp):
            print(f"MISSING metrics  {n}"); ok = False; continue
        r = json.load(open(mp))
        good = (r["max_length"] == 640 and r["text_column"] == "text"
                and r["stage"] == expected[p] and r["holdout_work"] == folds[f"fold{f}"]
                and r["epochs_run"] == 4 and os.path.isdir(cp))
        ok = ok and good
        print(f"{'OK  ' if good else 'BAD '} {n:20s} stage={r['stage']:13s} "
              f"ml={r['max_length']} tc={r['text_column']:5s} "
              f"valF1={r['best_val_macro_f1']:.3f} ckpt={os.path.isdir(cp)}")
print("ALL_VALID" if ok else "VALIDATION_FAILED")
sys.exit(0 if ok else 1)
