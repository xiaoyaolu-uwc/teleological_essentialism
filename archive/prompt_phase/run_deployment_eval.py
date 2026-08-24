#!/usr/bin/env python3
"""
run_deployment_eval.py
======================
Checkpointing version of evaluate_deployment.py.
Writes results row-by-row to a CSV as they arrive.
Safe to interrupt and resume — already-written rows are skipped on restart.

Usage:
    python3 eval/run_deployment_eval.py --version d_v1 --model gpt-5.4 \
        --out eval/results/deployment/d_v1/eval_gpt-5.4_d_v1.csv
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import PATHS
from archive.prompt_phase.deployment import DeploymentModel


def load_dotenv():
    env = PATHS["env_file"]
    if env.exists():
        with open(env) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())


load_dotenv()

FIELDNAMES = [
    "text", "correct_tag", "your_rationale",
    "predicted_tag", "extract", "model_reasoning", "confidence", "correct",
]


def load_eval_set():
    with open(PATHS["evaluation_csv"], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done(out_path: Path) -> int:
    """Return number of rows already written (0 if file doesn't exist)."""
    if not out_path.exists():
        return 0
    with open(out_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version",    type=str, default="d_v1")
    parser.add_argument("--model",      type=str, default="gpt-5.4")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--out",        type=str, required=True,
                        help="Output CSV path")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_eval_set()
    done = already_done(out_path)
    remaining = rows[done:]

    if done:
        print(f"Resuming from row {done} ({len(remaining)} remaining)...")
    else:
        print(f"Starting fresh ({len(rows)} rows)...")

    model = DeploymentModel(
        model_name=args.model,
        prompt_version=args.version,
        batch_size=args.batch_size,
    )

    # Open in append mode
    write_header = not out_path.exists() or done == 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for batch_start in range(0, len(remaining), args.batch_size):
            batch = remaining[batch_start: batch_start + args.batch_size]
            texts = [r["text"] for r in batch]
            global_start = done + batch_start

            print(f"  Rows {global_start+1}–{global_start+len(batch)}/{len(rows)}...", flush=True)
            preds = model.classify(texts)

            for row, pred in zip(batch, preds):
                writer.writerow({
                    "text":            row["text"],
                    "correct_tag":     row["correct_tag"],
                    "your_rationale":  row.get("xy_rationale", ""),
                    "predicted_tag":   pred["tag"],
                    "extract":         pred["extract"],
                    "model_reasoning": pred["reasoning"],
                    "confidence":      pred["confidence"],
                    "correct":         str(pred["tag"] == row["correct_tag"]),
                })
            f.flush()
            print(f"    → written. Tags: {[p['tag'] for p in preds]}", flush=True)

    print(f"\nDone. Output: {out_path}")

    # Quick accuracy summary
    with open(out_path, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    correct = sum(r["correct"] == "True" for r in all_rows)
    print(f"Accuracy so far: {correct}/{len(all_rows)} = {correct/len(all_rows):.1%}")


if __name__ == "__main__":
    main()
