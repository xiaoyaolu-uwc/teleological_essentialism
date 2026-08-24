#!/usr/bin/env python3
"""
evaluate_deployment.py
======================
Runs a deployment prompt version against the hand-labelled evaluation set.
Outputs a CSV with the full five-field schema plus accuracy/calibration stats.

Usage (from repo root):
    python3 eval/evaluate_deployment.py --version d_v1 --model gpt-5.4 --runs 1 --run-dir deployment/d_v1

Output per run:
    eval/results/<run-dir>/eval_{MODEL}_{VERSION}.csv  (increments if exists)

After all runs (if --runs > 1), writes cross-run analysis.
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import PATHS
from archive.prompt_phase.deployment import DeploymentModel
from archive.prompt_phase.deployment_prompts import DEPLOYMENT_PROMPT_VERSIONS


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

VALID_TAGS = {"divine_teleology", "non_divine_teleology", "internal_essence", "junk"}
LABELS = sorted(VALID_TAGS)


def load_eval_set() -> list[dict]:
    with open(PATHS["evaluation_csv"], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def results_path(model: str, version: str, out_dir: Path) -> Path:
    safe_model = model.replace("/", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"eval_{safe_model}_{version}.csv"
    if not base.exists():
        return base
    run = 2
    while True:
        candidate = out_dir / f"eval_{safe_model}_{version}_{run}.csv"
        if not candidate.exists():
            return candidate
        run += 1


def write_results(rows: list[dict], preds: list[dict], out_path: Path):
    fieldnames = [
        "text", "correct_tag", "your_rationale",
        "predicted_tag", "extract", "model_reasoning", "confidence", "correct",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, pred in zip(rows, preds):
            writer.writerow({
                "text":           row["text"],
                "correct_tag":    row["correct_tag"],
                "your_rationale": row.get("xy_rationale", ""),
                "predicted_tag":  pred["tag"],
                "extract":        pred["extract"],
                "model_reasoning": pred["reasoning"],
                "confidence":     pred["confidence"],
                "correct":        str(pred["tag"] == row["correct_tag"]),
            })


def print_summary(rows: list[dict], preds: list[dict]):
    y_true = [r["correct_tag"] for r in rows]
    y_pred = [p["tag"] for p in preds]
    confs  = [p["confidence"] for p in preds]

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.1%}")

    # Per-class recall
    print("\nPer-class recall:")
    for label in LABELS:
        idxs = [i for i, t in enumerate(y_true) if t == label]
        if not idxs:
            continue
        rec = sum(y_pred[i] == label for i in idxs) / len(idxs)
        print(f"  {label:<25} {rec:.0%}  ({sum(y_pred[i]==label for i in idxs)}/{len(idxs)})")

    # Confidence calibration
    print("\nConfidence calibration:")
    print(f"  {'Bucket':<12} {'N':>4}  {'Avg conf':>9}  {'Accuracy':>9}")
    buckets = [(0.0, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]
    for lo, hi in buckets:
        idxs = [i for i, c in enumerate(confs) if lo <= c < hi]
        if not idxs:
            continue
        avg_c = sum(confs[i] for i in idxs) / len(idxs)
        acc   = sum(y_true[i] == y_pred[i] for i in idxs) / len(idxs)
        label = f"{lo:.0%}–{hi:.0%}" if hi < 1.01 else f"{lo:.0%}–100%"
        print(f"  {label:<12} {len(idxs):>4}  {avg_c:>9.2f}  {acc:>9.0%}")

    # Wrong high-confidence answers (most important to flag)
    wrong_hc = [
        (rows[i]["correct_tag"], preds[i]["tag"], confs[i], rows[i]["text"][:80])
        for i in range(total)
        if y_true[i] != y_pred[i] and confs[i] >= 0.85
    ]
    if wrong_hc:
        print(f"\nWrong + high confidence (≥0.85) — {len(wrong_hc)} cases:")
        for true, pred, c, snippet in wrong_hc:
            print(f"  [{c:.2f}] true={true} pred={pred}  \"{snippet}...\"")
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version",  type=str, default="d_v1",    help="Deployment prompt version")
    parser.add_argument("--model",    type=str, default="gpt-5.4", help="Model name")
    parser.add_argument("--runs",     type=int, default=1,         help="Number of runs (default: 1)")
    parser.add_argument("--run-dir",  type=str, default=None,      help="Subdirectory of eval/results/")
    parser.add_argument("--batch-size", type=int, default=5,       help="Passages per API call (default: 5)")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    base_results = Path(__file__).resolve().parent / "results"
    out_dir = base_results / args.run_dir if args.run_dir else base_results

    model = DeploymentModel(
        model_name=args.model,
        prompt_version=args.version,
        batch_size=args.batch_size,
    )
    print(f"Deployment eval | version={args.version} | model={args.model} | batch={args.batch_size}")

    rows = load_eval_set()
    texts = [r["text"] for r in rows]

    for run in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n--- Run {run}/{args.runs} ---")
        print(f"Evaluating {len(rows)} passages...")
        preds = model.classify(texts)
        out_path = results_path(args.model, args.version, out_dir)
        write_results(rows, preds, out_path)
        print(f"Results written to: {out_path}")
        print_summary(rows, preds)


if __name__ == "__main__":
    main()
