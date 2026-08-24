#!/usr/bin/env python3
"""
evaluate.py
===========
Runs a model + prompt version against the hand-labelled evaluation set and
reports accuracy, per-class precision/recall/F1, and a confusion matrix.

Usage (from repo root):
    # Single-layer (default):
    python3 eval/evaluate.py --runs 5 --run-dir v7c --version v7c

    # Single-layer, different model:
    python3 eval/evaluate.py --runs 5 --run-dir v7c_full --version v7c --model gpt-5.4

    # Two-layer:
    python3 eval/evaluate.py --method two-layer --l1-version l1_v1 --l2-version l2_v1 \
        --runs 5 --run-dir two_layer_l1v1_l2v1

Output per run:
    eval/results/<run-dir>/eval_{MODEL}_{VERSION}.csv  (increments if exists)

Two-layer result CSVs include extra columns: l1_tag, l1_reasoning.
After all runs, writes eval/results/<run-dir>/analysis/error_analysis.csv.
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import PATHS
from archive.prompt_phase.openai import OpenAIModel
from labelling.prompts import PROMPT_VERSIONS


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

# ---------------------------------------------------------------------------
# Config defaults (overridden by CLI flags)
# ---------------------------------------------------------------------------
MODEL          = os.environ.get("SCAN_MODEL", "gpt-5.4-mini")
PROMPT_VERSION = "v7c"
BATCH_SIZE     = int(os.environ.get("SCAN_BATCH_SIZE", 10))
# ---------------------------------------------------------------------------


def load_eval_set() -> list[dict]:
    with open(PATHS["evaluation_csv"], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def results_path(model: str, version_label: str, out_dir: Path) -> Path:
    safe_model = model.replace("/", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"eval_{safe_model}_{version_label}.csv"
    if not base.exists():
        return base
    run = 2
    while True:
        candidate = out_dir / f"eval_{safe_model}_{version_label}_{run}.csv"
        if not candidate.exists():
            return candidate
        run += 1


def write_results(rows: list[dict], preds: list[dict], out_path: Path, two_layer: bool = False):
    fieldnames = ["text", "correct_tag", "your_rationale", "predicted_tag", "model_reasoning", "correct"]
    if two_layer:
        fieldnames += ["l1_tag", "l1_reasoning"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, pred in zip(rows, preds):
            record = {
                "text":            row["text"],
                "correct_tag":     row["correct_tag"],
                "your_rationale":  row.get("xy_rationale", ""),
                "predicted_tag":   pred["tag"],
                "model_reasoning": pred["reasoning"],
                "correct":         str(pred["tag"] == row["correct_tag"]),
            }
            if two_layer:
                record["l1_tag"]       = pred.get("l1_tag", "")
                record["l1_reasoning"] = pred.get("l1_reasoning", "")
            writer.writerow(record)


def print_summary(rows: list[dict], preds: list[dict], valid_tags: set):
    from sklearn.metrics import classification_report, confusion_matrix

    ground_truth_tags = {r["correct_tag"] for r in rows}
    all_labels = sorted(valid_tags | ground_truth_tags)

    y_true = [r["correct_tag"] for r in rows]
    y_pred = [p["tag"] for p in preds]

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    total = len(y_true)
    print(f"\nOverall accuracy: {correct}/{total} = {correct/total:.1%}\n")

    print(classification_report(y_true, y_pred, labels=all_labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    col_width = max(len(l) for l in all_labels) + 2
    header = " " * col_width + "".join(l.ljust(col_width) for l in all_labels)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(header)
    for label, row_vals in zip(all_labels, cm):
        print(label.ljust(col_width) + "".join(str(v).ljust(col_width) for v in row_vals))
    print()


def main():
    import argparse
    from archive.prompt_phase.analyze_errors import load_results, aggregate, print_report, write_analysis

    parser = argparse.ArgumentParser(description="Evaluate model against the hand-labelled eval set")
    parser.add_argument("--runs",       type=int,   default=5,        help="Number of evaluation runs (default: 5)")
    parser.add_argument("--run-dir",    type=str,   default=None,     help="Subdirectory of eval/results/ to write output into")
    parser.add_argument("--model",      type=str,   default=None,     help="Model name (overrides SCAN_MODEL env var)")
    parser.add_argument("--method",     type=str,   default="single", choices=["single", "two-layer"],
                        help="Pipeline method: single (default) or two-layer")
    # Single-layer flags
    parser.add_argument("--version",    type=str,   default=None,     help="[single] Prompt version key")
    # Two-layer flags
    parser.add_argument("--l1-version", type=str,   default="l1_v1",  help="[two-layer] L1 prompt version")
    parser.add_argument("--l2-version", type=str,   default="l2_v1",  help="[two-layer] L2 prompt version")
    args = parser.parse_args()

    global MODEL, PROMPT_VERSION
    if args.model:
        MODEL = args.model

    base_results = Path(__file__).resolve().parent / "results"
    out_dir = base_results / args.run_dir if args.run_dir else base_results

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    rows  = load_eval_set()
    texts = [r["text"] for r in rows]

    # --- Build model ---
    two_layer = args.method == "two-layer"
    if two_layer:
        from archive.prompt_phase.two_layer import TwoLayerModel
        from archive.prompt_phase.two_layer_prompts import L1_PROMPT_VERSIONS, L2_PROMPT_VERSIONS
        model = TwoLayerModel(
            model_name=MODEL,
            l1_version=args.l1_version,
            l2_version=args.l2_version,
            batch_size=BATCH_SIZE,
        )
        version_label = f"{args.l1_version}_{args.l2_version}"
        valid_tags = {"divine_teleology", "non_divine_teleology", "internal_essence", "junk"}
        print(f"Method: two-layer | L1: {args.l1_version} | L2: {args.l2_version} | Model: {MODEL}")
    else:
        version = args.version or PROMPT_VERSION
        PROMPT_VERSION = version
        model = OpenAIModel(model_name=MODEL, prompt_version=version, batch_size=BATCH_SIZE)
        version_label = version
        valid_tags = PROMPT_VERSIONS[version]["valid_tags"]
        print(f"Method: single | Version: {version} | Model: {MODEL}")

    # --- Run evaluations ---
    batch_files = []
    for run in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n--- Run {run}/{args.runs} ---")
        print(f"Evaluating {len(rows)} passages...")
        preds = model.classify(texts)
        out_path = results_path(MODEL, version_label, out_dir)
        write_results(rows, preds, out_path, two_layer=two_layer)
        print(f"Results written to: {out_path}")
        print_summary(rows, preds, valid_tags)
        batch_files.append(out_path)

    if args.runs > 1:
        print(f"\n{'='*60}")
        print(f"Batch complete ({args.runs} runs). Cross-run error analysis:")
        print(f"{'='*60}\n")
        records = aggregate(load_results(batch_files))
        print_report(records)
        write_analysis(records, out_dir=out_dir)


if __name__ == "__main__":
    main()
