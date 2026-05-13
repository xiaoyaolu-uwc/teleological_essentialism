#!/usr/bin/env python3
"""
evaluate.py
===========
Runs a model + prompt version against the hand-labelled evaluation set and
reports accuracy, per-class precision/recall/F1, and a confusion matrix.

Usage (from repo root):
    python3 eval/evaluate.py

Output:
    eval/results/eval_{MODEL}_{PROMPT_VERSION}.csv
    CLI summary printed to stdout
"""

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.config import PATHS
from models.openai import OpenAIModel
from models.prompts import PROMPT_VERSIONS


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
# Config — PROMPT_VERSION and BATCH_SIZE can be edited here;
# MODEL defaults to SCAN_MODEL from .env
# ---------------------------------------------------------------------------
MODEL          = os.environ.get("SCAN_MODEL", "gpt-4o-mini")
PROMPT_VERSION = "v1"
BATCH_SIZE     = int(os.environ.get("SCAN_BATCH_SIZE", 10))
# ---------------------------------------------------------------------------


def load_eval_set() -> list[dict]:
    with open(PATHS["evaluation_csv"], newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def results_path(model: str, prompt_version: str) -> Path:
    safe_model = model.replace("/", "-")
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    return out_dir / f"eval_{safe_model}_{prompt_version}.csv"


def write_results(rows: list[dict], preds: list[dict], out_path: Path):
    fieldnames = ["text", "correct_tag", "your_rationale", "predicted_tag", "model_reasoning", "correct"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row, pred in zip(rows, preds):
            writer.writerow({
                "text":            row["text"],
                "correct_tag":     row["correct_tag"],
                "your_rationale":  row.get("rationale", ""),
                "predicted_tag":   pred["tag"],
                "model_reasoning": pred["reasoning"],
                "correct":         str(pred["tag"] == row["correct_tag"]),
            })


def print_summary(rows: list[dict], preds: list[dict]):
    from sklearn.metrics import classification_report, confusion_matrix

    valid_tags = PROMPT_VERSIONS[PROMPT_VERSION]["valid_tags"]
    all_labels = sorted(valid_tags | {"mixed"})

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
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    rows = load_eval_set()
    texts = [r["text"] for r in rows]

    print(f"Evaluating {MODEL!r} with prompt {PROMPT_VERSION!r} on {len(rows)} passages...")
    model = OpenAIModel(model_name=MODEL, prompt_version=PROMPT_VERSION, batch_size=BATCH_SIZE)
    preds = model.classify(texts)

    out_path = results_path(MODEL, PROMPT_VERSION)
    write_results(rows, preds, out_path)
    print(f"Results written to: {out_path}")

    print_summary(rows, preds)


if __name__ == "__main__":
    main()
