#!/usr/bin/env python3
"""
backfill_gate_proportions.py
=============================
Adds holdout_proportion_metrics / golden_proportion_metrics (per-category
recall, recall evenness, true vs. output category mix -- see
models/data_utils.py's gate_proportion_metrics) to LoRA runs trained before
that computation existed in train.py. Re-evaluates the already-saved
adapter checkpoint -- no retraining -- using the same config (prompt
variant, max_length, text column) the run itself used, read back from its
own metrics.json, so results are identical to what a fresh run would have
produced.

Usage:
    python3 eval/backfill_gate_proportions.py                  # all runs under eval/results/lora/
    python3 eval/backfill_gate_proportions.py --run-name junk_gate_lora_lr1e4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS
from models.data_utils import load_train_pool, load_golden_eval, gate_proportion_metrics
from models.train_bert import STAGE_LABELS, get_device, evaluate
from models.lora_gate.model import load_trained_model
from models.lora_gate.train import PromptedTagDataset, HOLDOUT_WORK

from torch.utils.data import DataLoader

STAGE_LABEL_LIST = STAGE_LABELS["junk_gate"]
BATCH_SIZE = 16


def backfill_one(run_name, device):
    metrics_path = PATHS["lora_results_dir"] / run_name / "metrics.json"
    if not metrics_path.exists():
        print(f"[{run_name}] no metrics.json found, skipping")
        return
    report = json.load(open(metrics_path))
    if "holdout_proportion_metrics" in report:
        print(f"[{run_name}] already has proportion metrics, skipping")
        return

    ckpt_dir = PATHS["lora_checkpoints_dir"] / run_name
    if not ckpt_dir.exists():
        print(f"[{run_name}] no checkpoint found at {ckpt_dir}, skipping")
        return

    model_name = report["model"]
    prompt_variant = report.get("prompt_variant", "current")
    max_length = report["max_length"]
    text_column = report["text_column"]

    model, tokenizer = load_trained_model(model_name, ckpt_dir, len(STAGE_LABEL_LIST), device)

    _, held_out = load_train_pool(HOLDOUT_WORK)
    held_ds = PromptedTagDataset(
        [r[text_column] for r in held_out], [0] * len(held_out), tokenizer, max_length, prompt_variant)
    held_preds, _ = evaluate(model, DataLoader(held_ds, batch_size=BATCH_SIZE), device)
    held_true_tags = [r["deploy_tag"] for r in held_out]
    held_gate_preds = [STAGE_LABEL_LIST[p] for p in held_preds]
    report["holdout_proportion_metrics"] = gate_proportion_metrics(held_true_tags, held_gate_preds)

    golden = load_golden_eval(text_column)
    golden_ds = PromptedTagDataset(
        [r["text"] for r in golden], [0] * len(golden), tokenizer, max_length, prompt_variant)
    golden_preds, _ = evaluate(model, DataLoader(golden_ds, batch_size=BATCH_SIZE), device)
    golden_true_tags = [r["tag"] for r in golden]
    golden_gate_preds = [STAGE_LABEL_LIST[p] for p in golden_preds]
    report["golden_proportion_metrics"] = gate_proportion_metrics(golden_true_tags, golden_gate_preds)

    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)
    ev = report["holdout_proportion_metrics"]["recall_evenness"]
    print(f"[{run_name}] backfilled -- held-out recall_evenness={ev:.3f} -> {metrics_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=None,
                     help="Backfill just one run; omit to do every run under eval/results/lora/")
    args = ap.parse_args()
    device = get_device()

    if args.run_name:
        backfill_one(args.run_name, device)
    else:
        for d in sorted(PATHS["lora_results_dir"].glob("*")):
            if d.is_dir():
                backfill_one(d.name, device)


if __name__ == "__main__":
    main()
