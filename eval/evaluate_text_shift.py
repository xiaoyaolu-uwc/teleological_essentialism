#!/usr/bin/env python3
"""
evaluate_text_shift.py
=======================
Takes an already-trained checkpoint (trained on deploy_extract, the
LLM-refined quote) and evaluates it -- no retraining -- on the RAW `text`
column of its held-out text instead, to measure how much performance drops
under this train/eval distribution shift. Motivated by the concern that if
BERT is meant to eventually run standalone (without an LLM pre-extraction
step), it needs to work directly on raw passages, not on a field only the
LLM can produce.

Usage:
    python3 eval/evaluate_text_shift.py --checkpoint junk_gate_loo_reign_of_law \
        --holdout-work "The Reign of Law" --stage junk_gate
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS
from models.train_bert import (
    STAGE_LABELS, stage_tag, get_device, load_train_pool, TagDataset, evaluate, metrics_from_preds,
)

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MAX_LENGTH = 256  # raw text is longer than deploy_extract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Checkpoint dir name under models/checkpoints/")
    ap.add_argument("--holdout-work", required=True)
    ap.add_argument("--stage", required=True, choices=list(STAGE_LABELS.keys()))
    args = ap.parse_args()

    stage_labels = STAGE_LABELS[args.stage]
    stage_label2id = {l: i for i, l in enumerate(stage_labels)}

    device = get_device()
    ckpt_dir = PATHS["bert_checkpoints_dir"] / args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)

    _, held_out = load_train_pool(args.holdout_work)
    if args.stage == "nonjunk_3way":
        held_out = [r for r in held_out if r["deploy_tag"] != "junk"]

    for text_col in ["deploy_extract", "text"]:
        labels = [stage_label2id[stage_tag(r["deploy_tag"], args.stage)] for r in held_out]
        ds = TagDataset([r[text_col] for r in held_out], labels, tokenizer, MAX_LENGTH)
        loader = DataLoader(ds, batch_size=16)
        preds, labels_out = evaluate(model, loader, device)
        m = metrics_from_preds(preds, labels_out, stage_labels)
        recall = {k: round(v["recall"], 2) for k, v in m["per_class"].items()}
        print(f"[{args.checkpoint}] eval on {text_col:15s} acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} recall={recall}", flush=True)

        if text_col == "text":
            out_dir = PATHS["bert_results_dir"] / f"{args.checkpoint}_rawtext_shift"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({"checkpoint": args.checkpoint, "holdout_work": args.holdout_work,
                           "stage": args.stage, "eval_text_column": text_col, **m}, f, indent=2)


if __name__ == "__main__":
    main()
