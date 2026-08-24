#!/usr/bin/env python3
"""
evaluate_cascade.py
====================
Chains a junk_gate checkpoint (binary junk/non_junk) and a nonjunk_3way
checkpoint (divine_teleology/non_divine_teleology/internal_essence) into one
end-to-end 4-way prediction, and scores it against the same held-out text and
golden evaluation_set.csv used for the single 4-way model diagnostics, so the
numbers are directly comparable (see eval/bert_cascade_evolution.md).

For each row:
  1. junk_gate predicts junk / non_junk.
  2. If non_junk, nonjunk_3way predicts the specific tag. If junk, the final
     label is junk directly (stage 2 never runs on that row).

Also reports stage-1-only accuracy and stage-2-only accuracy (restricted to
truly non-junk rows) separately, to distinguish stage-1 misrouting from
stage-2 confusion.

Usage:
    python3 eval/evaluate_cascade.py --holdout-suffix loo_reign_of_law \
        --holdout-work "The Reign of Law"
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS
from archive.bert_phase.train_bert import (
    LABELS, LABEL2ID, STAGE_LABELS, stage_tag, get_device,
    load_train_pool, load_golden_eval, TagDataset, evaluate, metrics_from_preds,
)

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BATCH_SIZE = 16
MAX_LENGTH = 128


def predict_stage(ckpt_dir, texts, tokenizer, device):
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model.to(device)
    dummy_labels = [0] * len(texts)
    ds = TagDataset(texts, dummy_labels, tokenizer, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    preds, _ = evaluate(model, loader, device)
    return preds


def run_cascade(texts, junk_gate_dir, nonjunk3way_dir, tokenizer, device):
    """texts: list of raw strings to classify. Returns (final_preds, stage1_preds)
    as lists of ints in the 4-class LABEL2ID space."""
    gate_labels = STAGE_LABELS["junk_gate"]
    stage1_preds = predict_stage(junk_gate_dir, texts, tokenizer, device)
    stage1_tags = [gate_labels[p] for p in stage1_preds]

    nonjunk_idx = [i for i, t in enumerate(stage1_tags) if t == "non_junk"]
    final_tags = list(stage1_tags)  # "junk" stays as-is; non_junk gets overwritten below

    if nonjunk_idx:
        nonjunk_labels = STAGE_LABELS["nonjunk_3way"]
        nonjunk_texts = [texts[i] for i in nonjunk_idx]
        stage2_preds = predict_stage(nonjunk3way_dir, nonjunk_texts, tokenizer, device)
        for i, p in zip(nonjunk_idx, stage2_preds):
            final_tags[i] = nonjunk_labels[p]

    final_preds = [LABEL2ID[t] for t in final_tags]
    return final_preds, stage1_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-suffix", required=True,
                     help="e.g. loo_reign_of_law -- expects checkpoints "
                          "junk_gate_<suffix> and nonjunk3way_<suffix>")
    ap.add_argument("--holdout-work", required=True,
                     help="Work name to pull the true held-out rows for, e.g. 'The Reign of Law'")
    ap.add_argument("--text-column", default="deploy_extract", choices=["deploy_extract", "text"])
    args = ap.parse_args()

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("emanjavacas/MacBERTh")
    junk_gate_dir = PATHS["bert_checkpoints_dir"] / f"junk_gate_{args.holdout_suffix}"
    nonjunk3way_dir = PATHS["bert_checkpoints_dir"] / f"nonjunk3way_{args.holdout_suffix}"

    report = {"holdout_suffix": args.holdout_suffix, "holdout_work": args.holdout_work}

    # --- held-out text ---
    _, held_out = load_train_pool(args.holdout_work)
    held_true = [LABEL2ID[r["deploy_tag"]] for r in held_out]
    held_texts = [r[args.text_column] for r in held_out]
    held_final_preds, held_stage1_preds = run_cascade(
        held_texts, junk_gate_dir, nonjunk3way_dir, tokenizer, device)
    held_metrics = metrics_from_preds(held_final_preds, held_true, LABELS)
    report["holdout_text_metrics"] = held_metrics
    print(f"[cascade {args.holdout_suffix}] HELD-OUT TEXT end-to-end: "
          f"acc={held_metrics['accuracy']:.4f} macro_f1={held_metrics['macro_f1']:.4f}", flush=True)

    # stage-1-only accuracy on held-out text
    held_gate_true = [0 if LABELS[t] == "junk" else 1 for t in held_true]
    held_gate_metrics = metrics_from_preds(held_stage1_preds, held_gate_true, STAGE_LABELS["junk_gate"])
    report["holdout_stage1_metrics"] = held_gate_metrics
    print(f"[cascade {args.holdout_suffix}] HELD-OUT stage-1 (junk gate) acc={held_gate_metrics['accuracy']:.4f}", flush=True)

    # stage-2-only accuracy, restricted to truly non-junk held-out rows
    truly_nonjunk = [(r, t) for r, t in zip(held_out, held_true) if LABELS[t] != "junk"]
    if truly_nonjunk:
        rows2, true2 = zip(*truly_nonjunk)
        texts2 = [r[args.text_column] for r in rows2]
        stage2_preds = predict_stage(nonjunk3way_dir, texts2, tokenizer, device)
        stage2_true = [STAGE_LABELS["nonjunk_3way"].index(LABELS[t]) for t in true2]
        stage2_metrics = metrics_from_preds(stage2_preds, stage2_true, STAGE_LABELS["nonjunk_3way"])
        report["holdout_stage2_metrics"] = stage2_metrics
        print(f"[cascade {args.holdout_suffix}] HELD-OUT stage-2 (3-way, true non-junk only) "
              f"acc={stage2_metrics['accuracy']:.4f}", flush=True)

    # --- golden eval set ---
    golden = load_golden_eval(args.text_column)
    golden_true = [LABEL2ID[r["tag"]] for r in golden]
    golden_texts = [r["text"] for r in golden]
    golden_final_preds, golden_stage1_preds = run_cascade(
        golden_texts, junk_gate_dir, nonjunk3way_dir, tokenizer, device)
    golden_metrics = metrics_from_preds(golden_final_preds, golden_true, LABELS)
    report["golden_eval_metrics"] = golden_metrics
    print(f"[cascade {args.holdout_suffix}] GOLDEN end-to-end: "
          f"acc={golden_metrics['accuracy']:.4f} macro_f1={golden_metrics['macro_f1']:.4f} n={golden_metrics['n']}", flush=True)

    golden_gate_true = [0 if LABELS[t] == "junk" else 1 for t in golden_true]
    golden_gate_metrics = metrics_from_preds(golden_stage1_preds, golden_gate_true, STAGE_LABELS["junk_gate"])
    report["golden_stage1_metrics"] = golden_gate_metrics
    print(f"[cascade {args.holdout_suffix}] GOLDEN stage-1 (junk gate) acc={golden_gate_metrics['accuracy']:.4f}", flush=True)

    truly_nonjunk_g = [(r, t) for r, t in zip(golden, golden_true) if LABELS[t] != "junk"]
    if truly_nonjunk_g:
        rows2, true2 = zip(*truly_nonjunk_g)
        texts2 = [r["text"] for r in rows2]
        stage2_preds = predict_stage(nonjunk3way_dir, texts2, tokenizer, device)
        stage2_true = [STAGE_LABELS["nonjunk_3way"].index(LABELS[t]) for t in true2]
        stage2_metrics = metrics_from_preds(stage2_preds, stage2_true, STAGE_LABELS["nonjunk_3way"])
        report["golden_stage2_metrics"] = stage2_metrics
        print(f"[cascade {args.holdout_suffix}] GOLDEN stage-2 (3-way, true non-junk only) "
              f"acc={stage2_metrics['accuracy']:.4f}", flush=True)

    results_dir = PATHS["bert_cascade_results_dir"] / args.holdout_suffix
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[cascade {args.holdout_suffix}] wrote {results_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
