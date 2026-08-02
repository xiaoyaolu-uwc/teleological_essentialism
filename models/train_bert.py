#!/usr/bin/env python3
"""
train_bert.py
=============
Fine-tunes MacBERTh (emanjavacas/MacBERTh) as a 4-way sequence classifier
(divine_teleology / non_divine_teleology / internal_essence / junk) on
sentences_train.csv, using the deploy_extract column as input text.

Two modes:
  - Diagnostic (--holdout-work "The Reign of Law"): trains on every other
    text, evaluates on the held-out text AND the golden evaluation_set.csv.
    Checkpoint is written but this is a throwaway run for measuring
    cross-text generalization.
  - Final (no --holdout-work): trains on the full corpus, evaluates only
    on the golden evaluation_set.csv.

Class-weighted cross-entropy (inverse frequency) is used throughout to
counter the junk-majority imbalance.

--stage controls the label space (see eval/bert_cascade_evolution.md for why):
  - full4way (default): the original 4-class problem, unchanged.
  - junk_gate: collapses labels to {junk, non_junk} — a binary gate meant to
    be run before nonjunk_3way.
  - nonjunk_3way: drops all junk rows (train/held-out/golden) and trains a
    3-way classifier over {divine_teleology, non_divine_teleology,
    internal_essence} only, with junk's 65%-of-corpus dominance removed from
    the training signal entirely. Use eval/evaluate_cascade.py to chain a
    junk_gate + nonjunk_3way pair into one end-to-end 4-way prediction.

Usage:
    python3 models/train_bert.py --run-name loo_reign_of_law \
        --holdout-work "The Reign of Law" --epochs 4
    python3 models/train_bert.py --run-name final --epochs 4
    python3 models/train_bert.py --run-name junk_gate_loo_reign_of_law \
        --holdout-work "The Reign of Law" --stage junk_gate --epochs 4
    python3 models/train_bert.py --run-name nonjunk3way_loo_reign_of_law \
        --holdout-work "The Reign of Law" --stage nonjunk_3way --epochs 4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS
from models.data_utils import (
    LABELS, LABEL2ID, load_train_pool, load_golden_eval,
    stratified_val_split, metrics_from_preds,
)

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "emanjavacas/MacBERTh"
MAX_LENGTH = 128

STAGE_LABELS = {
    "full4way": LABELS,
    "junk_gate": ["junk", "non_junk"],
    "nonjunk_3way": ["divine_teleology", "non_divine_teleology", "internal_essence"],
}


def stage_tag(tag, stage):
    """Maps a row's original 4-class deploy_tag/correct_tag onto the label
    space for the given --stage."""
    if stage == "full4way":
        return tag
    if stage == "junk_gate":
        return "junk" if tag == "junk" else "non_junk"
    if stage == "nonjunk_3way":
        return tag  # caller is responsible for filtering out junk rows first
    raise ValueError(f"unknown stage {stage!r}")


class TagDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        enc = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(labels, num_classes):
    counts = torch.zeros(num_classes)
    for l in labels:
        counts[l] += 1
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    return weights


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].tolist())
    return all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--holdout-work", default=None)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--text-column", default="deploy_extract", choices=["deploy_extract", "text"])
    ap.add_argument("--max-length", type=int, default=None,
                     help="Defaults to 128 for deploy_extract, 256 for raw text")
    ap.add_argument("--oversample", action="store_true",
                     help="Use a WeightedRandomSampler (inverse class frequency) to rebalance "
                          "training batches, instead of uniform shuffling")
    ap.add_argument("--no-class-weighted-loss", action="store_true",
                     help="Disable inverse-frequency loss weighting (useful when combined with "
                          "--oversample, to isolate which mechanism is doing the rebalancing)")
    ap.add_argument("--stage", default="full4way", choices=list(STAGE_LABELS.keys()),
                     help="full4way = original 4-class problem; junk_gate = binary junk/non_junk; "
                          "nonjunk_3way = 3-class DT/NDT/IE, junk rows dropped entirely")
    ap.add_argument("--model-name", default=MODEL_NAME,
                     help="Base model to fine-tune, e.g. bert-base-uncased instead of MacBERTh, "
                          "to isolate whether historical pretraining matters for a given stage")
    args = ap.parse_args()
    max_length = args.max_length or (256 if args.text_column == "text" else 128)
    stage_labels = STAGE_LABELS[args.stage]
    stage_label2id = {l: i for i, l in enumerate(stage_labels)}
    num_labels = len(stage_labels)

    device = get_device()
    print(f"[{args.run_name}] device={device} stage={args.stage} text_column={args.text_column} max_length={max_length}", flush=True)

    train_pool, held_out = load_train_pool(args.holdout_work)
    if args.stage == "nonjunk_3way":
        train_pool = [r for r in train_pool if r["deploy_tag"] != "junk"]
        held_out = [r for r in held_out if r["deploy_tag"] != "junk"]
    train_rows, val_rows = stratified_val_split(train_pool)
    print(f"[{args.run_name}] train={len(train_rows)} val={len(val_rows)} held_out={len(held_out)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    model.to(device)

    train_labels = [stage_label2id[stage_tag(r["deploy_tag"], args.stage)] for r in train_rows]
    class_weights = compute_class_weights(train_labels, num_labels).to(device)
    print(f"[{args.run_name}] class_weights={class_weights.tolist()}", flush=True)

    val_labels = [stage_label2id[stage_tag(r["deploy_tag"], args.stage)] for r in val_rows]
    train_ds = TagDataset([r[args.text_column] for r in train_rows], train_labels, tokenizer, max_length)
    val_ds = TagDataset([r[args.text_column] for r in val_rows], val_labels, tokenizer, max_length)

    if args.oversample:
        # Per-row weight = inverse frequency of that row's class, so rare
        # classes (divine_teleology, internal_essence) get drawn — with
        # repetition — into far more batches per epoch than their raw
        # corpus share would produce under uniform shuffling.
        cpu_class_weights = compute_class_weights(train_labels, num_labels)
        sample_weights = [cpu_class_weights[l].item() for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)

        realized_counts = torch.zeros(num_labels)
        for idx in sampler:
            realized_counts[train_labels[idx]] += 1
        print(f"[{args.run_name}] oversampled epoch composition: "
              f"{dict(zip(stage_labels, realized_counts.int().tolist()))}", flush=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    loss_weight = None if args.no_class_weighted_loss else class_weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = PATHS["bert_checkpoints_dir"] / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PATHS["bert_results_dir"] / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = -1
    epoch_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_preds, val_labels_out = evaluate(model, val_loader, device)
        val_metrics = metrics_from_preds(val_preds, val_labels_out, stage_labels)
        avg_loss = total_loss / len(train_loader)
        print(f"[{args.run_name}] epoch {epoch}: train_loss={avg_loss:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)
        epoch_log.append({"epoch": epoch, "train_loss": avg_loss, "val_metrics": val_metrics})

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    print(f"[{args.run_name}] best epoch={best_epoch} val_macro_f1={best_f1:.4f} -- reloading best checkpoint for final eval", flush=True)
    best_model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    best_model.to(device)

    report = {
        "run_name": args.run_name,
        "holdout_work": args.holdout_work,
        "model": args.model_name,
        "stage": args.stage,
        "stage_labels": stage_labels,
        "text_column": args.text_column,
        "max_length": max_length,
        "oversample": args.oversample,
        "class_weighted_loss": not args.no_class_weighted_loss,
        "epochs_run": args.epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "class_weights": class_weights.tolist(),
        "epoch_log": epoch_log,
    }

    if held_out:
        held_labels = [stage_label2id[stage_tag(r["deploy_tag"], args.stage)] for r in held_out]
        held_ds = TagDataset([r[args.text_column] for r in held_out], held_labels, tokenizer, max_length)
        held_loader = DataLoader(held_ds, batch_size=args.batch_size)
        held_preds, held_labels_out = evaluate(best_model, held_loader, device)
        held_metrics = metrics_from_preds(held_preds, held_labels_out, stage_labels)
        report["holdout_text_metrics"] = held_metrics
        print(f"[{args.run_name}] HELD-OUT TEXT ({args.holdout_work}): acc={held_metrics['accuracy']:.4f} macro_f1={held_metrics['macro_f1']:.4f}", flush=True)

    golden = load_golden_eval(args.text_column)
    if args.stage == "nonjunk_3way":
        golden = [r for r in golden if r["tag"] != "junk"]
    golden_labels = [stage_label2id[stage_tag(r["tag"], args.stage)] for r in golden]
    golden_ds = TagDataset([r["text"] for r in golden], golden_labels, tokenizer, max_length)
    golden_loader = DataLoader(golden_ds, batch_size=args.batch_size)
    golden_preds, golden_labels_out = evaluate(best_model, golden_loader, device)
    golden_metrics = metrics_from_preds(golden_preds, golden_labels_out, stage_labels)
    report["golden_eval_metrics"] = golden_metrics
    print(f"[{args.run_name}] GOLDEN EVAL SET: acc={golden_metrics['accuracy']:.4f} macro_f1={golden_metrics['macro_f1']:.4f} n={golden_metrics['n']}", flush=True)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "epoch_log"}, f, indent=2)

    print(f"[{args.run_name}] wrote {results_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
