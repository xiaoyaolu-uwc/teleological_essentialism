#!/usr/bin/env python3
"""
inspect_golden_errors.py
=========================
Loads several trained BERT checkpoints and predicts on the golden
evaluation_set.csv (via the same deploy_extract join used in train_bert.py),
to see which specific rows are chronically misclassified across every
checkpoint (likely genuinely hard/ambiguous) vs. which ones flip between
right and wrong depending on training config (likely a capacity/instability
issue rather than inherent ambiguity).

Usage:
    python3 eval/inspect_golden_errors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS
from archive.bert_phase.train_bert import LABELS, LABEL2ID, load_golden_eval, TagDataset, evaluate, get_device

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RUNS = [
    "loo_reign_of_law",
    "loo_reign_of_law_e8",
    "loo_reign_of_law_oversample",
    "loo_darwiniana",
    "loo_darwiniana_e8",
    "loo_darwiniana_oversample",
]


def main():
    device = get_device()
    golden = load_golden_eval()
    golden_labels = [LABEL2ID[r["tag"]] for r in golden]

    tokenizer = AutoTokenizer.from_pretrained("emanjavacas/MacBERTh")
    all_preds = {}

    for run in RUNS:
        ckpt_dir = PATHS["bert_checkpoints_dir"] / run
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        model.to(device)
        ds = TagDataset([r["text"] for r in golden], golden_labels, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=16)
        preds, _ = evaluate(model, loader, device)
        all_preds[run] = preds
        print(f"loaded {run}", flush=True)

    print("\n" + "=" * 100)
    for target_tag in ["divine_teleology", "internal_essence"]:
        print(f"\n### golden rows with true tag = {target_tag} ###\n")
        for i, r in enumerate(golden):
            if r["tag"] != target_tag:
                continue
            preds_here = {run: LABELS[all_preds[run][i]] for run in RUNS}
            n_correct = sum(1 for v in preds_here.values() if v == target_tag)
            status = "ALWAYS WRONG" if n_correct == 0 else ("ALWAYS RIGHT" if n_correct == len(RUNS) else "FLIPS")
            print(f"[{status}] work={r['work']!r} correct={n_correct}/{len(RUNS)}")
            print(f"  extract: {r['text'][:220]}")
            print(f"  per-run predictions: {preds_here}")
            print()


if __name__ == "__main__":
    main()
