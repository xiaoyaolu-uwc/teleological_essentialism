#!/usr/bin/env python3
"""
run_cascade_folds.py
====================
Runs the trained per-fold gate + stage-2 pair over each fold's held-out works
and writes ONE per-row prediction file. Every downstream number in the
proportion evaluation is computed from that file offline -- no reruns.

Design note: stage 2 is run on EVERY held-out row, not only on gate survivors.
That costs a little extra inference and buys all three stage-attribution
variants from a single pass:

  end-to-end        gate decides, survivors take stage 2's label
  perfect gate      true non-junk rows take stage 2's label
  perfect stage 2   gate decides, survivors credited to their TRUE label

Usage:
    python3 eval/run_cascade_folds.py --gate-prefix gate_fold --s2-prefix s2_fold \
        --s2-stage nonjunk_3way --s2-prompt S2_structured --text-column text \
        --out eval/results/proportions/per_row_predictions.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS
from models.data_utils import load_train_pool
from models.train_bert import STAGE_LABELS, get_device
from models.lora_gate.model import load_trained_model
from models.lora_gate.train import PromptedTagDataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

BATCH_SIZE = 16
GATE_LABELS = STAGE_LABELS["junk_gate"]


def get_probs(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            out.append(F.softmax(logits, dim=-1).cpu())
    return torch.cat(out, dim=0)


def predict(model_name, ckpt_dir, labels, texts, prompt, max_length, device):
    model, tok = load_trained_model(model_name, ckpt_dir, len(labels), device)
    ds = PromptedTagDataset(texts, [0] * len(texts), tok, max_length, prompt)
    probs = get_probs(model, DataLoader(ds, batch_size=BATCH_SIZE), device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="eval/folds.json")
    ap.add_argument("--gate-prefix", default="gate_fold")
    ap.add_argument("--gate-suffixes", nargs="+", default=[""],
                     help="Checkpoint suffixes to ENSEMBLE over, e.g. '' '_s7' averages "
                          "gate_fold<N> and gate_fold<N>_s7. Averaging probabilities across "
                          "independently seeded runs cancels per-seed idiosyncrasy, which "
                          "re-thresholding a single seed cannot do (see "
                          "eval/calibrate_gate_threshold.py). One suffix = no ensemble.")
    ap.add_argument("--s2-prefix", default="s2_fold")
    ap.add_argument("--s2-stage", default="nonjunk_3way", choices=["nonjunk_3way", "full4way"])
    ap.add_argument("--gate-prompt", default="A_structured")
    ap.add_argument("--s2-prompt", default="S2_structured")
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--text-column", default="text", choices=["deploy_extract", "text"],
                     help="'text' is the deployable setting -- deploy_extract is itself an "
                          "LLM-produced field that a real scanning run would not have.")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--out", default="eval/results/proportions/per_row_predictions.csv")
    args = ap.parse_args()

    device = get_device()
    folds = json.load(open(args.folds))
    s2_labels = STAGE_LABELS[args.s2_stage]
    nonjunk_idx = GATE_LABELS.index("non_junk")

    rows_out = []
    for fold_name, works in sorted(folds.items()):
        _, held = load_train_pool(works)
        texts = [r[args.text_column] for r in held]
        gate_dirs = [PATHS["lora_checkpoints_dir"] / f"{args.gate_prefix}{fold_name[-1]}{sfx}"
                     for sfx in args.gate_suffixes]
        s2_dir = PATHS["lora_checkpoints_dir"] / f"{args.s2_prefix}{fold_name[-1]}"
        print(f"[{fold_name}] {len(held)} rows | gate={[d.name for d in gate_dirs]} "
              f"s2={s2_dir.name}", flush=True)

        gate_stack = [predict(args.model_name, d, GATE_LABELS, texts,
                              args.gate_prompt, args.max_length, device) for d in gate_dirs]
        gate_probs = torch.stack(gate_stack, dim=0).mean(dim=0)
        s2_probs = predict(args.model_name, s2_dir, s2_labels, texts,
                           args.s2_prompt, args.max_length, device)

        for i, r in enumerate(held):
            p_nonjunk = float(gate_probs[i, nonjunk_idx])
            rec = {
                "fold": fold_name,
                "work": r["work"],
                "author": r["author"],
                "year": r["year"],
                "true_tag": r["deploy_tag"],
                "gate_prob_nonjunk": round(p_nonjunk, 5),
                "gate_pred": "non_junk" if p_nonjunk >= args.gate_threshold else "junk",
                "s2_pred": s2_labels[int(s2_probs[i].argmax())],
            }
            for j, lab in enumerate(s2_labels):
                rec[f"s2_p_{lab}"] = round(float(s2_probs[i, j]), 5)
            rows_out.append(rec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows_out)
    meta = {k: v for k, v in vars(args).items()}
    json.dump(meta, open(out.with_suffix(".meta.json"), "w"), indent=2)
    print(f"wrote {len(rows_out)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
