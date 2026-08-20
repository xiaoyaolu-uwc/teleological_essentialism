#!/usr/bin/env python3
"""
ensemble_gate.py
================
Two free, no-retraining checks on top of already-trained checkpoints:

1. Ensemble: average the non_junk softmax probability across multiple
   seeds of the same config, then threshold at 0.5. Averaging cancels
   per-seed idiosyncrasy (seed variance is now known to be large relative
   to most config deltas -- see eval/lora_prompt_evolution.md), so this
   is a plausible free way to raise the reliable floor without any new
   training.
2. Threshold sweep: using either a single model's or the ensemble's raw
   probabilities, sweep the non_junk decision threshold and report the
   full precision/recall/evenness curve. Precision is roughly as
   important as recall for this project (see eval/lora_junk_gate_evolution.md
   Status snapshot) -- a threshold is only worth adopting if it does not
   trade precision away, i.e. it must be Pareto-improving over 0.5, not a
   trade. This script reports the curve; it does not pick a winner.

Usage:
    python3 eval/ensemble_gate.py --run-names junk_gate_lora_r32a64 junk_gate_lora_r32a64_seed7 junk_gate_lora_r32a64_seed123
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS
from models.data_utils import load_train_pool, load_golden_eval, gate_proportion_metrics
from models.train_bert import STAGE_LABELS, get_device
from models.lora_gate.model import load_trained_model
from models.lora_gate.train import PromptedTagDataset, HOLDOUT_WORK

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

STAGE_LABEL_LIST = STAGE_LABELS["junk_gate"]
NON_JUNK_IDX = STAGE_LABEL_LIST.index("non_junk")
BATCH_SIZE = 16


def get_probs(model, loader, device):
    """Like train_bert.py's evaluate(), but returns raw softmax
    probabilities instead of collapsing to argmax -- needed here so
    multiple models' outputs can be averaged before thresholding."""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1).cpu()
            all_probs.append(probs)
    return torch.cat(all_probs, dim=0)


def preds_from_probs(probs, threshold):
    """probs: N x 2 tensor. Returns list of "junk"/"non_junk" strings."""
    non_junk_prob = probs[:, NON_JUNK_IDX]
    return ["non_junk" if p >= threshold else "junk" for p in non_junk_prob]


def junk_precision(true_tags, gate_preds):
    """Precision of the "junk" call: of everything predicted junk, what
    fraction was actually junk. High junk-precision means we're not
    over-aggressively discarding good non_junk rows -- it does NOT
    measure junk leaking into the kept set (that's non_junk_precision,
    below)."""
    predicted_junk = [i for i, p in enumerate(gate_preds) if p == "junk"]
    if not predicted_junk:
        return None
    correct = sum(1 for i in predicted_junk if true_tags[i] == "junk")
    return correct / len(predicted_junk)


def non_junk_precision(true_tags, gate_preds):
    """Precision of the "non_junk" call: of everything predicted
    non_junk (the set actually passed downstream to stage 2 / the final
    output), what fraction is truly non_junk. 1 minus this is exactly
    the junk-leakage-into-final-output rate marcus is most concerned
    about -- this is the metric his "precision" priority statement
    actually describes, distinct from junk_precision above."""
    predicted_nonjunk = [i for i, p in enumerate(gate_preds) if p == "non_junk"]
    if not predicted_nonjunk:
        return None
    correct = sum(1 for i in predicted_nonjunk if true_tags[i] != "junk")
    return correct / len(predicted_nonjunk)


def evaluate_probs(probs, true_tags, threshold):
    preds = preds_from_probs(probs, threshold)
    prop = gate_proportion_metrics(true_tags, preds)
    jprec = junk_precision(true_tags, preds)
    njprec = non_junk_precision(true_tags, preds)
    return prop, jprec, njprec


def load_data(text_column):
    _, held_out = load_train_pool(HOLDOUT_WORK)
    golden = load_golden_eval(text_column)
    held_texts = [r[text_column] for r in held_out]
    held_true_tags = [r["deploy_tag"] for r in held_out]
    golden_texts = [r["text"] for r in golden]
    golden_true_tags = [r["tag"] for r in golden]
    return held_texts, held_true_tags, golden_texts, golden_true_tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-names", nargs="+", required=True,
                     help="Checkpoint run-names to ensemble (average probabilities across all of them)")
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--prompt-variant", default="current")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--text-column", default="deploy_extract")
    ap.add_argument("--sweep", action="store_true",
                     help="Print the full threshold sweep (0.30-0.70) instead of just threshold=0.5")
    args = ap.parse_args()

    device = get_device()
    held_texts, held_true_tags, golden_texts, golden_true_tags = load_data(args.text_column)

    per_model_held_probs = []
    per_model_golden_probs = []
    for run_name in args.run_names:
        ckpt_dir = PATHS["lora_checkpoints_dir"] / run_name
        model, tokenizer = load_trained_model(args.model_name, ckpt_dir, len(STAGE_LABEL_LIST), device)
        held_ds = PromptedTagDataset(held_texts, [0] * len(held_texts), tokenizer, args.max_length, args.prompt_variant)
        golden_ds = PromptedTagDataset(golden_texts, [0] * len(golden_texts), tokenizer, args.max_length, args.prompt_variant)
        held_probs = get_probs(model, DataLoader(held_ds, batch_size=BATCH_SIZE), device)
        golden_probs = get_probs(model, DataLoader(golden_ds, batch_size=BATCH_SIZE), device)
        per_model_held_probs.append(held_probs)
        per_model_golden_probs.append(golden_probs)

        prop, jprec, njprec = evaluate_probs(held_probs, held_true_tags, 0.5)
        print(f"[{run_name}] held-out per-category recall: "
              f"{ {k: round(v, 3) for k, v in prop['per_category_recall'].items()} } "
              f"evenness={prop['recall_evenness']:.3f} junk_precision={jprec:.3f} "
              f"non_junk_precision={njprec:.3f}", flush=True)

    held_ensemble = torch.stack(per_model_held_probs, dim=0).mean(dim=0)
    golden_ensemble = torch.stack(per_model_golden_probs, dim=0).mean(dim=0)

    print(f"\n=== Ensemble of {len(args.run_names)} models (avg probability, threshold=0.5) ===")
    held_prop, held_jprec, held_njprec = evaluate_probs(held_ensemble, held_true_tags, 0.5)
    golden_prop, golden_jprec, golden_njprec = evaluate_probs(golden_ensemble, golden_true_tags, 0.5)
    print(f"HELD-OUT per-category recall: "
          f"{ {k: round(v, 3) for k, v in held_prop['per_category_recall'].items()} } "
          f"evenness={held_prop['recall_evenness']:.3f} junk_precision={held_jprec:.3f} "
          f"non_junk_precision={held_njprec:.3f}")
    print(f"GOLDEN per-category recall: "
          f"{ {k: round(v, 3) for k, v in golden_prop['per_category_recall'].items()} } "
          f"evenness={golden_prop['recall_evenness']:.3f} junk_precision={golden_jprec:.3f} "
          f"non_junk_precision={golden_njprec:.3f}")

    if args.sweep:
        print(f"\n=== Threshold sweep (held-out, ensemble) ===")
        print(f"{'threshold':>9s} | {'DT':>5s} {'NDT':>5s} {'IE':>5s} | {'even':>5s} | {'jprec':>5s} | {'non_junk_prec':>13s}")
        for t100 in range(30, 81, 5):
            t = t100 / 100.0
            prop, jprec, njprec = evaluate_probs(held_ensemble, held_true_tags, t)
            rec = prop["per_category_recall"]
            print(f"{t:9.2f} | {rec['divine_teleology']:.2f}  {rec['non_divine_teleology']:.2f}  "
                  f"{rec['internal_essence']:.2f} | {prop['recall_evenness']:.3f} | {jprec:.3f} | {njprec:>13.3f}")


if __name__ == "__main__":
    main()
