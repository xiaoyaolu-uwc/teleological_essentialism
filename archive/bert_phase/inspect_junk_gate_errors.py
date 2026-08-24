#!/usr/bin/env python3
"""
inspect_junk_gate_errors.py
============================
Compares BERT's and the LoRA-Qwen3 gate's predictions row-by-row on the
golden evaluation_set.csv, for the junk_gate (binary junk/non_junk) stage.
Written to answer a specific question: LoRA's golden non_junk recall (0.60)
came out statistically identical to BERT's (0.60) despite LoRA clearly
beating BERT on held-out text (0.65 vs 0.47 recall) -- is that the *same*
set of golden rows failing for both models (a shared, genuinely hard-case
weakness), or different rows (two models failing for unrelated reasons,
which would average out to a similar aggregate number by coincidence)?

Requires both checkpoints to already exist locally:
  - models/checkpoints/junk_gate_loo_reign_of_law  (BERT, full model)
  - models/checkpoints/lora/junk_gate_lora_v1       (LoRA adapter)

Usage:
    python3 eval/inspect_junk_gate_errors.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import PATHS
from models.data import load_golden_eval
from archive.bert_phase.train_bert import STAGE_LABELS, stage_tag, get_device, evaluate, TagDataset
from models.lora.model import load_trained_model
from models.lora.train import PromptedTagDataset

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GATE_LABELS = STAGE_LABELS["junk_gate"]
GATE_LABEL2ID = {l: i for i, l in enumerate(GATE_LABELS)}
BERT_CKPT = "junk_gate_loo_reign_of_law"
LORA_CKPT = "junk_gate_lora_v1"
LORA_BASE_MODEL = "Qwen/Qwen3-0.6B"


def main():
    device = get_device()
    golden = load_golden_eval()
    golden_labels = [GATE_LABEL2ID[stage_tag(r["tag"], "junk_gate")] for r in golden]

    # --- BERT predictions ---
    bert_ckpt_dir = PATHS["bert_checkpoints_dir"] / BERT_CKPT
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_ckpt_dir)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_ckpt_dir)
    bert_model.to(device)
    bert_ds = TagDataset([r["text"] for r in golden], golden_labels, bert_tokenizer, max_length=128)
    bert_loader = DataLoader(bert_ds, batch_size=16)
    bert_preds, _ = evaluate(bert_model, bert_loader, device)
    print(f"loaded BERT checkpoint {BERT_CKPT}", flush=True)

    # --- LoRA predictions ---
    lora_ckpt_dir = PATHS["lora_checkpoints_dir"] / LORA_CKPT
    lora_model, lora_tokenizer = load_trained_model(
        LORA_BASE_MODEL, lora_ckpt_dir, num_labels=len(GATE_LABELS), device=device,
    )
    lora_ds = PromptedTagDataset([r["text"] for r in golden], golden_labels, lora_tokenizer, max_length=160)
    lora_loader = DataLoader(lora_ds, batch_size=16)
    lora_preds, _ = evaluate(lora_model, lora_loader, device)
    print(f"loaded LoRA checkpoint {LORA_CKPT}", flush=True)

    print("\n" + "=" * 100)
    print("Golden rows where true stage tag = non_junk (these are what non_junk_recall is measured over)\n")

    both_wrong, only_bert_wrong, only_lora_wrong, both_right = [], [], [], []

    for i, r in enumerate(golden):
        true_tag = stage_tag(r["tag"], "junk_gate")
        if true_tag != "non_junk":
            continue
        bert_tag = GATE_LABELS[bert_preds[i]]
        lora_tag = GATE_LABELS[lora_preds[i]]
        bert_ok = bert_tag == "non_junk"
        lora_ok = lora_tag == "non_junk"

        row_desc = (
            f"work={r['work']!r} true_category={r['tag']} bert={bert_tag} lora={lora_tag}\n"
            f"  extract: {r['text'][:220]}"
        )
        if not bert_ok and not lora_ok:
            both_wrong.append(row_desc)
        elif not bert_ok and lora_ok:
            only_bert_wrong.append(row_desc)
        elif bert_ok and not lora_ok:
            only_lora_wrong.append(row_desc)
        else:
            both_right.append(row_desc)

    print(f"BOTH WRONG (n={len(both_wrong)}) -- shared weakness, not fixed by switching architecture:\n")
    for d in both_wrong:
        print(d, "\n")

    print(f"\nONLY BERT WRONG, LoRA GOT IT (n={len(only_bert_wrong)}) -- LoRA's actual wins:\n")
    for d in only_bert_wrong:
        print(d, "\n")

    print(f"\nONLY LoRA WRONG, BERT GOT IT (n={len(only_lora_wrong)}) -- LoRA's new mistakes:\n")
    for d in only_lora_wrong:
        print(d, "\n")

    print(f"\nBOTH RIGHT (n={len(both_right)})")
    print(f"\nSummary: both_wrong={len(both_wrong)} only_bert_wrong={len(only_bert_wrong)} "
          f"only_lora_wrong={len(only_lora_wrong)} both_right={len(both_right)}")


if __name__ == "__main__":
    main()
