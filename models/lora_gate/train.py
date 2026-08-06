#!/usr/bin/env python3
"""
train.py
========
LoRA-fine-tunes a small causal LM (default: Qwen3-0.6B) as the junk/non-junk
gate -- stage 1 of the cascade. Tests whether a decoder-style LLM draws the
junk/non-junk line better than BERT did, since BERT's gate is the project's
confirmed bottleneck (see LORA_JUNK_GATE_PLAN.md and
eval/bert_cascade_evolution.md).

Deliberately mirrors train_bert.py's manual training loop and reuses its
evaluation code (imported, not reimplemented) rather than using peft's
Trainer wrapper -- the point of this session is to see the training loop,
not call a black box. The only genuinely new piece is model.py's LoRA
wrapping; everything else about how a run is structured, scored, and
checkpointed is intentionally identical to the BERT script, which is what
makes the BERT-vs-LoRA numbers in the plan's target table comparable.

Single holdout only -- Reign of Law, per LORA_JUNK_GATE_PLAN.md section 9
(chosen because it's the one holdout BERT already has a measured baseline
for). This script doesn't take --holdout-work the way train_bert.py does;
that decision is already made and baked into the eval targets.

Usage:
    python3 models/lora_gate/train.py --run-name junk_gate_lora_v1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import PATHS
from models.data_utils import load_train_pool, load_golden_eval, stratified_val_split, metrics_from_preds
from models.train_bert import STAGE_LABELS, stage_tag, get_device, evaluate, compute_class_weights
from models.lora_gate.model import build_model_and_tokenizer, load_trained_model, DEFAULT_TARGET_MODULES

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

TARGET_MODULE_PRESETS = {
    "attn": DEFAULT_TARGET_MODULES,
    "attn_mlp": DEFAULT_TARGET_MODULES + ["gate_proj", "up_proj", "down_proj"],
}


def seed_everything(seed):
    """Fixes the two randomness sources train_bert.py never controlled for
    either (LoRA/classifier-head init, training shuffle order) -- both draw
    from the same global torch RNG, so one call here controls both. Needed
    for the hyperparameter sweep in LORA_JUNK_GATE_PLAN.md to mean anything:
    without it, a difference between two configs could just be luck, the
    same problem that made the BERT re-run comparison unreliable."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

HOLDOUT_WORK = "The Reign of Law"
STAGE = "junk_gate"
STAGE_LABEL_LIST = STAGE_LABELS[STAGE]
STAGE_LABEL2ID = {l: i for i, l in enumerate(STAGE_LABEL_LIST)}

# The task-framing prompt: unlike BERT, which just saw raw text, an
# instruction-tuned model like Qwen3 is expected to respond better to the
# task being stated explicitly. See LORA_JUNK_GATE_PLAN.md section 3 for why
# this doesn't turn the classification-head approach into the generative one
# -- the model still just picks argmax over 2 logits, it's only the *input*
# that's prompted.
#
# "none" is a bare-text control (like BERT ever saw) -- answers whether
# prompting helps at all before trusting any of the others. "rich" spells
# out the three positive subtypes explicitly rather than leaving "purpose,
# function, or structure" abstract. "fewshot" adds one concrete example of
# each class -- both examples below are synthetic, written for this prompt,
# not pulled from the corpus, so there's no risk of leaking held-out/golden
# text into training via the prompt itself.
PROMPT_VARIANTS = {
    "none": None,
    "current": (
        "Classify this passage from a historical natural-history text.\n"
        "non_junk: the passage explains or asserts something about an animal's "
        "purpose, function, or structure.\n"
        "junk: the passage does not.\n\n"
        "Passage: {text}"
    ),
    "rich": (
        "Classify this passage from a historical natural-history text as junk or non_junk.\n"
        "non_junk passages explain or assert something about an animal's purpose, function, "
        "or structure -- for example: a divine/teleological purpose (\"the wing is formed for "
        "flight, by design\"), a naturalized/evolutionary function (\"this trait persists "
        "because it aids survival\"), or an internal structural/anatomical account (\"the "
        "limb's structure follows a common archetype\").\n"
        "junk passages contain no such explanatory content (e.g. narrative, citation, or "
        "unrelated description).\n\n"
        "Passage: {text}"
    ),
    "fewshot": (
        "Classify passages from historical natural-history texts as junk or non_junk.\n\n"
        "Example (non_junk): \"The eye is admirably contrived for the purpose of vision, its "
        "lens and humours cooperating to focus light upon the retina.\"\n"
        "Example (junk): \"The author then proceeded to describe the voyage from Lisbon, "
        "dwelling at length upon the discomforts of the passage.\"\n\n"
        "Now classify this passage:\n"
        "non_junk: explains or asserts something about an animal's purpose, function, or structure.\n"
        "junk: does not.\n\n"
        "Passage: {text}"
    ),
}


class PromptedTagDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, prompt_variant="current"):
        template = PROMPT_VARIANTS[prompt_variant]
        prompted = texts if template is None else [template.format(text=t) for t in texts]
        enc = tokenizer(
            prompted, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4,
                     help="LoRA typically wants a higher LR than full fine-tuning "
                          "(train_bert.py uses 2e-5) since far fewer params are updated")
    ap.add_argument("--max-length", type=int, default=160,
                     help="Higher than train_bert.py's 128 to leave room for the prompt template")
    ap.add_argument("--text-column", default="deploy_extract", choices=["deploy_extract", "text"])
    ap.add_argument("--prompt-variant", default="current", choices=list(PROMPT_VARIANTS.keys()),
                     help="none = bare text (BERT-style control); current = existing abstract "
                          "framing; rich = names the three positive subtypes explicitly; "
                          "fewshot = adds one synthetic junk/non_junk example")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="attn", choices=list(TARGET_MODULE_PRESETS.keys()),
                     help="attn = q/k/v/o_proj only (default, cheapest); "
                          "attn_mlp = attn plus gate/up/down_proj (~2x trainable params)")
    ap.add_argument("--oversample", action="store_true",
                     help="Use a WeightedRandomSampler (inverse class frequency) instead of "
                          "uniform shuffling, mirroring train_bert.py's --oversample")
    ap.add_argument("--seed", type=int, default=42,
                     help="Fixes LoRA/classifier-head init and training shuffle order, so sweep "
                          "runs differ only in the hyperparameter being varied")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"[{args.run_name}] device={device} model={args.model_name} "
          f"holdout={HOLDOUT_WORK!r} stage={STAGE} seed={args.seed}", flush=True)

    train_pool, held_out = load_train_pool(HOLDOUT_WORK)
    train_rows, val_rows = stratified_val_split(train_pool)
    print(f"[{args.run_name}] train={len(train_rows)} val={len(val_rows)} held_out={len(held_out)}", flush=True)

    model, tokenizer = build_model_and_tokenizer(
        args.model_name, num_labels=len(STAGE_LABEL_LIST),
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULE_PRESETS[args.target_modules],
    )
    model.to(device)
    model.print_trainable_parameters()

    train_labels = [STAGE_LABEL2ID[stage_tag(r["deploy_tag"], STAGE)] for r in train_rows]
    class_weights = compute_class_weights(train_labels, len(STAGE_LABEL_LIST)).to(device)
    print(f"[{args.run_name}] class_weights={class_weights.tolist()}", flush=True)

    val_labels = [STAGE_LABEL2ID[stage_tag(r["deploy_tag"], STAGE)] for r in val_rows]
    train_ds = PromptedTagDataset([r[args.text_column] for r in train_rows], train_labels, tokenizer, args.max_length, args.prompt_variant)
    val_ds = PromptedTagDataset([r[args.text_column] for r in val_rows], val_labels, tokenizer, args.max_length, args.prompt_variant)

    if args.oversample:
        # Same rationale as train_bert.py's --oversample: draw rare classes
        # (non_junk, here) into more batches per epoch than their raw share
        # would produce under uniform shuffling.
        cpu_class_weights = compute_class_weights(train_labels, len(STAGE_LABEL_LIST))
        sample_weights = [cpu_class_weights[l].item() for l in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
    )

    ckpt_dir = PATHS["lora_checkpoints_dir"] / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PATHS["lora_results_dir"] / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = -1
    epoch_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
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
        val_metrics = metrics_from_preds(val_preds, val_labels_out, STAGE_LABEL_LIST)
        avg_loss = total_loss / len(train_loader)
        print(f"[{args.run_name}] epoch {epoch}: train_loss={avg_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}", flush=True)
        epoch_log.append({"epoch": epoch, "train_loss": avg_loss, "val_metrics": val_metrics})

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    print(f"[{args.run_name}] best epoch={best_epoch} val_macro_f1={best_f1:.4f} "
          f"-- reloading best adapter for final eval", flush=True)
    best_model, _ = load_trained_model(args.model_name, ckpt_dir, len(STAGE_LABEL_LIST), device)

    report = {
        "run_name": args.run_name,
        "holdout_work": HOLDOUT_WORK,
        "model": args.model_name,
        "stage": STAGE,
        "stage_labels": STAGE_LABEL_LIST,
        "text_column": args.text_column,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "prompt_variant": args.prompt_variant,
        "oversample": args.oversample,
        "seed": args.seed,
        "lr": args.lr,
        "epochs_run": args.epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_n": len(train_rows),
        "val_n": len(val_rows),
        "class_weights": class_weights.tolist(),
        "epoch_log": epoch_log,
    }

    held_labels = [STAGE_LABEL2ID[stage_tag(r["deploy_tag"], STAGE)] for r in held_out]
    held_ds = PromptedTagDataset([r[args.text_column] for r in held_out], held_labels, tokenizer, args.max_length, args.prompt_variant)
    held_loader = DataLoader(held_ds, batch_size=args.batch_size)
    held_preds, held_labels_out = evaluate(best_model, held_loader, device)
    held_metrics = metrics_from_preds(held_preds, held_labels_out, STAGE_LABEL_LIST)
    report["holdout_text_metrics"] = held_metrics
    print(f"[{args.run_name}] HELD-OUT TEXT ({HOLDOUT_WORK}): "
          f"acc={held_metrics['accuracy']:.4f} macro_f1={held_metrics['macro_f1']:.4f} "
          f"non_junk_recall={held_metrics['per_class']['non_junk']['recall']:.4f}", flush=True)

    golden = load_golden_eval(args.text_column)
    golden_labels = [STAGE_LABEL2ID[stage_tag(r["tag"], STAGE)] for r in golden]
    golden_ds = PromptedTagDataset([r["text"] for r in golden], golden_labels, tokenizer, args.max_length, args.prompt_variant)
    golden_loader = DataLoader(golden_ds, batch_size=args.batch_size)
    golden_preds, golden_labels_out = evaluate(best_model, golden_loader, device)
    golden_metrics = metrics_from_preds(golden_preds, golden_labels_out, STAGE_LABEL_LIST)
    report["golden_eval_metrics"] = golden_metrics
    print(f"[{args.run_name}] GOLDEN EVAL SET: acc={golden_metrics['accuracy']:.4f} "
          f"macro_f1={golden_metrics['macro_f1']:.4f} "
          f"non_junk_recall={golden_metrics['per_class']['non_junk']['recall']:.4f} "
          f"n={golden_metrics['n']}", flush=True)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(ckpt_dir / "metadata.json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "epoch_log"}, f, indent=2)

    print(f"[{args.run_name}] wrote {results_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
