#!/usr/bin/env python3
"""
model.py
========
Builds the LoRA-wrapped junk-gate model: loads a causal LM (default:
Qwen3-0.6B) as a 2-class sequence classifier via
AutoModelForSequenceClassification -- the same API train_bert.py uses for
MacBERTh, just pointed at a decoder model -- then freezes it and attaches
trainable LoRA adapters to the attention projections via peft.

This is the one function in lora_gate/ that's genuinely new relative to
train_bert.py. Everything else (the training loop, data loading, metrics)
deliberately reuses or mirrors the BERT script, so this function is where
"how do I LoRA-fine-tune an LLM" actually lives.

Decoder models need two adjustments encoder models (BERT) don't:
  - A pad token. Causal LMs are usually trained without one (they just
    don't pad during pretraining), so we fall back to the EOS token, which
    is the standard recipe.
  - Right-padding specifically. HF's *ForSequenceClassification decoder
    implementations find the last real (non-pad) token per row by locating
    the first pad_token_id in input_ids -- that only gives the right answer
    if padding comes after the real content, i.e. right-padding.
"""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Attention projections only (not the MLP layers) -- the standard, cheapest
# LoRA target set, and the one most likely to matter for a task that's
# fundamentally "attend to the right words," not "learn new facts."
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_model_and_tokenizer(
    model_name,
    num_labels,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # The passage being classified sits at the very end of every prompt
    # template ("...Passage: {text}"). Default truncation cuts from the end,
    # which would silently drop the actual passage on any row that overflows
    # max_length -- truncating from the left instead sacrifices prompt
    # boilerplate first, never the content being classified.
    tokenizer.truncation_side = "left"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    return model, tokenizer


def build_full_finetune_model_and_tokenizer(model_name, num_labels):
    """Baseline comparison point only -- no LoRA, every parameter
    trainable. Answers "does the LoRA-adapter capacity bottleneck explain
    the recall/precision trade-off we've been seeing," not meant to be
    iterated on the way the LoRA config was."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def load_full_finetune_model(ckpt_dir, num_labels, device):
    """Mirrors load_trained_model for the full-fine-tune case: ckpt_dir
    already holds the complete model weights (save_pretrained on a plain
    AutoModelForSequenceClassification, not just an adapter), so there's no
    base-model-plus-adapter step."""
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir, num_labels=num_labels)
    model.to(device)
    return model, tokenizer


def load_trained_model(model_name, ckpt_dir, num_labels, device):
    """Reloads a saved adapter for inference/eval. Unlike a full fine-tune
    checkpoint, ckpt_dir only contains the small LoRA adapter weights --
    the base model still has to be re-fetched (from local HF cache after
    the first run, not re-downloaded) and the adapter layered on top."""
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base_model, ckpt_dir)
    model.to(device)
    return model, tokenizer
