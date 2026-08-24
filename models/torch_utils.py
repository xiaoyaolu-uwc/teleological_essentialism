#!/usr/bin/env python3
"""
torch_utils.py
==============
Torch helpers shared by every model architecture in this project.

These used to live in train_bert.py, which meant the current LoRA pipeline
imported from the superseded BERT trainer just to get a device handle. Keeping
them here lets the BERT code be archived without breaking anything live, and
keeps cross-architecture comparisons honest by giving both paths literally the
same helpers.

Label spaces are NOT here -- they are in models/labels.py, which is kept free
of torch so the analysis path runs without an ML stack installed.
"""
import torch
from torch.utils.data import Dataset

from models.labels import LABELS, LABEL2ID, NONJUNK_LABELS, STAGE_LABELS, stage_tag  # noqa: F401


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(labels, num_classes):
    """Inverse-frequency weights, used to counter junk's majority share."""
    counts = torch.zeros(num_classes)
    for l in labels:
        counts[l] += 1
    total = counts.sum()
    return total / (num_classes * counts.clamp(min=1))


def evaluate(model, loader, device):
    """Argmax predictions over a loader. Returns (preds, labels) as int lists."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())
    return all_preds, all_labels


class TagDataset(Dataset):
    """Plain (unprompted) tokenized dataset -- what the BERT-family models use.
    The LoRA path uses PromptedTagDataset instead, which wraps each row in an
    instruction template first."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        enc = tokenizer(texts, truncation=True, padding="max_length",
                        max_length=max_length, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx]}
