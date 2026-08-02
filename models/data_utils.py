#!/usr/bin/env python3
"""
data_utils.py
=============
Data loading, splitting, and scoring utilities shared by every model
architecture trained on this task (BERT in train_bert.py, the LoRA-tuned
LLM junk gate in lora_gate/). Kept separate from any one architecture so
the same functions run the same way regardless of what's being trained --
that's what makes cross-architecture comparisons (e.g. LoRA vs. MacBERTh on
the junk gate) valid instead of apples-to-oranges.

LABELS / LABEL2ID here are the base 4-way task label space
(divine_teleology / non_divine_teleology / internal_essence / junk).
Stage-specific label spaces (e.g. the binary junk_gate collapse) stay in
train_bert.py for now, since only BERT's cascade code uses them today.
"""
import csv
import re

from config.config import PATHS

csv.field_size_limit(10**7)

LABELS = ["divine_teleology", "non_divine_teleology", "internal_essence", "junk"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


def norm(s):
    return re.sub(r"\s+", " ", s.strip())


def load_train_pool(holdout_work):
    with open(PATHS["sentences_train_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["deploy_tag"] in LABEL2ID and r["deploy_extract"].strip()]

    held_out = []
    if holdout_work:
        held_out = [r for r in rows if r["work"] == holdout_work]
        rows = [r for r in rows if r["work"] != holdout_work]
        if not held_out:
            raise ValueError(f"No rows found for holdout work {holdout_work!r}")
    return rows, held_out


def load_golden_eval(text_column="deploy_extract"):
    """Join evaluation_set.csv (correct_tag, ground truth) against
    sentences_labeled.csv to recover the deploy_extract (or raw text) for
    each golden row, via substring match on normalized text (see prior
    session matching)."""
    with open(PATHS["evaluation_csv"], newline="", encoding="utf-8") as f:
        ev = list(csv.DictReader(f))
    with open(PATHS["sentences_labeled_csv"], newline="", encoding="utf-8") as f:
        sl = list(csv.DictReader(f))

    by_work = {}
    for i, r in enumerate(sl):
        by_work.setdefault(r["work"], []).append(i)

    out = []
    for r in ev:
        key = norm(r["text"])
        snippet = key[:60]
        found = None
        for i in by_work.get(r["work"], []):
            t = norm(sl[i]["text"])
            if snippet in t or key in t:
                found = i
                break
        if found is None:
            for i, rr in enumerate(sl):
                t = norm(rr["text"])
                if snippet in t or key in t:
                    found = i
                    break
        if found is None:
            continue
        tag = r["correct_tag"].strip()
        if tag not in LABEL2ID:
            continue
        out.append({"text": sl[found][text_column], "tag": tag, "work": r["work"]})
    return out


def stratified_val_split(rows, val_frac=0.1, seed=13):
    import random
    rng = random.Random(seed)
    by_tag = {}
    for r in rows:
        by_tag.setdefault(r["deploy_tag"], []).append(r)
    train, val = [], []
    for _, items in by_tag.items():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def metrics_from_preds(preds, labels, label_names):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", labels=list(range(len(label_names))), zero_division=0)
    recall = recall_score(labels, preds, average=None, labels=list(range(len(label_names))), zero_division=0)
    precision = precision_score(labels, preds, average=None, labels=list(range(len(label_names))), zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names)))).tolist()
    per_class = {
        label_names[i]: {"recall": float(recall[i]), "precision": float(precision[i])}
        for i in range(len(label_names))
    }
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": label_names,
        "n": len(labels),
    }
