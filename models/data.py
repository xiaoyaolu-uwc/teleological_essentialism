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

The label spaces themselves live in models/common.py and are re-exported here
for convenience, so callers that only need data loading import one module.
"""
import csv
import re

from config.config import PATHS

csv.field_size_limit(10**7)

from models.labels import LABELS, LABEL2ID, NONJUNK_LABELS  # noqa: F401  (re-exported)


def norm(s):
    return re.sub(r"\s+", " ", s.strip())


def load_train_pool(holdout_work):
    """holdout_work may be a single work title or a list/tuple of titles --
    the 6-fold design (evaluation/folds.json) holds out 2-3 works at once, so every
    work still gets an out-of-sample prediction while only 6 pipelines are
    trained instead of 16."""
    with open(PATHS["sentences_train_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["deploy_tag"] in LABEL2ID and r["deploy_extract"].strip()]

    held_out = []
    if holdout_work:
        works = [holdout_work] if isinstance(holdout_work, str) else list(holdout_work)
        missing = [w for w in works if not any(r["work"] == w for r in rows)]
        if missing:
            raise ValueError(f"No rows found for holdout work(s) {missing!r}")
        wset = set(works)
        held_out = [r for r in rows if r["work"] in wset]
        rows = [r for r in rows if r["work"] not in wset]
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


def _gate_proportion_summary(category_total, category_survived, predicted_junk_total, total_n):
    """Shared core for gate_proportion_metrics / gate_proportion_metrics_from_4way_confusion.
    See either caller's docstring for what each field means and why."""
    per_category_recall = {c: category_survived[c] / category_total[c] for c in NONJUNK_LABELS}
    evenness = max(per_category_recall.values()) - min(per_category_recall.values())

    true_nonjunk_total = sum(category_total[c] for c in NONJUNK_LABELS)
    true_nonjunk_mix = {c: category_total[c] / true_nonjunk_total for c in NONJUNK_LABELS}

    output_mix = {"junk": predicted_junk_total / total_n}
    output_mix.update({c: category_survived[c] / total_n for c in NONJUNK_LABELS})
    # True junk rows the gate incorrectly passed can't be credited to any
    # category under a perfect-3-way-classifier assumption (it has no junk
    # option) -- this residual is exactly the gate's junk-leakage rate.
    leaked_junk_uncredited_frac = 1.0 - sum(output_mix.values())

    # Same DT/NDT/IE mix, but normalized over survivors only (not the whole
    # text) -- directly comparable to true_nonjunk_mix on the same basis, so
    # the relative distortion (e.g. essentialism's share shrinking among
    # survivors) is visible without junk's share diluting the comparison.
    survived_total = sum(category_survived.values())
    survived_relative_mix = (
        {c: category_survived[c] / survived_total for c in NONJUNK_LABELS}
        if survived_total else {c: None for c in NONJUNK_LABELS}
    )

    return {
        "per_category_recall": per_category_recall,
        "recall_evenness": evenness,
        "true_nonjunk_mix": true_nonjunk_mix,
        "survived_relative_mix": survived_relative_mix,
        "output_mix_perfect_stage2": output_mix,
        "leaked_junk_uncredited_frac": leaked_junk_uncredited_frac,
        "n": total_n,
    }


def gate_proportion_metrics(true_tags, gate_preds):
    """Given per-row true 4-way tags and per-row binary gate predictions
    (junk/non_junk), computes the category-level view the project actually
    needs (accurate proportions across a text), not just pooled accuracy:
    per-category recall, how uneven those recalls are across DT/NDT/IE, the
    text's true non-junk mix, and the 4-way mix the gate would produce
    assuming a PERFECT downstream 3-way classifier -- a survivor is credited
    to its true category, not whatever stage 2 would guess, which isolates
    gate-induced distortion from stage-2 error entirely."""
    category_total = {c: 0 for c in NONJUNK_LABELS + ["junk"]}
    category_survived = {c: 0 for c in NONJUNK_LABELS}
    predicted_junk_total = 0
    for true_tag, pred in zip(true_tags, gate_preds):
        category_total[true_tag] += 1
        if pred == "junk":
            predicted_junk_total += 1
        elif true_tag in NONJUNK_LABELS:
            category_survived[true_tag] += 1
    return _gate_proportion_summary(category_total, category_survived, predicted_junk_total, len(true_tags))


def gate_proportion_metrics_from_4way_confusion(confusion_matrix, labels):
    """Same output as gate_proportion_metrics, but derived from an existing
    4-way (true x predicted) confusion matrix from an end-to-end cascade run
    -- no per-row data or rerun needed. Valid because 'predicted != junk'
    already means 'gate said non_junk'; what stage 2 specifically guessed
    doesn't matter here, only whether the row survived the gate at all,
    which the perfect-stage-2 assumption then credits to the row's true
    category regardless of stage 2's actual (possibly wrong) guess."""
    junk_idx = labels.index("junk")
    category_total, category_survived = {}, {}
    predicted_junk_total, total_n = 0, 0
    for i, true_label in enumerate(labels):
        row = confusion_matrix[i]
        row_total = sum(row)
        total_n += row_total
        category_total[true_label] = row_total
        predicted_junk_total += row[junk_idx]
        if true_label != "junk":
            category_survived[true_label] = row_total - row[junk_idx]
    return _gate_proportion_summary(category_total, category_survived, predicted_junk_total, total_n)


def category_mix_metrics(true_tags, pred_tags):
    """The stage-2 counterpart to gate_proportion_metrics: compares the
    DT/NDT/IE *mix* a classifier produces against the true mix for the same
    rows, which is what the blog post reports and what selection between
    stage-2 candidates should be decided on. Pooled accuracy can look healthy
    while the mix is skewed (errors that cancel row-for-row do not cancel in a
    proportion), so this is reported alongside, not instead of, accuracy.

    Junk is excluded from both sides so they share a basis. A full4way stage 2
    can predict junk; those rows are dropped from the predicted mix exactly as
    a real deployment would drop them -- which is the entire point of giving
    stage 2 a junk option, so it is not forced to assign leaked junk to a real
    category.

    signed_error is (predicted share - true share) per category: its mean
    across texts is the bias we disclose, its spread is the error bar.
    """
    true_counts = {c: 0 for c in NONJUNK_LABELS}
    pred_counts = {c: 0 for c in NONJUNK_LABELS}
    for t in true_tags:
        if t in true_counts:
            true_counts[t] += 1
    for p in pred_tags:
        if p in pred_counts:
            pred_counts[p] += 1
    true_n, pred_n = sum(true_counts.values()), sum(pred_counts.values())
    true_mix = {c: (true_counts[c] / true_n if true_n else None) for c in NONJUNK_LABELS}
    pred_mix = {c: (pred_counts[c] / pred_n if pred_n else None) for c in NONJUNK_LABELS}
    if not true_n or not pred_n:
        return {"true_mix": true_mix, "pred_mix": pred_mix, "signed_error": None,
                "max_abs_error": None, "tvd": None, "true_n": true_n, "pred_n": pred_n}

    signed = {c: pred_mix[c] - true_mix[c] for c in NONJUNK_LABELS}
    # Teleology (DT+NDT) vs essentialism (IE) -- the binary the blog post
    # most likely leads with, and a visibly tighter estimate than the 3-way.
    tel_true = true_mix["divine_teleology"] + true_mix["non_divine_teleology"]
    tel_pred = pred_mix["divine_teleology"] + pred_mix["non_divine_teleology"]
    return {
        "true_mix": true_mix,
        "pred_mix": pred_mix,
        "signed_error": signed,
        "max_abs_error": max(abs(v) for v in signed.values()),
        "tvd": 0.5 * sum(abs(v) for v in signed.values()),
        "teleology_true": tel_true,
        "teleology_pred": tel_pred,
        "teleology_signed_error": tel_pred - tel_true,
        "true_n": true_n,
        "pred_n": pred_n,
    }


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
