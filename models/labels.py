#!/usr/bin/env python3
"""
labels.py
=========
The task's label spaces. Deliberately dependency-free -- no torch, no
transformers.

That matters: the whole analysis path (models/data.py ->
evaluation/evaluate_proportions.py) imports these and nothing else heavy, so
every confidence claim in docs/PROPORTION_EVAL_RESULTS.md can be recomputed
from the committed per-row predictions on a laptop with no ML stack installed.
Keep it that way -- do not add a torch import to this file.

LABELS is the base 4-way task space. STAGE_LABELS maps a training stage onto
the label space it actually predicts in:

  full4way      all four classes, one model does everything
  junk_gate     binary {junk, non_junk} -- stage 1 of the cascade
  nonjunk_3way  {DT, NDT, IE} with junk removed entirely -- stage 2

See docs/history/bert_cascade_evolution.md for why the cascade exists: junk is
~65% of the corpus, and its dominance in a 4-way softmax dilutes the gradient
signal separating the three real categories.
"""

LABELS = ["divine_teleology", "non_divine_teleology", "internal_essence", "junk"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

NONJUNK_LABELS = ["divine_teleology", "non_divine_teleology", "internal_essence"]

STAGE_LABELS = {
    "full4way": LABELS,
    "junk_gate": ["junk", "non_junk"],
    "nonjunk_3way": NONJUNK_LABELS,
}


def stage_tag(tag, stage):
    """Maps a row's original 4-class deploy_tag/correct_tag onto the label
    space for the given stage."""
    if stage == "full4way":
        return tag
    if stage == "junk_gate":
        return "junk" if tag == "junk" else "non_junk"
    if stage == "nonjunk_3way":
        return tag  # caller is responsible for filtering out junk rows first
    raise ValueError(f"unknown stage {stage!r}")
