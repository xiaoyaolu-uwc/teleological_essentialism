# BERT Fine-Tuning Evolution Log

Tracks fine-tuning MacBERTh (`emanjavacas/MacBERTh`, BERT-base pretrained
from scratch on English text 1450-1950) as a 4-way sequence classifier over
`data/sentences_train.csv` (the full labeled corpus minus the 40 rows
underlying the 49-row golden `evaluation_set.csv`), as a candidate to
eventually replace/complement the LLM deployment prompt (`d_v3`, 91.8% on
the golden set) for scanning the full corpus.

**Training script**: `models/train_bert.py`. Input text is the
`deploy_extract` column (the LLM-identified, OCR-corrected verbatim span
that drives each label), not the raw sentence-level `text` column — see the
ablation below for why. Loss is class-weighted cross-entropy
(inverse frequency) throughout, to counter junk's ~65% share of the corpus.
Diagnostic runs use `--holdout-work` to leave one text out entirely for
training, evaluating generalization on that held-out text plus the golden
set; checkpoints select the epoch with best validation macro-F1, not the
last epoch.

## Leave-one-text-out (LOO) baseline — 3 texts, 4 epochs

Three texts were chosen as holdout candidates because each contains all
four tags with reasonable counts and removing it doesn't badly skew the
remaining corpus's class proportions: Argyll's *The Reign of Law*, Owen's
*On the Nature of Limbs*, and Gray's *Darwiniana*.

| Holdout | Held-out acc / macro-F1 | Golden acc / macro-F1 |
|---|---|---|
| Reign of Law | 0.733 / 0.554 | 0.673 / 0.686 |
| Darwiniana | 0.837 / 0.487 | 0.653 / 0.653 |
| Nature of Limbs | 0.659 / 0.600 | 0.735 / 0.679 |

All three are well below the LLM deployment prompt's 91.8% golden-set
accuracy. Confusion matrices showed a consistent pattern: divine_teleology
and internal_essence rows getting pulled toward junk specifically on the
held-out (never-trained-on) text, even though in-distribution validation
accuracy looked fine (76-83%) — the model doesn't survive a genuine
cross-text distribution shift as well as it fits its own training
distribution.

## Ablation: `deploy_extract` vs. raw `text`

Hypothesis: since BERT only sees the short model-extracted quote, not the
full surrounding passage the LLM had when writing that extract, maybe the
extract throws away context BERT needs — training on the raw `text` chunk
might do better.

Rerun on Reign of Law holdout, raw `text` column, longer max sequence length
(256 vs. 128): **worse across nearly every metric**, not better. Held-out
macro-F1 dropped 0.554 → 0.466, and held-out internal_essence recall
collapsed from an already-weak 0.26 to 0.07. The noisier raw chunk dilutes
the (already scarce) structuralist signal rather than adding useful context.
`deploy_extract` confirmed as the better input field; ruled out revisiting
this lever.

## Epochs (4 → 8) and oversampling experiments

Two levers tried against the rare-class-collapsing-to-junk problem, each on
the two worst-affected holdouts (Reign of Law and Darwiniana had the weakest
internal_essence recall: 0.26 and 0.23):

| Run | Held-out macro-F1 | Held-out IE recall | Golden macro-F1 | Golden DT recall |
|---|---|---|---|---|
| Reign of Law, baseline (4ep) | 0.554 | 0.26 | 0.686 | 0.70 |
| Reign of Law, 8 epochs | 0.565 | 0.26 (unchanged) | 0.659 | 0.40 |
| Reign of Law, oversample | 0.548 | 0.19 (worse) | 0.590 | 0.30 |
| Darwiniana, baseline (4ep) | 0.487 | 0.23 | 0.653 | 0.50 |
| Darwiniana, 8 epochs | 0.495 | 0.45 (better) | 0.628 | 0.40 |
| Darwiniana, oversample | 0.518 | 0.41 (better) | 0.595 | 0.20 |

Neither lever is a general fix — more epochs helped internal_essence recall
on Darwiniana but did nothing for the identical problem on Reign of Law, and
oversampling showed the same split personality (better on Darwiniana, worse
on Reign of Law). Most tellingly, **both interventions consistently tanked
divine_teleology recall on the golden set** in every variant (dropping from
0.70/0.50 baseline to 0.20–0.40) — a "whack-a-mole" pattern where fixing one
rare class's recall costs another's, not a genuine improvement in the
model's capacity to separate the classes.

## Golden-set error inspection

`eval/inspect_golden_errors.py` loaded all 6 checkpoints above (baselines +
e8 + oversample variants) and checked, per golden-set row, whether every
checkpoint agreed, flipped, or was consistently wrong. This reframed the
problem:

- **internal_essence is mostly already solved**: 4 of its 6 golden rows are
  classified correctly by literally every one of the 6 checkpoints. The
  instability chased by epochs/oversampling was largely a red herring here.
- **divine_teleology is the real weak point**, but narrowly: 3 of its 10
  golden rows are wrong in *every* checkpoint tested, and those three share
  a clear pattern — none use explicit religious/providential vocabulary
  ("God," "Creator," "Providence," "Almighty"). They argue for design
  *implicitly* (an aesthetic/philosophical inference, or Agassiz's
  characteristic move of describing "categories of structure" and "the plan
  of creation" — language that reads as structuralist but is actually a
  divine-design argument). The 2 rows every checkpoint gets right every time
  use overt religious vocabulary instead.

This explains why neither epochs nor oversampling helped consistently: the
failure isn't "hasn't seen enough divine_teleology examples," it's that
recognizing an implicit design-argument needs more inferential room than a
short extract classified in one forward pass provides — echoing the LLM
deployment prompt's own hardest remaining failures (Agassiz meta-passages at
the junk/DT boundary, `eval/deployment_evolution.md`), which the LLM handles
via an explicit reasoning step BERT's architecture doesn't have.

## Next step taken

This diagnosis directly motivated the two-stage cascade (junk gate + 3-way
DT/NDT/IE classifier) to remove junk's 65%-of-corpus dominance from the
harder DT/NDT/IE separation — see `eval/bert_cascade_evolution.md`.
