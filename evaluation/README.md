# Evaluation

Everything here serves one question: **when the pipeline reports that a text is
60% non-divine teleology, how much should you believe it?**

Only live scripts live in this directory. Superseded evaluation code from
earlier phases is in `archive/`, and the narrative logs are in `docs/history/`.

## Run order

| # | Script | Needs | Produces |
|---|---|---|---|
| 1 | `run_cascade_folds.py` | GPU + trained adapters | `results/proportions/per_row_predictions.csv` |
| 2 | `evaluate_proportions.py` | nothing (stdlib only) | `results/proportions/proportion_metrics.json` |
| 3 | `plot_proportions.py` | matplotlib | `results/proportions/proportion_figures.png` |

Steps 2 and 3 read only the CSV from step 1, which is committed — so every
published number can be reproduced without a GPU, without torch, in seconds.

### Optional / diagnostic

| Script | Purpose |
|---|---|
| `calibrate_gate_threshold.py` | Tests whether moving the gate's decision threshold helps. It does not (+0.15 pp). Kept because the negative result is load-bearing. |
| `compare_stage2_search.py` | Ranks stage-2 config-search runs under `results/lora/` by mix error rather than accuracy. |

## The design decision that shapes everything

**The unit of observation is the book, not the sentence.** Six folds
(`folds.json`) hold out 2–3 works each, so all 16 books get an out-of-sample
prediction from only 6 trained pipelines. Each book yields one signed error per
category; the spread of those 16 errors *is* the error bar. No resampling is
involved, because classifier bias does not shrink as rows are added and is the
dominant term — bootstrapping rows would measure the wrong thing.

`run_cascade_folds.py` deliberately runs **stage 2 on every held-out row**, not
just gate survivors. That costs a little inference and buys all three
stage-attribution variants (`end_to_end`, `perfect_gate`, `perfect_stage2`)
from a single pass, with no reruns.

## Reading `evaluate_proportions.py`

| Function | Computes |
|---|---|
| `per_work_errors` | the 16 × 3 matrix of signed errors — the atom under every claim |
| `bias_table` | bias, sd, parametric ±90% band, and the empirical p90 beside it |
| `within_work_ranking` | is the DT/NDT/IE order inside a book recovered |
| `across_work_ranking` | all 120 book-pairs × 3 categories, binned by true gap |
| `cluster_bootstrap_ci` | resamples **books**, because pairwise comparisons are not independent |
| `variant_labels` | the three stage-attribution relabelings |

Two honesty notes are wired into the output rather than left to the reader:
Wilson intervals on the pairwise rates assume independence they do not have
(the cluster bootstrap is printed alongside and is roughly twice as wide), and
the ±90% band is parametric while the empirical tail is fatter for NDT and IE.
