# Proportion Evaluation — Plan of Record & Resume State

**Owner:** xiaoyao (marcus). **Started:** 2026-08-23.
This file is the durable state for the overnight proportion-evaluation build.
If a session dies, read this file first — it holds every locked decision and
the current progress checklist. Update the checklist as steps complete.

---

## Goal

Measure how accurately the deployed cascade reproduces the DT/NDT/IE
*proportions* of an unseen text, and attach an honest confidence statement
to it, for a blog post whose central figure is a line graph of category
share across decades.

## Locked decisions

| Decision | Value | Rationale |
|---|---|---|
| Decade aggregation | **Balanced quota pooling**, m = 250 raw rows per text | Equalises each text's vote AND gives every vote enough signal; every decade point then carries the same error bar. Quota is in RAW extracted rows (junk unknown pre-gate). All 16 anchor texts have >= 314 raw rows. |
| Reported proportions | **Among non-junk only** (DT+NDT+IE = 100%) | "Of the sentences that explain animals, what share is teleological." Junk rate reported separately as data quality. Makes gate *evenness* matter far more than gate recall level. |
| Folds | **6**, held-out sets in `eval/folds.json` | Balanced on raw rows (1888-2359) with the 5 DT-rich works spread across 5 folds so no fold's training set is starved of DT (train-side DT 469-638). |
| Seeds | **1 per config for folds**, 2 per config during search | Seed noise is known to exceed config deltas in this project; 2 seeds during selection guards against that, 1 is enough per fold since fold variation supplies the randomness. |
| Bias policy | **Disclose, never correct** | Matches the decision in `eval/bert_cascade_evolution.md`. |
| Dev text | **The Reign of Law** (169 non-junk), confirm on **Darwiniana** (73) | RoL is larger and currently weaker (0.775 vs 0.863) so more headroom + less noise. Darwiniana is a regression check, applied FLEXIBLY: significant gain on RoL + no large regression on Darwiniana is acceptable. |
| Ground truth | `deploy_tag` (GPT-5.4 d_v3) | Validated at ~91% vs the 49-row human golden set. No further human labelling this round. |
| Dropped | synthetic decade buckets, bootstrapping, scaling table, calibration regression, Spearman-by-year, quantification corrections, human validation, BHL out-of-anchor check | Sampling noise is small relative to classifier bias; classifier bias is the only quantity of interest. |

## Stage-2 candidates (the thing being optimised)

Current stage 2 is MacBERTh `nonjunk_3way` — unseen-text accuracy 0.775 (RoL)
/ 0.863 (Darwiniana). Candidates to beat it, all Qwen3-0.6B LoRA reusing what
worked for the junk gate (r32/alpha64, attn target modules, lr 2e-4,
`A_structured`-style prompt, max_length 384, 4 epochs):

1. **LoRA 3-way** — direct replacement for MacBERTh stage 2.
2. **LoRA 4-way** — DT/NDT/IE/**junk**, run *after* the junk gate. Gives the
   second stage a way to reject the junk the gate leaked (leakage is ~19%),
   instead of being forced to assign every leaked row to a real category.
   Marcus's idea; promising because leakage currently converts 1:1 into
   proportion error.

Selection metric is **proportion accuracy on the held-out text**, not pooled
accuracy: signed per-category error, then max absolute category error.

## Headline statistics to produce

1. **Signed bias per category** — mean and sd of (predicted - true) share
   across the 16 held-out texts. sd is the error bar.
2. **Within-text ranking accuracy** — how often the DT/NDT/IE ordering within
   a text is recovered; and the pairwise version, binned by true gap.
3. **Across-text ranking accuracy** — for each of the 120 text pairs and each
   category, is the ordering right, binned by true gap. This is the evidence
   for decade-to-decade comparisons and is *conservative*: real decade
   buckets pool ~10 texts, so errors partly cancel.
4. **Stage attribution** — perfect-gate+real-stage2, real-gate+perfect-stage2,
   real+real. Free from the same inference run.
5. Secondary: same numbers for the pooled **teleology (DT+NDT) vs
   essentialism (IE)** binary, which should be visibly tighter and is the
   likely headline of the blog post.

Report ranking claims as observed rate + Wilson interval over 16 texts /
48 within-text pairs / 360 across-text comparisons. Do NOT claim "95%
confident" from 16 texts.

## Compute plan

- **Local (Mac)**: all code edits, fold assignment, analysis, write-up.
- **Marlowe** (`ssh marlowe`, needs `module load slurm/slurm/25.05.2`,
  partition `preempt`, env `~/envs/lora_gate`, repo `~/teleological_essentialism`):
  the two heavy parallel bursts — stage-2 config search, then the 6-fold
  training. Minimal footprint: submit, poll, pull results.
- **Vast A4000** (`ssh vast`, torch in `/venv/main`, transformers/peft NOT yet
  installed, only 9.5G free on `/`, big volume under `/workspace` — DO NOT
  touch `/workspace/ARENA_3.0`): inference, stage attribution, any re-scoring.

## Checklist

- [x] Confirm Marlowe + Vast access
- [x] Compute balanced 6-fold assignment -> `eval/folds.json`
- [x] Write this plan of record
- [x] P0: add `nonjunk_3way` + `full4way` stages and 3/4-way prompts to `models/lora_gate/train.py`
- [x] P0: sync code to Marlowe (+ smoke test job 442905 passed)
- [x] P1: stage-2 config search — done, 16 jobs. Winner: **nonjunk_3way, r32/a64, S2_structured prompt, 4 epochs**.
- [x] P1: confirmation burst — done (jobs 442959-442966)
- [x] P1b: max_length sweep on raw text — done; **640 adopted**
- [x] P1: **FINAL STAGE-2 CONFIG**: `--stage nonjunk_3way --prompt-variant S2_structured
      --lora-r 32 --lora-alpha 64 --epochs 4 --text-column text --max-length 640`
      (Qwen3-0.6B LoRA). Gate uses the same text column and max_length with the
      junk-gate SOTA prompt `A_structured`, r32/a64, seed 42.
- [~] P2: 6-fold training **RUNNING (take 2)** — jobs 443074-443085. First
      attempt (442996-443007) failed: see log. All 6 s2_fold* done; gate_fold*
      jobs were preempted and requeued by the `preempt` partition, so some are
      re-running. VALIDATE before inference: every metrics.json must show
      max_length=640, text_column=text, and the right holdout works, and the
      matching checkpoint dir must exist.
- [ ] P3: pull checkpoints, end-to-end cascade inference on all 16 held-out texts (Vast)
- [ ] P3: stage attribution variants
- [ ] P4: `eval/evaluate_proportions.py` + results JSON
- [ ] P5: write-up

## Progress log

- 2026-08-23: **P2 attempt 1 failed, two bugs, both fixed.**
  1. Comma-delimited `--holdout-work` split three real titles that CONTAIN
     commas ('The History of Creation, Vol. 1'; 'On the Creation of Animals
     (Bridgewater VII), Vol. 1') into nonexistent works, killing folds 0-2.
     Fixed by adding `--holdout-fold`, which reads titles from eval/folds.json
     by fold name -- no delimiter involved. `load_train_pool` raised a clear
     ValueError rather than training on wrong data, so the failure was loud.
  2. The 20-min SLURM wall limit was sized for max_length=384; at 640 the
     ~11k-row gate job overruns it and was killed mid-epoch-2. New wrapper
     `/tmp/train.slurm` uses 3h. (`/tmp/smoke.slurm`, 20 min, still exists for
     short jobs.)
  Stale `_resume` checkpoints from the timed-out gates were deleted so nothing
  half-trained gets picked up.

- 2026-08-23: NOTE on the `preempt` partition: jobs can be preempted and
  requeued mid-run, so a job may appear RUNNING again after its metrics.json
  was written. Always confirm the queue is empty AND validate each run's config
  before consuming its checkpoint.

- 2026-08-23: **P1b truncation fix — max_length 640 adopted for raw text.**

  | fold | ml=384 | ml=512 | ml=640 | deploy_extract@384 |
  |---|---|---|---|---|
  | f3 acc / foldTVD | 0.808 / 0.077 | 0.831 / 0.036 | **0.852 / 0.048** | 0.870 / 0.083 |
  | f4 acc / foldTVD | 0.780 / 0.073 | 0.790 / 0.072 | **0.808 / 0.046** | 0.861 / 0.026 |

  Raw text at 640 recovers most of the truncation loss: accuracy up in both
  folds, fold-level mix error roughly halved (0.075 -> 0.047), now comparable
  to deploy_extract and better than it on Darwiniana. The residual 2-5pp
  accuracy gap is the genuine text-shift cost, matching the 2-6pp the junk
  gate showed. DECISION: train and report on raw `text` at max_length 640 --
  the deployable setting, since deploy_extract does not exist for a new book.

- 2026-08-23: plan written; folds computed; access to both machines confirmed.
- 2026-08-23: P0 done. `models/lora_gate/train.py` now takes `--stage`
  {junk_gate,nonjunk_3way,full4way} and `--holdout-work` (comma-separated for
  merged folds); `load_train_pool` accepts a list; junk rows dropped from
  train/held-out/golden for `nonjunk_3way`; added `S2_structured` and
  `S2_structured_4way` prompts; added `category_mix_metrics` to
  `models/data_utils.py` and per-work mix reporting to the run report.
  Module-level `HOLDOUT_WORK`/`STAGE` kept as defaults so `eval/ensemble_gate.py`
  and `eval/backfill_gate_proportions.py` still import cleanly.
- 2026-08-23: smoke test (job 442905, 1 epoch, nonjunk_3way, fold4 holdout):
  held-out acc=0.828 macro_f1=0.731, mix tvd=0.065. Pipeline works end to end.
- 2026-08-23: **P1 confirmation.** Winner config, both dev folds, both text columns, 2 seeds:

  | run | acc | macroF1 | foldTVD | devTVD |
  |---|---|---|---|---|
  | f4 deploy_extract (Reign of Law) | 0.861 | 0.801 | 0.026 | 0.036 |
  | f3 deploy_extract (Darwiniana) | 0.870 | 0.838 | 0.083 | 0.041 |
  | f4 text | 0.780 | 0.670 | 0.073 | 0.071 |
  | f3 text | 0.808 | 0.762 | 0.077 | 0.082 |

  On `deploy_extract` the LoRA beats MacBERTh on BOTH folds (RoL acc
  0.861 vs 0.775; Darwiniana 0.870 vs 0.863) and roughly halves mix error on
  both (0.036 vs 0.047; 0.041 vs 0.082). Confirmed, not a single-text artifact.

  On raw `text` -- the setting that actually applies to an unseen book, since
  `deploy_extract` is produced by the LLM labeling pass -- accuracy drops 6-8pp
  and mix error roughly doubles. That is far worse than the 2-6pp text-shift
  cost measured for the junk gate.

  **Diagnosis: truncation.** 27.8% of raw `text` rows exceed max_length=384
  once the ~210-token prompt is prepended, vs 2.9% of `deploy_extract` rows
  (raw text median 74 words, p90 226, max 510). The junk-gate log's finding
  that longer max_length HURTS was measured on `deploy_extract`, where
  truncation was never binding, so it does not transfer. max_length 512 (est.
  ~9% truncated) and 640 (est. 2.8%) are under test as P1b.

- 2026-08-23: **P1 COMPLETE.** Full search ranking by dev-text (Reign of Law)
  mix error, 2 seeds each:

  | config | devTVD | foldTVD | acc | macroF1 |
  |---|---|---|---|---|
  | nonjunk_3way r32 S2_structured | **0.030** | **0.025** | **0.859** | **0.796** |
  | nonjunk_3way r16 none | 0.083 | 0.067 | 0.832 | 0.767 |
  | nonjunk_3way r32 none | 0.083 | 0.059 | 0.847 | 0.788 |
  | full4way r16 S2_structured_4way | 0.092 | 0.049 | 0.793 | 0.593 |
  | full4way r32 S2_structured_4way | 0.092 | 0.052 | 0.782 | 0.604 |
  | nonjunk_3way r16 S2_structured | 0.098 | 0.049 | 0.836 | 0.766 |
  | full4way r32 none | 0.149 | 0.055 | 0.815 | 0.607 |
  | full4way r16 none | 0.176 | 0.086 | 0.814 | 0.612 |

  MacBERTh baseline on the dev text: devTVD=0.047, acc=0.775, macroF1=0.725.

  **The 4-way stage-2 idea is rejected.** Every full4way run loses on accuracy
  (0.78-0.82 vs 0.86) and collapses on macro-F1 (0.59-0.61 vs 0.80). Reason: a
  4-way stage 2 must relearn junk rejection from the full 65%-junk corpus,
  reintroducing exactly the gradient dilution the cascade exists to avoid --
  the same effect documented in eval/bert_cascade_evolution.md's original
  hypothesis. Giving stage 2 a junk option does not pay for the signal it costs.
  Not carried into the 6-fold.

  Rank 32 > rank 16 (matches the junk-gate finding). The S2 prompt moves the
  MIX a lot while barely moving accuracy -- consistent with the project's
  standing finding that prompting mainly reshapes which categories survive.

  CAVEAT: seed spread on the winner's devTVD is 0.036, larger than the mean --
  the dev text has only 169 non-junk rows. Fold-level TVD (570 rows) is the
  more stable number and the winner leads there too.

- 2026-08-23: P1 partial results (3-way arm complete, 4-way still running).
  Ranked by mix error on the dev text (Reign of Law), 2 seeds each:
  `nonjunk_3way_r32_S2_structured` devTVD=0.021 (spread .018) acc=0.854;
  `..._r32_none` devTVD=0.062 acc=0.853; `..._r16_S2_structured` devTVD=0.098;
  `..._r16_none` devTVD=0.083. MacBERTh baseline on the same text: devTVD=0.047
  acc=0.775. So rank 32 > rank 16 (matches the junk-gate finding), the S2
  prompt helps the MIX substantially while barely moving accuracy, and the best
  LoRA config roughly halves mix error vs. MacBERTh while adding ~8pp accuracy.
- 2026-08-23: P1 search burst submitted — 8 configs x 2 seeds (42/7) on fold4:
  {nonjunk_3way, full4way} x {r16a32, r32a64} x {S2 prompt, no prompt}, 4 epochs.
  SLURM job submitter is `~/search.sh` on Marlowe; wrapper `/tmp/smoke.slurm`.
  NOTE: `/tmp/smoke.slurm` is on the Marlowe login node and may vanish on
  reboot -- it is a 6-line sbatch wrapper forwarding "$@" to train.py, trivially
  recreatable (see `models/lora_gate/submit.slurm` for the same pattern; the
  only change is `#SBATCH --account=marlowe-m000151` plus a 20-min limit).
