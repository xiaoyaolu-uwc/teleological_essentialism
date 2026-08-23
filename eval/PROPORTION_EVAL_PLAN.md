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
- [~] P1: stage-2 config search — **RUNNING**, Marlowe jobs 442916-442931 (16 jobs), ids in `~/search_jobids.txt` on Marlowe
- [ ] P1: confirm top configs on Darwiniana fold
- [ ] P1: pick stage-2 config, record it here
- [ ] P2: 6-fold training, gate + stage 2 — Marlowe burst 2
- [ ] P3: pull checkpoints, end-to-end cascade inference on all 16 held-out texts (Vast)
- [ ] P3: stage attribution variants
- [ ] P4: `eval/evaluate_proportions.py` + results JSON
- [ ] P5: write-up

## Progress log

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
- 2026-08-23: P1 search burst submitted — 8 configs x 2 seeds (42/7) on fold4:
  {nonjunk_3way, full4way} x {r16a32, r32a64} x {S2 prompt, no prompt}, 4 epochs.
  SLURM job submitter is `~/search.sh` on Marlowe; wrapper `/tmp/smoke.slurm`.
  NOTE: `/tmp/smoke.slurm` is on the Marlowe login node and may vanish on
  reboot -- it is a 6-line sbatch wrapper forwarding "$@" to train.py, trivially
  recreatable (see `models/lora_gate/submit.slurm` for the same pattern; the
  only change is `#SBATCH --account=marlowe-m000151` plus a 20-min limit).
