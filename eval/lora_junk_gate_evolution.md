# LoRA Junk Gate — Evolution Log

Tracks what we've tried for the LoRA-fine-tuned junk gate, what's queued,
and the results, so a new session (or a future you) can see the current
state in one place without re-deriving it. Update this whenever a batch of
runs finishes: run `eval/compare_lora_sweep.py` (and
`eval/backfill_gate_proportions.py` first, for any run trained before
category-proportion metrics existed), then fill in the Experiments table
and add anything durable to Findings.

See `LORA_JUNK_GATE_PLAN.md` for the full plan/decisions; this doc is the
running log, that one is the design doc.

---

## Objective

Not pooled accuracy — accurate category *proportions* across a text. Three
metrics anchor evaluation (see `models/data_utils.py`'s
`gate_proportion_metrics`):

1. **Per-category recall** (DT/NDT/IE individually) through the gate
2. **Recall evenness** (max−min across the three) — low is good; a gate
   that's uneven distorts the category mix even if pooled recall looks fine
3. **Precision** (junk leakage into non_junk, i.e. purity of the kept set
   actually passed downstream) — **CORRECTED**: this is `non_junk_precision`
   (precision of the *non_junk* call), not the `junk`-class precision this
   doc used throughout. See the "Precision metric was wrong" Finding below
   — `junk` precision measures over-aggressive discarding of good rows, a
   different (and less important, per marcus's stated priority) failure
   mode than junk leaking into the kept set.

Targets (held-out text, Reign of Law — see `LORA_JUNK_GATE_PLAN.md` §9):
per-category recall ≥ 0.65 on **each** of DT/NDT/IE, evenness ≤ 0.10,
**non_junk** precision ≥ BERT's 0.745 (no regression on leakage) — not
0.752, which was BERT's `junk`-class precision, a different number that
happens to be close by coincidence for BERT specifically.

## Status snapshot

*(update this paragraph each review)*

**As of this entry: hyperparameter sweep + prompt loop (all 5 rounds) +
a broader multi-seed reliability pass are done and logged.** Best
hyperparameter-only result at seed=42 was `r32a64` (0.66/0.82/0.63,
evenness=0.19); combining all four individual winners (`combined_v1`)
does not beat that (0.62/0.82/0.59, evenness=0.23) — see Findings.

**Update — multi-seed reliability pass reverses the earlier "superseded
by prompting" framing.** The prompt loop's `B_structured_antiheuristic`
initially looked dominant (0.86/0.88/0.70, evenness=0.17, beating every
row here) but that result was shown not to hold up under reseeding — see
`eval/lora_prompt_evolution.md`'s "Broader multi-seed reliability pass".
`r32a64` was reseeded at 7/123 alongside B and `A_structured`: across 3
seeds, `r32a64` has the highest or tied-highest mean recall on every
category (DT .703, NDT .827, IE .607) **and** by far the tightest spread
(IE spread .07 vs. .26-.30 for every prompt-based candidate) — more
robust than any prompt-loop candidate once seed noise is accounted for.
`r32a64` is therefore the best-supported single config across the whole
project right now, not superseded by prompting after all. No candidate
(hyperparameter or prompt) clears the evenness ≤0.10 target on average —
all cluster at .22-.26. Next reasonable step: test `r32a64`'s
hyperparameters combined with `A_structured` or `B`'s prompt at multiple
seeds (the one integration run tried, Round 5, used `B` at a single seed
and predates this finding).

**Final resolution**: that integration test was run properly (5 seeds:
42/7/123/99/7777) in `eval/lora_prompt_evolution.md`'s "Final resolution"
section. `r32a64`+`A_structured`, ensembled across the 5 seeds, is the
final adopted SOTA: DT=.586, NDT=.796, IE=.593, evenness=.210,
non_junk_precision=.759 — beats BERT's non_junk_precision (.745) while
far exceeding it on every recall number. Cursory (non-iterated) baselines
for full fine-tuning and Qwen3-1.7B were also run; neither clearly beats
this config (full fine-tune: non_junk_precision=.651, worse; 1.7B:
non_junk_precision=.758, ties but doesn't beat on recall) — see that
section for the full comparison. Accessible write-up for external
stakeholders: `JUNK_GATE_PROGRESS_REPORT.md`.

## Experiments

Status: `queued` (submitted, not yet finished) / `done` (results filled in
below). Config column only lists what differs from defaults (lr=2e-4,
r=16/alpha=32, attn-only, no oversample, prompt=current, max_length=384).

| Run name | Status | Config (vs. default) | Held DT/NDT/IE recall | Held evenness | Golden DT/NDT/IE recall | Notes |
|---|---|---|---|---|---|---|
| *(BERT baseline)* | done | MacBERTh, not LoRA | 0.48 / 0.50 / 0.33 | 0.162 | 0.40 / 0.71 / 0.67 | Reference floor — derived from `bert_cascade` metrics.json, no rerun. IE survivor share 16.0%→11.4%, the known essentialism distortion, now quantified. |
| `junk_gate_lora_v1` | done | (first LoRA run, pre-seed-control) | — | — | — | Golden non_junk recall 0.60 (tied BERT's archived number, later shown to be noise — see Findings). Predates per-category metrics; not worth backfilling (pre-seed, not comparable to sweep). |
| `junk_gate_lora_lr1e4` | done | lr=1e-4 | 0.59/0.76/0.56 | 0.21 | 0.30/0.79/0.83 | Strictly better than lr2e4 default on every held-out number. LR sweep winner. |
| `junk_gate_lora_lr2e4` | done | (= defaults, seed=42) | 0.59/0.72/0.52 | 0.20 | 0.50/0.79/0.83 | Baseline default; also the shared base for all Round 0 prompt-variant runs. |
| `junk_gate_lora_lr3e4` | done | lr=3e-4 | 0.59/0.89/0.41 | 0.49 | 0.20/0.71/0.83 | NDT to .89 but IE craters to .41 — worst evenness of the sweep; textbook case of the NDT-rockets/IE-collapses pattern. |
| `junk_gate_lora_r32a64` | done | r=32, alpha=64 | 0.66/0.82/0.63 | 0.19 | 0.50/0.71/0.83 | Wins on every held-out number vs. default — the strongest single hyperparameter dimension. |
| `junk_gate_lora_r32a64_seed7` | done | r=32, alpha=64, seed=7 | 0.76/0.78/0.56 | 0.22 | 0.50/0.71/0.67 | Reseed check. |
| `junk_gate_lora_r32a64_seed123` | done | r=32, alpha=64, seed=123 | 0.69/0.88/0.63 | 0.26 | 0.40/0.79/0.83 | Reseed check. 3-seed mean: 0.703/0.827/0.607, evenness .22, IE spread only .07 — the tightest/most robust config found across the whole project once reseeded; see Findings and `eval/lora_prompt_evolution.md`'s broader reliability pass. |
| `junk_gate_lora_attnmlp` | done | target_modules=attn_mlp | 0.66/0.76/0.59 | 0.17 | 0.30/0.79/1.00 | Beats default on all three categories + evenness; golden IE=1.00 is a small-n golden artifact, not trustworthy. |
| `junk_gate_lora_oversample` | done | oversample=True | 0.62/0.72/0.56 | 0.16 | 0.40/0.71/0.83 | Smallest effect of the four hyperparameter dimensions, but real and consistent. |
| `junk_gate_lora_combined_v1` | done | lr=1e-4, r=32/a=64, attn_mlp, oversample=True | 0.62/0.82/0.59 | 0.23 | 0.60/0.79/0.83 | All four individual winners combined. Does **not** beat r32a64 alone on any held-out axis except NDT (tied) — see Findings, gains didn't stack. |
| `junk_gate_lora_combined_v2` | done | same as v1, oversample=False | 0.45/0.77/0.41 | 0.36 | 0.30/0.71/0.83 | Ablation: dropping oversample from the combo made it much worse, confirming oversample *was* contributing within the combo — the combo's shortfall vs. r32a64 alone isn't oversample's fault. |
| `junk_gate_lora_prompt_none` | done | prompt=none | 0.45/0.68/0.52 | 0.23 | 0.40/0.71/0.67 | Bare-text control; worst DT of the prompt set. |
| `junk_gate_lora_prompt_rich` | done | prompt=rich | 0.76/0.81/0.48 | 0.33 | 0.50/0.71/0.83 | Strong DT/NDT, IE sacrificed — same pattern as lr3e4. Superseded by Round 1's B (see `eval/lora_prompt_evolution.md`). |
| `junk_gate_lora_prompt_fewshot` | done | prompt=fewshot | 0.62/0.68/0.59 | **0.09** | 0.60/0.71/0.83 | Best evenness in the Round-0 prompt set. |
| `junk_gate_lora_prompt_fewshot_multi` | done | prompt=fewshot_multi | 0.69/0.86/0.44 | 0.41 | 0.50/0.79/0.83 | Worst evenness of the Round-0 prompt set. |
| `junk_gate_lora_prompt_fewshot_ml640` | done | prompt=fewshot, max_length=640 | 0.59/0.63/0.44 | 0.18 | 0.50/0.71/0.83 | Truncation check: max_length=640 made *every* number worse than the 384 version, despite fewshot being barely truncated at 384 (0.4% of rows) — see Findings. |
| `junk_gate_lora_prompt_fewshot_multi_ml640` | done | prompt=fewshot_multi, max_length=640 | 0.55/0.71/0.37 | 0.34 | 0.30/0.64/0.83 | Truncation check: also worse than the 384 version on every number, despite fewshot_multi being 50.1% truncated at 384 — extending max_length did not recover the lost signal. |

**Superseded by prompting** (see `eval/lora_prompt_evolution.md` for the full loop): `junk_gate_lora_prompt_B_structured_antiheuristic` (same hyperparameter base as `lr2e4`, prompt-only change) reaches **0.86/0.88/0.70, evenness=0.17** — beats every row in this table, including every hyperparameter combination above. The hyperparameter sweep's conclusions above still stand as the best *hyperparameter-only* result, but prompting turned out to be the dominant lever, matching the finding that `rich` and `lr3e4` (both "sharpen the decision boundary" levers) show the identical NDT-rockets/IE-collapses signature.

## Findings

- **The precision metric tracked throughout this whole doc was the wrong
  one.** "Junk precision" (precision of the `junk` call: of rows labeled
  junk, how many really are) measures over-aggressive discarding of good
  non_junk rows. It does NOT measure junk leaking into the kept set — that
  requires `non_junk_precision` (precision of the `non_junk` call: of rows
  labeled non_junk and passed downstream, how many are truly non_junk),
  the mirror-image metric. Marcus's stated priority ("if we have low
  precision, that is a sin, because we necessarily get things wrong in the
  final output") is unambiguously about the latter. The two happened to be
  close for BERT (junk=0.752, non_junk=0.745) which is why this went
  unnoticed, but they diverge sharply for the LoRA sweep: `r32a64`'s
  junk-precision looked great (0.85-0.88 across seeds) while its
  non_junk_precision is 0.670-0.706 per seed (0.732 ensembled) — **below
  BERT's 0.745 on every single seed, and still below it even ensembled.**
  This means the "LoRA beats BERT" framing throughout this document was
  built on the wrong axis; on the metric that actually matters most now,
  LoRA has not yet clearly beaten BERT. See
  `eval/lora_prompt_evolution.md`'s precision-focused reliability pass for
  the full multi-config comparison on the corrected metric.
- **Golden's 30 non-junk rows are too noisy for single-run comparisons.**
  Retraining BERT's junk gate fresh (same config, different random
  init/shuffle) swung golden non_junk recall from 0.60 (archived run) to
  0.73 — a bigger difference than most hyperparameter changes we're testing
  for. `--seed` was added to `train.py` specifically so sweep runs are
  comparable to each other; held-out text (169 non-junk rows) is far more
  stable and is now the primary comparison surface.
- **The gate's distortion isn't narrowly an essentialism problem in the
  per-category-recall view** — DT (0.483) and NDT (0.496) survive at
  similar rates on held-out, IE lags at 0.333. But in the *mix* view, IE is
  the one whose *share* visibly shrinks (16.0%→11.4%) because it's the
  smallest true category to begin with — recall and share-of-mix can tell
  different parts of the story, worth checking both.
- **The four individual hyperparameter winners don't stack.** Combining
  lr=1e-4 + r32/alpha64 + attn_mlp + oversample (`combined_v1`) scores
  0.62/0.82/0.59 (evenness=0.23) — worse than `r32a64` alone on every
  held-out axis except NDT (tied), despite each dimension winning
  individually. Ablating oversample back out (`combined_v2`) made it much
  worse still (0.45/0.77/0.41), so oversample wasn't the problem — the
  likely story is too much simultaneous capacity/adapter surface
  (attn_mlp roughly doubles trainable params, stacked with r32 and a lower
  lr) overshooting rather than composing additively. Lesson: greedy
  single-dimension wins don't automatically combine; a combined config
  needs its own validation, not just concatenation.
- **Prompting turned out to dominate every hyperparameter lever tested
  here.** See `eval/lora_prompt_evolution.md` — `B_structured_antiheuristic`
  (0.86/0.88/0.70, evenness=0.17) beats every row in this table by a wide
  margin, at the *default* hyperparameters. `rich` and `lr3e4` both show
  the identical NDT-rockets/IE-collapses signature, suggesting that lever
  and this lever were hitting the same underlying failure mode from
  different directions.

## Backlog

- [x] Finish greedy hyperparameter sweep (LR → rank/alpha → target modules
      → class-weighting), each step keeping the prior step's winner
- [x] Finish prompt-variant comparison (none/rich/fewshot/fewshot_multi,
      plus the two max_length=640 truncation checks)
- [x] Combine the sweep winner's hyperparameters — done, did not beat the
      single best dimension (see Findings)
- [x] Combine the prompt-loop winner (`B`) with `r32a64` — done (Round 5 of
      `eval/lora_prompt_evolution.md`), did not stack, regressed vs. B alone
- [x] Multi-seed reliability pass — done on the top 4 candidates
      (baseline, `A_structured`, `B`, `r32a64`); reverses the earlier
      "superseded by prompting" conclusion — `r32a64` is now the
      best-supported config, not `B` (see Findings)
- [ ] Test `r32a64`'s hyperparameters combined with `A_structured` or `B`'s
      prompt at multiple seeds (only the single-seed `B` + `r32a64`
      combination has been tried so far, and predates the reliability pass)
- [ ] Decide whether Qwen3-1.7B is worth trying, now that the measured
      prompting effect is smaller than first thought and `r32a64` (a
      hyperparameter change) is the more robust lever found so far
- [ ] Investigate the duplicate golden-row join artifact flagged earlier
      (three golden rows resolved to the same extract, one with `work=''`)
      — deprioritized as minor, not yet fixed
