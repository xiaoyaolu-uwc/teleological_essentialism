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
3. **Precision** (junk leakage into non_junk) — already the existing `junk`
   per-class precision, no new computation needed

Targets (held-out text, Reign of Law — see `LORA_JUNK_GATE_PLAN.md` §9):
per-category recall ≥ 0.65 on **each** of DT/NDT/IE, evenness ≤ 0.10, junk
precision ≥ BERT's 0.752 (no regression on leakage).

## Status snapshot

*(update this paragraph each review)*

**As of this entry: baseline established, no LoRA run yet evaluated under
the new framing.** BERT's measured baseline on Reign of Law: per-category
recall DT=0.483/NDT=0.496/IE=0.333, evenness=0.162, junk precision=0.752.
The hyperparameter sweep (6 runs) and prompt-variant runs (6 runs) are
queued/running on Marlowe as of this writing but not yet backfilled with
proportion metrics or reviewed. Next step once they land: run
`backfill_gate_proportions.py` then `compare_lora_sweep.py`, fill in the
table below, and pick the greedy-search winner to carry into the next
sweep stage.

## Experiments

Status: `queued` (submitted, not yet finished) / `done` (results filled in
below). Config column only lists what differs from defaults (lr=2e-4,
r=16/alpha=32, attn-only, no oversample, prompt=current, max_length=384).

| Run name | Status | Config (vs. default) | Held DT/NDT/IE recall | Held evenness | Golden DT/NDT/IE recall | Notes |
|---|---|---|---|---|---|---|
| *(BERT baseline)* | done | MacBERTh, not LoRA | 0.48 / 0.50 / 0.33 | 0.162 | 0.40 / 0.71 / 0.67 | Reference floor — derived from `bert_cascade` metrics.json, no rerun. IE survivor share 16.0%→11.4%, the known essentialism distortion, now quantified. |
| `junk_gate_lora_v1` | done | (first LoRA run, pre-seed-control) | — | — | — | Golden non_junk recall 0.60 (tied BERT's archived number, later shown to be noise — see Findings). Predates per-category metrics; not worth backfilling (pre-seed, not comparable to sweep). |
| `junk_gate_lora_lr1e4` | queued | lr=1e-4 | | | | LR sweep step 1/3 |
| `junk_gate_lora_lr2e4` | queued | (= defaults, seed=42) | | | | LR sweep step 2/3; also serves as `prompt=current` baseline |
| `junk_gate_lora_lr3e4` | queued | lr=3e-4 | | | | LR sweep step 3/3 |
| `junk_gate_lora_r32a64` | queued | r=32, alpha=64 | | | | rank/alpha step |
| `junk_gate_lora_attnmlp` | queued | target_modules=attn_mlp | | | | target-module step |
| `junk_gate_lora_oversample` | queued | oversample=True | | | | class-weighting step |
| `junk_gate_lora_prompt_none` | queued | prompt=none | | | | bare-text control |
| `junk_gate_lora_prompt_rich` | queued | prompt=rich | | | | rule-only, no examples |
| `junk_gate_lora_prompt_fewshot` | queued | prompt=fewshot | | | | 1 real-quote example pair |
| `junk_gate_lora_prompt_fewshot_multi` | queued | prompt=fewshot_multi | | | | 2 real-quote example pairs |
| `junk_gate_lora_prompt_fewshot_ml640` | queued | prompt=fewshot, max_length=640 | | | | truncation check vs. 384 version |
| `junk_gate_lora_prompt_fewshot_multi_ml640` | queued | prompt=fewshot_multi, max_length=640 | | | | truncation check vs. 384 version |

## Findings

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

## Backlog

- [ ] Finish greedy hyperparameter sweep (LR → rank/alpha → target modules
      → class-weighting), each step keeping the prior step's winner
- [ ] Finish prompt-variant comparison (none/rich/fewshot/fewshot_multi,
      plus the two max_length=640 truncation checks)
- [ ] Combine the sweep winner's hyperparameters with the prompt winner
      into one run
- [ ] Multi-seed reliability pass on the combined-winner config before
      trusting small deltas
- [ ] Decide whether Qwen3-1.7B is worth trying if 0.6B plateaus below
      target
- [ ] Investigate the duplicate golden-row join artifact flagged earlier
      (three golden rows resolved to the same extract, one with `work=''`)
      — deprioritized as minor, not yet fixed
