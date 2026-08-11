# LoRA Junk Gate — Prompt Evolution Log

Tracks the prompt-iteration loop specifically (separate from
`eval/lora_junk_gate_evolution.md`, which tracks the hyperparameter side).
Same update convention: run `eval/backfill_gate_proportions.py` then
`eval/compare_lora_sweep.py`, fill in the table, add durable findings.

## Objective

Fix the pattern diagnosed from Round-0 sweep data: levers that sharpen the
model's decision boundary (higher LR, bigger adapter, more assertive
prompts) all improve DT/NDT recall by leaning on a shared surface cue
(purposive/teleological language) while IE — whose true criterion is a
categorical/structural claim that often has *no* purpose language —
collapses. Goal: raise IE recall specifically, without giving back DT/NDT,
and get evenness very small. Not chasing a numeric bar rigidly (see
Findings) — evaluating candidates against real data each round instead.

## Methodology

Each round tests several **independent** hypotheses, not variations on one
prompt. A round closes with either a documented winner or an explicit
dead-end conclusion. Survivors only get combined *after* independent
validation (never mid-round). Every candidate's token length is checked
against real held-out passages before submitting (lesson from
`fewshot_multi`'s 50%-truncated-at-384 discovery) — see Round 1 check
below. Fixed hyperparameter base per round: lr=2e-4, r=16/alpha=32,
target=attn, no oversample, max_length=384, seed=42
(`junk_gate_lora_lr2e4` — chosen for direct comparability with existing
prompt data (none/current/rich/fewshot/fewshot_multi), not the sweep's
strongest hyperparameter config, since the hyperparameter side is being
finalized independently — see `eval/lora_junk_gate_evolution.md`). The
round's prompt winner gets re-tested against that final hyperparameter
config in a later integration round.

## Round 1 — independent directions

Token-length check against all 469 held-out passages (Qwen3-0.6B
tokenizer, no truncation):

| variant | template-only tokens | median total | max total | % >384 |
|---|---|---|---|---|
| A_structured | 203 | 258 | 415 | 0.4% |
| B_structured_antiheuristic | 213 | 268 | 425 | 0.6% |
| C_hard_contrastive | 162 | 217 | 374 | 0.0% |
| D_structured_plus_example | 200 | 255 | 412 | 0.4% |

All four safely fit within max_length=384 for effectively all rows.

| variant | hypothesis tested |
|---|---|
| A_structured | Structure alone (categorize → ground in purpose OR structure) fixes IE without an anti-heuristic clause or examples. |
| B_structured_antiheuristic | Same structure, plus a clause directly naming the purpose-language shortcut. Isolates whether naming the bias helps on top of structure. |
| C_hard_contrastive | Short rule + one hard-negative pair: an IE example with zero purpose language, and a junk example that uses purpose-adjacent language but makes no categorical claim. Tests examples-as-signal for the IE boundary specifically. |
| D_structured_plus_example | A's structure + C's single IE-anchored example (not two, to respect length). The "explanation + examples, mind length" direction. |

| Run name | Status | Held DT/NDT/IE recall | Held evenness | Golden DT/NDT/IE recall | Notes |
|---|---|---|---|---|---|
| `junk_gate_lora_prompt_A_structured` | queued | | | | |
| `junk_gate_lora_prompt_B_structured_antiheuristic` | queued | | | | |
| `junk_gate_lora_prompt_C_hard_contrastive` | queued | | | | |
| `junk_gate_lora_prompt_D_structured_plus_example` | queued | | | | |

For reference, Round 0 prompt results (same hyperparameter base) from
`eval/lora_junk_gate_evolution.md`:

| variant | Held DT/NDT/IE recall | Held evenness |
|---|---|---|
| none | .45/.68/.52 | .23 |
| current | .59/.72/.52 | .20 |
| rich | .76/.81/.48 | .33 |
| fewshot | .62/.68/.59 | **.09** |
| fewshot_multi | .69/.86/.44 | .41 |

## Findings

- **No hard numeric selection bar set for Round 3** — per explicit
  decision, candidates get judged against real Round 1 data together
  rather than a bar picked before seeing results.
- **max_length=640 does not recover truncated content — it makes things
  worse.** Two independent controlled pairs (`fewshot`@384 vs. @640, and
  `fewshot_multi`@384 vs. @640 — same prompt, only max_length differs)
  both regressed on every metric at 640, despite `fewshot_multi` having
  50.1% of held-out rows genuinely truncated at 384. Mechanism unclear
  (padding/pooling should be robust to this for a causal decoder model
  with last-real-token pooling); not chasing it further since max_length
  is fixed at 384 by explicit constraint regardless. See
  `eval/lora_junk_gate_evolution.md` for the truncation-rate measurement.

## Backlog

- [ ] Round 1: submit, backfill, compare all four candidates
- [ ] Round 3: select survivors against Round 1 data with the user
- [ ] Round 4: combine top ≤2 survivors into one prompt
- [ ] Round 5: integrate round winner with the finalized combined
      hyperparameter config from `eval/lora_junk_gate_evolution.md`,
      multi-seed check before calling it done
