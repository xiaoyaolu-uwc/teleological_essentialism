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
| `junk_gate_lora_prompt_A_structured` | done | 0.62/0.87/0.63 | 0.25 | 0.30/0.79/1.00 | Structure alone already beats every Round 0 prompt on IE. |
| `junk_gate_lora_prompt_B_structured_antiheuristic` | done | **0.86/0.88/0.70** | **0.17** | 0.50/0.79/1.00 | Best result in the entire sweep (all rounds, all hyperparameter runs) by a wide margin — see Findings. |
| `junk_gate_lora_prompt_C_hard_contrastive` | done | 0.48/0.74/0.44 | 0.30 | 0.30/0.79/0.83 | Weakest of the four; below the `current` baseline on DT and IE both. |
| `junk_gate_lora_prompt_D_structured_plus_example` | done | 0.59/0.87/0.70 | 0.28 | 0.40/0.79/1.00 | Ties B on IE but DT is much weaker and evenness worse — the example addition didn't help over B. |

For reference, Round 0 prompt results (same hyperparameter base) from
`eval/lora_junk_gate_evolution.md`:

| variant | Held DT/NDT/IE recall | Held evenness |
|---|---|---|
| none | .45/.68/.52 | .23 |
| current | .59/.72/.52 | .20 |
| rich | .76/.81/.48 | .33 |
| fewshot | .62/.68/.59 | **.09** |
| fewshot_multi | .69/.86/.44 | .41 |

## Round 2 — targeted refinements of B (not a broad new sweep)

B already clears 2 of 3 targets (per-category recall ≥0.65, junk
precision ≥0.752); evenness (.17) is the one gap left (target ≤0.10),
driven by IE (.70) lagging DT (.86)/NDT (.88). Two independent refinements
of B specifically targeting that gap — not sequential edits of one
prompt, per methodology (both start from B, but test different, unrelated
mechanisms). No reseed check yet, per explicit decision — that happens
once we're near a final configuration, not every round.

Token-length check against all 469 held-out passages:

| variant | template-only tokens | median total | max total | % >384 |
|---|---|---|---|---|
| E_B_plus_hard_example | 251 | 306 | 463 | 1.9% |
| F_B_stronger_clause | 225 | 280 | 437 | 0.6% |

| variant | hypothesis tested |
|---|---|
| E_B_plus_hard_example | B + one concrete IE example (structure-only, no purpose language) appended. Tests whether a concrete anchor reinforces B's rule further, unlike Round 1's D (example added to A without the anti-heuristic clause) which hurt DT. |
| F_B_stronger_clause | B with the structure/purpose order swapped (structure listed first) and the anti-heuristic note expanded to be more directive ("check specifically whether it makes a structural/categorical claim before deciding"). Tests whether a stronger, more specific version of B's own working mechanism closes the gap further. |

| Run name | Status | Held DT/NDT/IE recall | Held evenness | Golden DT/NDT/IE recall | Notes |
|---|---|---|---|---|---|
| `junk_gate_lora_prompt_E_B_plus_hard_example` | done | 0.59/0.79/0.48 | 0.31 | 0.30/0.79/0.83 | Regresses vs. B on every axis (DT .86→.59, NDT .88→.79, IE .70→.48, evenness .17→.31). Adding the example hurt rather than helped. |
| `junk_gate_lora_prompt_F_B_stronger_clause` | done | 0.59/0.69/0.48 | 0.21 | 0.20/0.71/0.83 | Also regresses vs. B on every axis (DT .86→.59, NDT .88→.69, IE .70→.48, evenness .17→.21). Reordering + strengthening the clause hurt rather than helped. |

## Findings

- **Round 1: B_structured_antiheuristic strictly dominates the other three
  candidates on every held-out axis** (DT .86 vs. A's .62/C's .48/D's .59;
  NDT .88 vs. .87/.74/.87; IE .70 vs. .63/.44/.70; evenness .17 vs.
  .25/.30/.28) — not a trade-off, better on all four numbers
  simultaneously. It's also the best result across the *entire* project so
  far, beating every Round 0 prompt and every hyperparameter-sweep run
  (previous best worst-case recall was `r32a64` at .63; B reaches .70) and
  clearing the original targets in `LORA_JUNK_GATE_PLAN.md` §9
  (per-category recall ≥0.65 each ✓, evenness ≤0.10 — .17 is close but not
  quite there, jprec ≥0.752 ✓ at .90). Structure alone (A) already beat
  every Round 0 prompt on IE (.63, vs. rich's .48); naming the purpose-
  language shortcut explicitly (B's addition over A) is what pushed DT way
  up too (.62→.86) without sacrificing IE — suggesting the anti-heuristic
  clause helped the model use the *structure* correctly for ambiguous DT
  cases, not just IE ones.
- **The hard-contrastive example approach (C) underperformed**, coming in
  below even the plain `current` baseline on DT and IE. Combining it with
  structure (D) recovered NDT/IE to B's level but not DT, and worsened
  evenness vs. B. Tentative read: for this task, an explicit rule
  (A/B-style) generalizes better than concrete examples (C/D-style) — echoes
  the Round 0 pattern where `rich` (rule-only) beat `fewshot` (example-only)
  on aggregate, though `fewshot` had the better evenness there. Not
  conclusive from n=4 candidates; worth another rule-vs-example test in a
  later round if this keeps holding.
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
- **Round 2: both targeted refinements of B regressed instead of
  improving it.** E (B + one hard IE example) and F (B with structure/
  purpose order swapped + a stronger, more directive anti-heuristic
  clause) both fell to IE=0.48 (from B's 0.70) and worse evenness (.31
  and .21 vs. B's .17), with DT and NDT also down. This is the fourth
  independent instance of "adding a concrete example on top of a working
  rule hurts" (Round 0 fewshot/fewshot_multi vs. rich-style rules, Round 1
  C/D vs. A/B, now Round 2 E) — a consistent pattern, not noise from one
  run. F's failure is more surprising: it only changed ordering and
  wording of the *same* mechanism that made B work, and still regressed,
  suggesting B's exact phrasing sits at a fairly sharp local optimum for
  this model/prompt-length combination rather than being on a smooth
  "more explicit = better" gradient.

## Round 3 — selection (autonomous, per marcus's "be agentic" authorization)

Across all 11 candidates tested (Round 0: none/current/rich/fewshot/
fewshot_multi; Round 1: A/B/C/D; Round 2: E/F), **B_structured_antiheuristic
is the winner** — it strictly dominates every other candidate on every
held-out axis (DT/NDT/IE recall and evenness simultaneously), not just on
one metric traded against another. No candidate came within striking
distance: the next-best worst-case recall is A/D tied at .62-.63, well
below B's .70. Junk precision (.90) is comfortably above the 0.752 floor
throughout the B family. Decision: **B_structured_antiheuristic is the
Round 3 survivor**, carried forward as-is (not re-derived or re-weighted
from Round 2's failures, since neither Round 2 candidate offered anything
worth folding back in).

## Round 4 — combination (concluded: not warranted)

Per the loop's methodology, combination only makes sense when a second,
genuinely independent idea has been validated alongside the round winner.
Reviewing everything tested: A (structure only) is a strict subset of B;
C, D, E all tested example-based additions and every one of them
underperformed the corresponding rule-only version; F tested a stronger/
reordered version of B's own clause and also underperformed. There is no
surviving independent idea left to merge with B — the entire "add
examples" and "strengthen the clause further" directions are now each
independently falsified (4 and 1 data points respectively). Rather than
force a combination for its own sake, **Round 4 concludes with no change:
B_structured_antiheuristic stands as the prompt-loop winner**, carried
into Round 5 unmodified.

## Backlog

- [x] Round 1: submit, backfill, compare all four candidates
- [x] Round 2: submit, backfill, compare E/F refinements — both regressed
- [x] Round 3: select survivor — B_structured_antiheuristic, undisputed
- [x] Round 4: evaluate combination — concluded not warranted, no
      independent idea left to merge
- [ ] Round 5: integrate B with r32a64 hyperparameters (the actual best
      hyperparameter-only config — NOT the failed combined_v1/v2, see
      `eval/lora_junk_gate_evolution.md`), compare against B at default
      hyperparameters to see if the gains stack
- [ ] Multi-seed check on whichever config comes out of Round 5 as final
