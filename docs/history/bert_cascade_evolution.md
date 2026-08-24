# Two-Stage Cascade Evolution Log

Tracks the junk-gate + 3-way-teleology cascade, proposed as an alternative to
the single 4-way MacBERTh classifier after `eval/inspect_golden_errors.py`
showed divine_teleology collapsing into junk on implicit design-argument
passages, and neither more epochs nor oversampling fixed it consistently
(see `models/train_bert.py` docstring and the single-model diagnostics under
`eval/results/bert/loo_*`).

**Hypothesis**: junk is 65% of every training batch in the 4-way setup,
diluting the gradient signal available to separate divine_teleology from
internal_essence/non_divine_teleology. Removing junk entirely for a dedicated
3-way classifier should give that separation task a near-balanced,
undiluted training signal.

**Caution carried in**: this project already tried a structurally similar
two-layer idea for the LLM prompts (`eval/two_layer_evolution.md`) and it
underperformed the single-pass prompt (61.2% vs ~70%) due to stage-1 error
propagation. This run reused the same 2 holdout texts as the single-4-way
diagnostics (Reign of Law, Darwiniana) for direct comparability, and reports
stage-1-only / stage-2-only / end-to-end numbers separately specifically to
check whether the same failure mode recurs.

## Setup

- `models/train_bert.py --stage junk_gate`: binary `{junk, non_junk}`.
- `models/train_bert.py --stage nonjunk_3way`: 3-way
  `{divine_teleology, non_divine_teleology, internal_essence}`, trained only
  on ground-truth non-junk rows (junk dropped from train/held-out/golden
  entirely for this stage).
- `eval/evaluate_cascade.py`: chains a `junk_gate_*` + `nonjunk3way_*`
  checkpoint pair — stage 1 predicts junk/non_junk; non_junk rows are handed
  to stage 2 for the specific tag; junk rows keep the stage-1 label directly.
  Scored against the true 4-way label with the same `metrics_from_preds`
  helper used everywhere else, so numbers are diffable against
  `eval/results/bert/loo_reign_of_law/metrics.json` and `loo_darwiniana/metrics.json`.
- 4 epochs each, same as the single-4-way baselines being compared against.

## Results

### Stage 2 (nonjunk_3way) in isolation — the clear win

| Holdout | Held-out acc / macro-F1 | Golden acc / macro-F1 | Golden DT recall |
|---|---|---|---|
| Reign of Law | 0.775 / 0.725 | **0.900 / 0.904** | **0.90** |
| Darwiniana | 0.863 / 0.864 | **0.867 / 0.863** | **0.80** |

This is the best BERT result seen anywhere in this project. Compare golden
divine_teleology recall of 0.90/0.80 here against 0.70/0.50 for the single
4-way baseline, and 0.30–0.40 for every epoch/oversampling variant tried
previously. Confirms the hypothesis directly: once junk's 65% dominance is
removed, MacBERTh separates DT/NDT/IE cleanly and generalizes well
cross-text — held-out recall is 0.70–0.95 across all three classes on both
texts, no whack-a-mole pattern.

### Stage 1 (junk_gate) in isolation — the weak link

| Holdout | Held-out acc / macro-F1 | Held-out non_junk recall | Golden non_junk recall |
|---|---|---|---|
| Reign of Law | 0.751 / 0.699 | **0.47** | 0.60 |
| Darwiniana | 0.870 / 0.664 | **0.42** | 0.67 |

The binary gate misses more than half of genuinely non-junk passages on
unseen text, routing them to junk before stage 2 ever sees them. This is
worse non-junk recall than the single 4-way model implicitly achieves for
the same distinction (junk vs. not), despite junk_gate being a strictly
simpler 2-class problem. Likely explanation: in the 4-way softmax, weak
"maybe DT" or "maybe IE" signal can still win an argmax against junk because
it's competing against 3 diluted alternatives; collapsed into one binary
junk-vs-not decision, that same weak signal has to clear a single higher bar,
and cross-entropy on the majority-junk binary problem defaults uncertain
cases to junk.

### End-to-end cascade — bottlenecked by stage 1, worse than the single 4-way baseline

| Holdout | Held-out acc / macro-F1 | Golden acc / macro-F1 | Golden DT recall |
|---|---|---|---|
| Cascade, Reign of Law | 0.706 / 0.482 | 0.653 / 0.650 | 0.40 |
| Cascade, Darwiniana | 0.865 / 0.460 | 0.633 / 0.611 | 0.30 |
| *(single 4-way baseline, for reference)* | *0.733 / 0.554 (RoL), 0.837 / 0.487 (Darw.)* | *0.673 / 0.686 (RoL), 0.653 / 0.653 (Darw.)* | *0.70 (RoL), 0.50 (Darw.)* |

The cascade's end-to-end golden DT recall (0.30–0.40) is barely different
from the failed epoch/oversampling variants, and slightly *worse* than the
plain single 4-way baseline on golden macro-F1. This is exactly the
propagation risk flagged going in: stage 2 alone would have given DT recall
of 0.80–0.90, but most true DT/NDT/IE rows never reach stage 2 because stage
1 already misclassified them as junk.

## Decision

**Do not adopt the cascade as-is.** The single 4-way model remains the better
end-to-end choice for now — it's simpler and performs at least as well as the
cascade on every metric that matters. The cascade is not a dead end, though:
stage 2 in isolation is unambiguously the best BERT result in this project,
and the fix is narrowly scoped to stage 1's recall problem, not the whole
architecture. Next actionable step, if this is worth pursuing further: retry
`junk_gate` with `--oversample` (oversampling `non_junk` specifically, cheap
to test since junk_gate is only a 2-class problem) or adjust the decision
threshold away from the default 0.5 argmax to trade some junk precision for
non_junk recall, since the failure looks like a calibration/threshold issue
on a fundamentally learnable signal (the model *can* tell DT/NDT/IE apart
extremely well once junk is out of the way) rather than a data or capacity
problem.

## Follow-up: base model, distribution shift, and proportion analysis

Three follow-up checks on the junk gate specifically, plus a reframing of
what "good enough" means given the project's actual research use (tracking
category *proportions* across texts over time, not per-row accuracy).

### bert-base-uncased vs. MacBERTh as the junk gate

`--model-name` added to `models/train_bert.py` to swap the base model.
Plain `bert-base-uncased`, same architecture/task, no historical pretraining:

| Holdout | Model | Golden acc / macro-F1 | Golden non_junk recall |
|---|---|---|---|
| Reign of Law | MacBERTh | 0.673 / 0.672 | 0.60 |
| Reign of Law | bert-base-uncased | **0.776 / 0.766** | **0.80** |
| Darwiniana | MacBERTh | 0.694 / 0.689 | 0.67 |
| Darwiniana | bert-base-uncased | 0.694 / 0.692 | 0.63 |

Plain BERT clearly wins on Reign of Law, is a wash on Darwiniana, never
loses. Historical pretraining isn't buying anything for this specific binary
distinction — evidence the junk-gate weakness is a framing/architecture
issue (binary loses the softmax "cushion" weak signal had in the 4-way
setup), not a domain-vocabulary gap. Ruled out chasing a different
historical-BERT variant as the fix.

### Raw-text distribution shift (deploy_extract-trained, evaluated on raw `text`)

`eval/evaluate_text_shift.py` re-evaluates an already-trained checkpoint (no
retraining) on the raw `text` column instead of `deploy_extract`, since
`deploy_extract` is itself an LLM-produced field — a standalone deployed
model would need to work on raw passages directly. Cost is real but modest:
roughly 2-6pp of macro-F1/recall across all 4 junk-gate configs (e.g.
MacBERTh/Reign of Law: macro-F1 0.703 → 0.685; non_junk recall 0.47 → 0.43).
Confirms the junk gate's core weakness isn't mainly this shift — it's
present already when evaluated on-distribution.

### The false-junk skew is against essentialism, not (only) divine_teleology

Re-examined at the grouping that matches the actual research question —
teleology (DT+NDT) vs. essentialism (IE) — rather than the finer 4-way
split. On held-out text (larger, more reliable n than golden):

| Model | Holdout | Teleology false-junk rate | Essentialism false-junk rate |
|---|---|---|---|
| MacBERTh | Reign of Law | 0.51 (72/142) | **0.67** (18/27) |
| bert-base | Reign of Law | 0.23 (32/142) | **0.59** (16/27) |
| MacBERTh | Darwiniana | 0.51 (26/51) | **0.73** (16/22) |
| bert-base | Darwiniana | 0.57 (29/51) | **0.77** (17/22) |

Essentialism is the more disadvantaged camp in all 4 configs — opposite of
what the DT-only view suggested, because NDT survives the gate well and
pulling it into "teleology" drags that pooled rate down. Golden-set numbers
on this grouping are too small (n=6 for essentialism) to trust directionally
and shouldn't be used for this comparison.

Separately, the reverse error — true junk leaking through as a real
category — is small (8-9% on held-out text) and roughly evenly spread
across DT/NDT/IE (see cascade end-to-end confusion matrices). Leakage is not
the risk; wrongful discarding of essentialism specifically is.

### Output proportions: true vs. cascade end-to-end vs. hypothetical-perfect-gate

Computed on each held-out text's own rows (not the pooled golden set, which
isn't representative of any single text's true distribution). "True" here
means `deploy_tag` (the LLM's own labels for that text — the golden set's
human labels don't cover enough per-text rows to use directly).

| Text | | Divine teleology | Non-divine teleology | Internal essentialism |
|---|---|---|---|---|
| Reign of Law (n=169) | True | 17.2% | 66.9% | 16.0% |
| | Cascade end-to-end | 19.8% | 63.2% | 17.0% |
| | Perfect-gate hypothetical (stage 2 alone) | 21.9% | 62.1% | 16.0% |
| Darwiniana (n=73) | True | 15.1% | 54.8% | 30.1% |
| | Cascade end-to-end | 20.7% | 63.4% | 15.9% |
| | Perfect-gate hypothetical (stage 2 alone) | 16.4% | 46.6% | 37.0% |

Reign of Law: cascade end-to-end is within a few points of true on every
category, and a perfect gate wouldn't change much. Darwiniana: the cascade
meaningfully overstates DT and understates IE (30.1% true → 15.9%); a
perfect gate would *overstate* IE instead (→37.0%) — the sign flips. That
flip is strong evidence the 3-way categorizer itself is trustworthy and the
distortion is coming from the gate, but also that the gate's error isn't a
fixed, correctable bias — it varies by text in a way not yet predictable.

## Decision (updated)

Not deployable for proportion-tracking yet, on either the single 4-way model
or the cascade. Next steps, decided with the user:

1. **Commit to fine-tuning a small LLM (via LoRA) specifically as the
   junk/non-junk gate**, rather than further BERT-side tuning
   (oversampling/threshold, above) — a bigger swing, justified because the
   gate is the clearly-isolated bottleneck rather than a minor knob, and
   because plain bert-base already matched or beat MacBERTh here, suggesting
   more BERT tuning has limited headroom left.
2. **Build a margin-of-error framework for reported proportions, not a
   correction.** Explicitly decided *against* trying to mathematically
   correct output proportions using known error rates (too risky given the
   gate's error isn't a stable, correctable bias per the Darwiniana sign-flip
   above) — instead, attach an honest uncertainty range to any proportion
   estimate, derived from the model's track record on texts with known
   answers.

PI update summarizing all of the above sent — draft in `bert_progress_email.md`
(repo root, untracked).
