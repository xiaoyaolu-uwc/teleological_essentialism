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
