# Proportion Evaluation — Results

*How accurately does the pipeline recover the DT / NDT / IE mix of a book it
has never seen, and how confident should we be in a reported proportion?*

Run 2026-08-23. Method and locked decisions: `eval/PROPORTION_EVAL_PLAN.md`.

---

## Headline

Across **16 held-out books (12,823 sentences)**, evaluated by 6-fold
leave-works-out so every book is scored by a model that never saw it:

| Category | Bias | Spread (sd) | Mean abs. error | Worst book | ±90% (parametric) | Empirical p90 of \|err\| |
|---|---|---|---|---|---|---|
| Divine teleology | +0.3 pp | 5.2 | 2.6 pp | 17.9 pp | ±8.5 pp | 5.6 pp |
| Non-divine teleology | −1.5 pp | 8.8 | 6.0 pp | 21.6 pp | ±14.4 pp | **15.7 pp** |
| Internal essence | +1.2 pp | 7.4 | 5.6 pp | 17.9 pp | ±12.3 pp | **15.1 pp** |

The ±90% column is **parametric** (1.645 × sd, assuming normality). With n=16
that assumption is not guaranteed, so the empirical 90th percentile of absolute
error is shown beside it. For NDT and IE the empirical tail is *fatter* than the
parametric band (15.7 vs 14.4; 15.1 vs 12.3), so **quote the empirical figure
when being careful**. Measured coverage of the parametric band is 94% / 88% /
88% against a nominal 90%.

*Note:* "teleology (DT+NDT) vs essentialism" is **not** an independent result —
among non-junk rows DT+NDT = 1 − IE exactly, so its error is algebraically the
mirror of IE's (bias −1.2 pp, sd 7.4). An earlier design note predicted the
binary would be "visibly tighter"; it cannot be.

**There is essentially no systematic bias** — every category's mean signed
error is within 1.5 pp of zero. The error is noise around the right answer,
not a tilt. That is the single most important result here: it means the model
is not systematically inflating or deflating any category, so a trend line
built from it is not being bent in a fixed direction.

**Ranking is much more reliable than magnitude.**

- The full DT/NDT/IE ordering within a book is recovered in **14 of 16 books**
  (88%, 95% CI 64–97%). Pairwise, **47 of 48** comparisons are correct (98%).
- Comparing the *same* category across two books — the operation every
  research question actually performs — is correct in **320 of 354** cases
  (90%, **95% CI 83.5–95.8%**).

  That interval is a **cluster bootstrap over books**, not a Wilson interval.
  The 354 comparisons come from only 16 books, each appearing in 15 pairs, so
  they are strongly dependent; a Wilson interval treats them as independent and
  returns 87–93%, which is too narrow. An earlier version of this report quoted
  the Wilson figure. Per-category rows below still show Wilson intervals and
  are subject to the same caveat.

## When can two texts be ordered? (the resolving-power result)

This is the number that answers "is it 60-30-10 ±5 or ±30". Pooled across all
three categories and all 120 book pairs:

| True gap between the two texts | Ordering correct | 95% CI |
|---|---|---|
| 0–2 pp | 54% (14/26) | 35–71% |
| 2–5 pp | 74% (14/19) | 51–88% |
| 5–10 pp | 84% (43/51) | 72–92% |
| 10–20 pp | 90% (73/81) | 82–95% |
| **20 pp +** | **99% (176/177)** | 97–100% |

Read plainly: **differences under ~5 pp are not resolvable** — at 0–2 pp the
model is barely better than a coin flip, exactly as it should be. From about
10 pp the ordering is reliable, and above 20 pp it is essentially never wrong.

For the blog post this converts into one honest sentence: *a difference of
20 points or more between two periods is solid; 10–20 points is likely but
not certain; anything under 5 points should not be interpreted.*

**This is a conservative bound for the decade chart.** These figures compare
two individual books. A decade bucket pools ~10 books under balanced quota
sampling, so independent per-book errors partly cancel and the real
decade-to-decade comparison will be better than this table, not worse. We
have deliberately not estimated how much better — that would require the
resampling machinery we scrapped.

## Where the remaining error comes from

Mean total-variation distance across books, by stage attribution:

| Configuration | Mean TVD |
|---|---|
| Perfect gate, real stage 2 | 5.3 pp |
| Real gate, perfect stage 2 | 4.9 pp |
| **End-to-end (both real)** | **7.1 pp** |

The two stages contribute almost equally, and they partly cancel (7.1 < 5.3 +
4.9). Neither is "the bottleneck" any more — a change from the situation in
`eval/bert_cascade_evolution.md`, where stage 1 dominated the error.

**But the three worst books share one cause: junk leakage.**

| Book | Max error | Junk as share of kept set |
|---|---|---|
| On the Origin of Species | 21.6 pp | 52% |
| Wisdom of God (Ray) | 17.9 pp | 33% |
| Darwiniana | 15.2 pp | 58% |

Overall junk leakage into the kept set is **31%**, i.e. gate precision ≈ 69%.
The adopted production gate reaches **81%** precision using a 5-seed ensemble
plus a calibrated decision threshold — both skipped here, since 5 seeds × 6
folds means 30 trainings.

**Threshold calibration was then tested, and does not help** (see
`eval/calibrate_gate_threshold.py`, run under nested selection so the threshold
for each fold is chosen on the other five and never on the books it is scored
on). The nested threshold lands on 0.54 for every fold and buys **+0.15 pp of
TVD** — negligible, improving 10 of 16 books. The pooled sweep is flat near 0.5
and worse in both directions:

| threshold | 0.30 | 0.40 | **0.50** | 0.60 | 0.70 | 0.80 |
|---|---|---|---|---|---|---|
| mean TVD | .081 | .076 | **.071** | .075 | .077 | .088 |

The gate is therefore **not miscalibrated — it is simply not sharp enough**.
Moving the operating point trades recall for precision at roughly 1:1 in TVD
terms, so no threshold recovers the leakage. That rules out the cheap half of
the production gate's recipe and leaves **ensembling** as the remaining lever,
since averaging several seeds' probabilities reduces variance in a way
re-thresholding a single seed cannot.

Darwiniana is also simply hard: the gate keeps only 18% of its true DT and IE
rows, so its estimate rests on very few survivors.

## Does ensembling the gate help? (2 seeds, tested)

Threshold calibration having failed, the remaining lever was ensembling. A
**2-seed** gate ensemble (6 extra trainings, not the 30 a 5-seed design would
need) was trained and evaluated identically — same books, same stage 2, same
metric, so the gate is the only variable.

| Metric | 1 seed | 2-seed ensemble | Change |
|---|---|---|---|
| Junk leakage into kept set | 31.1% | **28.3%** | −2.8 pp |
| Gate-only distortion (TVD) | 4.86 pp | **3.92 pp** | −0.94 pp |
| Worst-book TVD | 21.6 pp | **18.2 pp** | −3.4 pp |
| NDT ±90% band | 14.4 pp | **12.3 pp** | −2.1 pp |
| IE ±90% band | 12.3 pp | **11.7 pp** | −0.6 pp |
| DT ±90% band | 8.6 pp | 9.0 pp | +0.5 pp |
| **Mean TVD, end-to-end** | 7.14 pp | 7.24 pp | +0.10 pp |
| Within-book ordering | 14/16 | 14/16 | — |
| Across-book ordering | 90.4% | 90.7% | — |

**The ensemble clearly improves the gate and barely moves the end-to-end
result.** Leakage falls, gate-only distortion drops by a full point, the worst
book improves by 3.4 pp, and the widest error band (NDT) tightens by 2 pp — but
mean TVD is flat, because stage-2 error is unchanged at 5.3 pp and now
dominates the 3.9 pp gate term.

**Recommendation: adopt the 2-seed ensemble** — it is 6 extra trainings for a
tighter worst case and a meaningfully narrower NDT band, with no axis
materially worse (DT widens by 0.5 pp, within noise at n=16). But **do not
expect much from going to 5 seeds**: the gate is no longer the larger error
term, so the headroom that remains is in stage 2, not in more gate seeds.

Artifacts: `per_row_predictions_ens2.csv`, `ens2/proportion_metrics.json`.

## What changed in the model

Stage 2 was replaced: MacBERTh → Qwen3-0.6B LoRA (r32/α64, `S2_structured`
prompt, 4 epochs). On the two dev folds it beats MacBERTh on both, roughly
halving mix error (TVD 0.036 vs 0.047 on Reign of Law; 0.041 vs 0.082 on
Darwiniana) and adding up to 8.6 pp accuracy.

Two findings worth carrying forward:

1. **The 4-way stage-2 idea does not work.** Giving stage 2 its own junk class
   (so it can reject what the gate leaked) loses on every axis, with macro-F1
   collapsing to 0.59–0.61 vs 0.80. It has to relearn junk rejection from the
   full 65%-junk corpus, reintroducing exactly the gradient dilution the
   cascade exists to remove. Independent confirmation of the original cascade
   hypothesis.
2. **A quarter of raw-text rows were being silently truncated.** At
   `max_length=384`, 27.8% of raw `text` rows overflow once the prompt is
   prepended (vs 2.9% of `deploy_extract` rows). Raising to 640 recovered most
   of the loss — fold mix error 0.075 → 0.047. The junk gate's earlier finding
   that longer inputs *hurt* was measured on `deploy_extract`, where truncation
   never bound, and does not transfer.

## Caveats

- **"True" here means the GPT-5.4 `d_v3` labels**, which agree with the 49-row
  human golden set at ~91%. Everything above measures agreement with that
  labeller, not with ground truth. The ~9% disagreement is not propagated into
  any interval quoted here.
- **All 16 books are hand-picked polemical anchor texts.** The BHL scanning
  corpus is ordinary natural history with a higher junk rate and weaker signal;
  expect these numbers to be optimistic there.
- **Everything is measured on raw `text` at `max_length=640`** — the deployable
  setting, since `deploy_extract` does not exist for a new book.
- **n = 16.** Ranking rates are reported with Wilson intervals and no claim of
  "95% confidence above gap X" is made; several of those intervals are wide.
- The single-seed gate used here underperforms the production gate (69% vs 81%
  precision), so these are a **floor**, not a ceiling.

## Reproducing

```bash
python3 eval/run_cascade_folds.py --text-column text --max-length 640
python3 eval/evaluate_proportions.py
python3 eval/plot_proportions.py
```

Artifacts: `eval/results/proportions/per_row_predictions.csv`,
`proportion_metrics.json`, `proportion_figures.png`.
