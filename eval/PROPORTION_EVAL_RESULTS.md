# Proportion Evaluation — Results

*How accurately does the pipeline recover the DT / NDT / IE mix of a book it
has never seen, and how confident should we be in a reported proportion?*

Run 2026-08-23. Method and locked decisions: `eval/PROPORTION_EVAL_PLAN.md`.

---

## Headline

Across **16 held-out books (12,823 sentences)**, evaluated by 6-fold
leave-works-out so every book is scored by a model that never saw it:

| Category | Bias | Spread (sd) | Mean abs. error | Worst book | **±90% for a new book** |
|---|---|---|---|---|---|
| Divine teleology | +0.3 pp | 5.2 | 2.6 pp | 17.9 pp | **±8.5 pp** |
| Non-divine teleology | −1.5 pp | 8.8 | 6.0 pp | 21.6 pp | **±14.4 pp** |
| Internal essence | +1.2 pp | 7.4 | 5.6 pp | 17.9 pp | **±12.3 pp** |
| *Teleology (DT+NDT) vs essence* | −1.2 pp | 7.4 | 5.6 pp | 17.9 pp | **±12.3 pp** |

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
  (90%, 95% CI 87–93%).

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
plus a calibrated decision threshold — both deliberately skipped here, since
running 5 seeds × 6 folds meant 30 trainings. **Restoring the ensemble and
threshold calibration is the clearest single improvement available**, and it
requires no new training method, only more of the same runs.

Darwiniana is also simply hard: the gate keeps only 18% of its true DT and IE
rows, so its estimate rests on very few survivors.

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
