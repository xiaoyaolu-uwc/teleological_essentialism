# Junk Gate — Progress Report

*A status update on the first stage of the classification pipeline (the
"junk gate"), written for a non-technical audience.*

## The short version

We built a replacement for the weakest link in the labeling pipeline and
it's a clear improvement — it keeps far more of the sentences that
matter, especially the hardest category, without letting more junk slip
into the final output than before. It hasn't hit every target we set for
ourselves, and we checked two bigger/more expensive alternatives; neither
clearly beats what we already have, so we're not pursuing them further
for now.

## What this stage does

Before a passage can be categorized as divine teleology (DT), non-divine
teleology (NDT), or internal essence (IE), it first has to pass a filter
that throws out "junk" — text that isn't making any relevant claim at
all. This filter is the **junk gate**. If it's too aggressive, it throws
away good sentences before they ever get a chance to be categorized. If
it's too lenient, junk leaks through and gets mislabeled downstream. Both
failures directly corrupt the final output.

The previous version of this gate (built on a model called BERT) was the
project's known bottleneck. This report covers our attempt to replace it
with a fine-tuned small language model, and where that effort landed.

## How we measure success

Plain accuracy isn't the right yardstick here, because it hides exactly
the failure that matters most. We track three things instead:

1. **Recall, per category (DT / NDT / IE)** — of all the sentences that
   truly belong to a category, what fraction actually make it through the
   gate? If this is low for one category, that category is
   under-represented in everything downstream, even if it did nothing
   wrong.
2. **Evenness** — how much recall varies across the three categories. A
   gate that lets through 80% of one category but only 30% of another
   distorts the final mix even if its *average* performance looks fine.
3. **Precision (leakage)** — of everything the gate passes through as
   "good," what fraction actually is? This is the "don't get it wrong"
   number — every junk sentence that leaks through becomes a real error
   in the final labeled output, which is the mistake we care about most.

Recall and precision are treated as **equally important**. A gate that
keeps everything (high recall, low precision) is just as unacceptable as
one that's overly cautious (high precision, low recall) — both distort
the final output, just in different directions.

## Where we started (the old gate)

| Metric | Old gate (BERT) |
|---|---|
| Recall — Divine Teleology | 48% |
| Recall — Non-Divine Teleology | 50% |
| Recall — Internal Essence | 33% |
| Evenness (spread across categories) | 16 points |
| Precision (leakage) | 75% |

The old gate's biggest problem: Internal Essence, already the rarest
category, survived the gate only a third of the time — meaning it was
being systematically under-represented in the final output.

## Where we landed (the new gate)

| Metric | Old gate (BERT) | **New gate (final)** | Change |
|---|---|---|---|
| Recall — Divine Teleology | 48% | **59%** | +11 pts |
| Recall — Non-Divine Teleology | 50% | **80%** | +30 pts |
| Recall — Internal Essence | 33% | **59%** | **+26 pts** |
| Evenness (spread across categories) | 16 points | 21 points | slightly worse |
| Precision (leakage) | 75% | **76%** | +1 pt |

The new gate keeps substantially more of every category — especially
Internal Essence, the one that mattered most — **without leaking more
junk through than before.** Precision improved slightly even while
recall rose sharply across the board.

The one honest gap: **evenness is slightly worse in raw terms**, not
better. This needs context, though — the old gate's narrow spread came
from uniformly *low* recall (nothing survived well, so nothing could look
uneven), not from fairness. In practical terms, the new gate is
unambiguously better for every category's actual representation in the
final output; we're flagging the evenness number honestly rather than
declaring full success, since our original target was a much tighter
spread (a 10-point gap) than we ended up with.

**Bottom line**: this is a real, adopted improvement over the old gate —
not a marginal one — but it falls short of the original stretch targets
we set at the outset, particularly on evenness.

## What we tried

- **Tuning the model's internal settings** (how much capacity it has to
  learn, how aggressively it trains, etc.) — one specific adjustment
  (giving the model more internal capacity) was the single biggest lever
  we found and is part of the final configuration.
- **Rewriting the instructions given to the model** (the "prompt") —
  several different phrasings were tried. The clearest finding: giving
  the model an explicit, structured rule to follow worked consistently
  better than giving it worked examples to imitate. One particular
  rewrite, paired with the capacity change above, is the other half of
  the final configuration.
- **Combining our best individual improvements** — we expected our best
  settings changes and our best instruction rewrite to stack additively.
  They mostly didn't; combining them landed close to what either
  achieved alone, not meaningfully better than both together.
- **Averaging multiple independently-trained copies of the final model**
  ("ensembling") — this genuinely helped, for free, with no additional
  training. It's part of the final, adopted configuration.
- **Adjusting the pass/fail sensitivity of the gate directly** — we
  checked whether shifting the gate's decision boundary could buy
  recall without costing precision (or vice versa). It couldn't: every
  gain on one side cost the other, confirming our suspicion that this
  wasn't a free lever. We kept the default setting.

## What we abandoned, and why

- **Extra-long input formatting** — giving the model more room to
  "read" each passage, to guard against the instructions plus passage
  text running too long. It didn't help — results got *worse*, not
  better, and we don't have a fully satisfying explanation why. Since
  the room was never actually the constraint for most cases, we dropped
  it.
- **Weighting the rarest category more heavily during training** — a
  natural idea, given Internal Essence's known weakness. Tested and
  discarded: it didn't move the needle and wasn't worth the added
  complexity.
- **Combining that reweighting with other capacity changes** — same
  story, discarded for the same reason.
- **Training for longer** — tested once and it didn't help enough to
  justify pursuing further.
- A **measurement bug** we caught partway through: an earlier version of
  this report would have used a subtly wrong precision metric — one that
  looks similar but actually measures a different, less important
  failure mode (over-discarding good sentences, rather than junk leaking
  through). All numbers in this report use the corrected metric.

## Alternatives considered: are bigger/different approaches worth it?

We ran two quick, one-off checks (not fully tuned — just enough to see if
there's obvious headroom worth chasing):

| Approach | Recall (avg. across categories) | Precision |
|---|---|---|
| **Current final gate** | 66% | **76%** |
| Training the *entire* model instead of a small piece of it | 65% | 65% |
| A larger version of the same model | 64% | **76%** |

**Neither alternative clearly beats what we already have.** Training the
entire model actually hurt precision noticeably. The larger model matched
our precision but didn't improve recall enough to justify its extra cost.
Given this, we're not recommending either path right now — though the
larger-model result is close enough that it could be worth a fuller
attempt later if more headroom is needed.

## Recommendation

Adopt the new gate as the production replacement for the old one — it's
a clear, broad-based improvement, especially for the previously
under-served Internal Essence category, with no precision cost. The
evenness target is the one place we're not where we originally wanted to
be, and closing that gap further would need either a new idea we haven't
found yet, or a larger investment (bigger model, full retraining) that
our quick checks don't currently justify.
