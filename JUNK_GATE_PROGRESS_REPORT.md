# Junk Gate — Progress Report

*A status update on the first stage of the classification pipeline (the
"junk gate"), written for a non-technical audience.*

## The short version

We built a replacement for the weakest link in the labeling pipeline. The
final version is a clear, all-around improvement over what it replaces:
it keeps substantially more of the sentences that matter in every
category, it's *more* consistent across categories than before (not
less), and it leaks less junk into the final output, not more. We also
checked two bigger, more expensive alternatives; neither beats what we
already have, so we're not pursuing them further right now.

## What this stage does

Before a passage can be categorized as divine teleology (DT), non-divine
teleology (NDT), or internal essence (IE), it first has to pass a filter
that throws out "junk" — text that isn't making any relevant claim at
all. This filter is the **junk gate**. If it's too aggressive, it throws
away good sentences before they ever get a chance to be categorized. If
it's too lenient, junk leaks through and gets mislabeled downstream. Both
failures directly corrupt the final output.

The previous version of this gate (built on a model called BERT) was the
project's known bottleneck. This report covers our effort to replace it
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

## Where we landed (the new gate — current state of the art)

| Metric | Old gate (BERT) | **New gate (final)** | Change |
|---|---|---|---|
| Recall — Divine Teleology | 48% | **55%** | +7 pts |
| Recall — Non-Divine Teleology | 50% | **69%** | +19 pts |
| Recall — Internal Essence | 33% | **59%** | **+26 pts** |
| Evenness (spread across categories) | 16 points | **14 points** | improved |
| Precision (leakage) | 75% | **81%** | +6 pts |

This is a clean win across the board: every category keeps more of its
sentences than before, the spread across categories is *tighter* than
before (not wider), and less junk leaks through into the final labeled
output than before. There's no axis on which the new gate is worse.

The path here wasn't a straight line — an earlier version of the new gate
traded away some of this evenness and precision gain for higher raw
recall, and for a while that was our leading candidate. We deliberately
gave some of that recall back to land on a version that's balanced across
all three metrics simultaneously, since a lopsided gate distorts the
final output even when its raw numbers look strong.

## Core innovations

A few ideas made the biggest difference, in the order we found them:

1. **Fine-tuning a small AI language model, instead of patching the old
   one.** The old gate's architecture had a real ceiling; a different
   kind of model gave us a much larger and more controllable range of
   outcomes to work with.
2. **Catching a hidden measurement bug.** Partway through, we discovered
   we'd been tracking a subtly wrong version of the precision metric —
   one that measures a different, less important failure (being overly
   cautious) rather than the one that actually matters (junk leaking
   through). Every number in this report uses the corrected metric; some
   earlier "wins" turned out to be smaller, or not real, once measured
   correctly.
3. **Simple, explicit instructions beat worked examples.** We tried
   giving the model a clear rule to follow versus showing it example
   sentences to imitate. The clear rule consistently won — a durable
   finding that held up across many different tests.
4. **Averaging multiple independently-trained copies of the model**
   ("ensembling"). Training the same setup twice can give meaningfully
   different results just from randomness in the training process.
   Combining several independently-trained copies cancels out a lot of
   that randomness, for free, with no extra training cost.
5. **Directly tuning the gate's confidence bar.** Rather than accepting
   the model's default sense of "confident enough to pass," we tested
   many different confidence thresholds and picked the one that gives the
   best simultaneous balance of evenness and leakage — the final lever
   that got us to a fully balanced result.

## What we tried

- **Tuning the model's internal settings** (how much capacity it has to
  learn, how aggressively it trains, etc.) — giving the model more
  internal capacity was a real, meaningful lever, though it came with
  trade-offs of its own (see below).
- **Rewriting the instructions given to the model** (the "prompt") —
  several phrasings were tried; the winning one uses a short, clear rule
  plus one worked example, and is part of the final configuration.
- **Combining our best individual improvements** — we expected our best
  settings changes and our best instruction rewrite to stack additively.
  They didn't: combining them tended to *overcorrect*, pushing recall up
  unevenly across categories rather than lifting everything together.
- **Averaging multiple independently-trained copies of the final model**
  and **directly tuning its confidence threshold** — both described above
  as core innovations, and both part of the final, adopted configuration.

## What we abandoned, and why

- **The higher-recall configuration we initially adopted as our leading
  candidate.** It looked like the clear winner at first, but it achieved
  its recall by leaning unevenly on the categories that were already
  easiest — a pattern we saw repeatedly whenever we pushed the model to
  be more aggressive. We traded some of that recall back for a version
  that's balanced across all three metrics, rather than keeping the
  highest raw number.
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
- **Giving extra weight to underrepresented sentences in general**
  (not category-specific) — looked promising in an initial check, but
  didn't hold up once tested more thoroughly.
- **Training for longer** — tested once and it didn't help enough to
  justify pursuing further.

## Alternatives considered: are bigger/different approaches worth it?

We ran two quick, one-off checks (not fully tuned — just enough to see if
there's obvious headroom worth chasing):

| Approach | Recall (avg. across categories) | Precision |
|---|---|---|
| **Current final gate** | 61% | **81%** |
| Training the *entire* model instead of a small piece of it | 65% | 65% |
| A larger version of the same model | 64% | 76% |

**Neither alternative clearly beats what we already have.** Training the
entire model gets a bit more recall but leaks noticeably more junk
through. The larger model lands in between on both counts. Given this,
we're not recommending either path right now — though both remain
reasonable options to revisit later if more headroom is needed than
further tuning of the current approach can provide.

## Recommendation

Adopt the new gate as the production replacement for the old one. It's a
clean, all-around improvement — better recall in every category, a
tighter spread across categories, and less junk leaking through — with no
axis left worse off than before. We got here by deliberately choosing
balance over the highest possible raw recall number, which we believe is
the right call given how directly evenness and leakage affect the
trustworthiness of the final labeled output.
