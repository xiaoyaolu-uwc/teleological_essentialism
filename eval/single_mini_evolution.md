# Single-Layer Prompt Evolution — gpt-5.4-mini

Tracks each generation of the single-layer prompt optimization loop on gpt-5.4-mini.
Each generation: identify target passages → write 3 variants → run 5 evals each → check targets → promote winner.

**Rule:** Every prompt change must be justified by an argument about what the category
definition requires. Flag any passage-specific changes explicitly before making them.

---

## Eval set composition (as of June 2026)
49 passages. mixed_dt-ie and mixed_ndt-ie collapsed into divine_teleology and
non_divine_teleology respectively.
- junk: 19
- non_divine_teleology: 14
- divine_teleology: 10
- internal_essence: 6

---

## Historical record (carried over from prompt_evolution.md)

### v1 — Original scan prompt (12 runs, 69.6% avg)

Module-level SYSTEM_PROMPT / USER_PROMPT_TEMPLATE in models/prompts.py.
Used by scan_passages.py to label the full 6000-row corpus.

**Top failure modes (12-run aggregate):**
- junk → divine_teleology: Ufefulnefs (12/12)
- junk → internal_essence: parallelism/gradation (12/12)
- mixed → internal_essence: microscopic identity, man last/highest (12/12)
- junk → non_divine_teleology: NS never produce, eye and hand
- NDT → junk: NS specialise organ, two organs modified (10/12 each)

---

### v2 — Two-step ontological gate (5 runs, 64.9% avg)

Junk promoted to a gating criterion. divine_teleology tightened (both conditions required).
Mixed tags added (subsequently collapsed — see eval set note above).

---

### Iteration 1: v3a / v3b / v3c

| Variant | Avg | DT | IE | NDT | Junk prec |
|---|---|---|---|---|---|
| v2 | 64.9% | 67% | 67% | 65% | 71% |
| v3a (implicit DT) | 68.0% | 70% | 50% | 69% | 79% |
| v3b (organ-function discriminator) | 68.0% | 81% | 56% | 62% | 77% |
| v3c (surface trigger suppression) | **70.1%** | 78% | 72% | 64% | 77% |

Winner: v3c. v3a dropped (hurts IE). v3b's discriminator preserved for v4 stacking.

---

### Iteration 2: v4 — v3c + v3b NDT discriminator (71.4% avg)

Stacked cleanly. New regressions: two Agassiz DT passages now consistently junk.

---

### Iteration 3: v5a / v5b / v5c — all regressed from v4

**Key learning:** Prompt too long/complex for gpt-5.4-mini. More text → cross-tag interference.

---

### Iteration 4: v6a / v6b / v6c

| Variant | Avg | Notes |
|---|---|---|
| v6a (radical simplification) | 58.5% | Proved v2→v4 structure is load-bearing |
| v6b (short arg-from-design bullet) | 68.7% | IE jumped to 89% but junk→DT FPs spiked |
| v6c (positive-first junk gate) | 63.3% | DT recall collapsed to 41% |

**Key learning:** arg-from-design fix needs explicit "→ junk, NOT divine_teleology" redirect.

---

### Iteration 5: v7a / v7b / v7c / v7d

| Version | Avg | DT | IE | NDT | Junk |
|---|---|---|---|---|---|
| v7a (CoT format) | 58.2% | 14% | 58% | 63% | 82% |
| v7b (arg-from-design redirect) | 72.1% | 70% | 83% | 67% | 81% |
| **v7c (3 abstract junk patterns)** | **74.8%** | 74% | 56% | 72% | 91% |
| v7d (v2 + only useful changes) | 70.7% | 89% | 61% | 64% | 77% |

**Winner: v7c. Current best.**
v7a definitively dead — CoT collapses DT to 14%. Do not revisit.

**Per-target at v7c:**
| Passage | v4 | v7c |
|---|---|---|
| Ufefulnefs (junk) | 3/3 ✗ | 1/3 ✗ (partial) |
| NS never produce (junk) | 3/3 ✗ | 1/3 ✗ (partial) |
| eye and hand (junk) | 3/3 ✗ | **0/3 ✓ FIXED** |
| NS specialise organ (NDT) | 3/3 ✗ | 3/3 ✗ (stuck) |
| two organs modified (NDT) | 3/3 ✗ | 3/3 ✗ (stuck) |
| microscopic identity (IE) | variable | 3/3 ✗ (new regression) |

---

### Iteration 6: v8a / v8b / v8c — all regressed from v7c

**Key learnings:**
- NDT stuck pair: root cause is HYPOTHETICAL phrasing ("might"), not grammatical subject.
- Any IE-specific addition causes DT→IE interference (model hunts for structural component in DT passages).
- IE fix confirmed mechanistically: v8a/v8c fixed microscopic identity but at heavy DT cost.

---

### Iteration 7: v9a / v9b / v9c — all regressed from v7c

**Key learnings:**
- v9b: first ever movement on NDT stuck pair (2/3) but broke NS-never-produce.
- v9c: DT recall jumped for Agassiz but opened 5-6 junk→DT FPs. Direction right, needs guard.
- **Stop pursuing IE fix** — any IE text triggers DT→IE interference. IE is a lower-priority problem.

---

### Iteration 8: v10a / v10b / v10c — all regressed from v7c

v10b best (73.5%) but NS-never-produce regressed to 3/3 NDT due to DT "divine thought" contamination.
Rule-based changes exhausted on this axis.

---

### Iteration 9: v11a / v11b / v11c — all regressed from v7c

**Critical find:** v11b's pattern (d) fixed parallelism/gradation AND mechanical weapons for the
first time ever — but broke DT ("exhibits thought" caught as junk).

---

### Iteration 10: v12a / v12b / v12c — all regressed from v7c. Loop closed.

v12c fixed NS-never-produce (0/3) but DT recall collapsed across the board.

**FINAL STATE: v7c (74.8%) is the best single-layer mini prompt.**

---

## Stuck passages summary (entering new loop)

| Passage | Correct | v7c wrong rate | Notes |
|---|---|---|---|
| NS specialise organ | NDT | 3/3 ✗ | Hypothetical framing ("might") — never fixed |
| two organs modified | NDT | 3/3 ✗ | Hypothetical framing ("might") — never fixed |
| Ufefulnefs | junk | 1/3 ✗ | Partially fixed in v7c; regresses in v8+ |
| NS never produce | junk | 1/3 ✗ | Partially fixed; NDT in v8+, v9b+, v10b, v12c |
| parallelism/gradation | junk | 3/3 ✗ | Fixed only by v11b (which broke DT) |
| microscopic identity | IE | 3/3 ✗ | Fixed by v8a/v9a but both broke DT badly |

**Hard constraints from iteration history:**
- Do NOT use CoT output format (v7a: DT collapses to 14%)
- Do NOT add IE-specific language (any IE text → DT→IE interference)
- Do NOT use broad NDT hypothetical language (breaks NS-never-produce)
- Do NOT use "embodies divine thought/intellect" phrasing for DT (v5b, v9c: breaks junk gate)
- Adding text generally hurts more than it helps at this length — prefer surgical changes

---

## Baseline: v7c on gpt-5.4-mini (5 runs, new eval set)

Run dir: `eval/results/single_54mini/v7c_5run/`
**Overall: 70.6% avg** (346/490 correct across 5 runs)

Per-class recall:
- divine_teleology: 54% (27/50)
- internal_essence: 73% (22/30)
- junk: 86% (82/95)
- non_divine_teleology: 60% (42/70)

### Stuck passages (5/5 wrong):

| Passage snippet | Correct | Predicted | Notes |
|---|---|---|---|
| "parallelism between gradation among animals...exhibits thought" | junk | internal_essence ×5 | "exhibits thought" triggers IE/DT |
| "perfect identity of the most delicate microscopic structures" | internal_essence | junk ×5 | Structural similarity claim gated as junk |
| "order and arrangement of our studies are...indisputable" | divine_teleology | junk ×5 | Agassiz: implicit DT via classification order |
| "nature and foundation of our scientific classifications" | divine_teleology | junk ×5 | Agassiz: implicit DT via classification purpose |
| "man is not only the last and highest...last term of one of these natural series" | divine_teleology | internal_essence ×5 | DT divine ordering misread as structural IE |
| "natural selection might easily specialise...one part...one function" | non_divine_teleology | junk ×5 | Hypothetical NDT construction |
| "one of the two organs might with ease be modified and perfected" | non_divine_teleology | junk ×5 | Hypothetical NDT construction |

### Stuck passages (4/5 wrong):

| Passage snippet | Correct | Predicted | Notes |
|---|---|---|---|
| "merely mechanical weapons" (variety and rank among animals) | non_divine_teleology | junk ×3, divine_teleology ×1 | NDT about weapons/rank |
| "Natural selection will never produce...anything injurious" | junk | non_divine_teleology ×4 | General NS principle misread as NDT |

---

## Generation 1 (new loop)

### Target passages

Primary targets (all 5/5 wrong):
1. **"order and arrangement of our studies"** (DT → junk): Agassiz arguing classification reflects divine plan
2. **"nature and foundation of our scientific classifications"** (DT → junk): Agassiz arguing taxonomy has divine foundation
3. **"natural selection might easily specialise...one part...one function"** (NDT → junk): hypothetical NDT
4. **"one of the two organs might with ease be modified and perfected"** (NDT → junk): hypothetical NDT

Secondary targets (3/5 wrong):
5. **"mutual dependence of animal/vegetable kingdoms...exhibits thought"** (DT → junk ×3): implicit ecological DT

Success criterion: targeted passages drop from 5/5 or 4/5 wrong → ≤2/5 wrong, without overall accuracy regressing below 70.6%.

---

### Root cause analysis

**DT recall failure (passages 1, 2, 5):** The two Agassiz DT passages argue that zoological
CLASSIFICATION ITSELF reflects divine thought — not that any specific organism was designed,
but that the ARRANGEMENT and ORDER of taxa expresses God's plan. The model's current DT
definition probably requires an explicit claim about an organism or its parts being designed.
These passages make the claim at the level of the whole system of nature / taxonomic order.
This IS DT: the argument is that divine purpose explains why classification takes the form it
does, not just that any individual creature was made for a purpose.

Principle basis for fix: DT should include "the ordering/arrangement/classification of animal
groups reflects divine thought or plan" — an implicit divine teleology operating at the
taxonomic level, not the organism level.

Guard needed: must not catch "parallelism between gradation...exhibits thought" (junk). That
passage observes a pattern and calls it evidence of thought, but makes NO claim about what
defines animals or their classification. The Agassiz passages explicitly make a claim about the
PURPOSE of classification. Key distinction: observing a thought-like pattern ≠ claiming divine
purpose governs the classification of organisms.

**NDT recall failure (passages 3, 4):** "Might easily specialise...one part to one function"
and "might with ease be modified" are hypothetical descriptions of what natural selection CAN
DO to specific organs. The model calls these junk because the "might" construction looks like a
general statement rather than an actual claim. But NDT is about positing functional purpose
for organic structures — and these passages DO posit that specific parts could serve specific
functions via NS. The hypothetical framing is about the mechanism (NS), not about whether the
function exists.

Principle basis for fix: NDT includes passages that describe specific organs or structures
acquiring or serving specific functions through natural selection, even when framed
hypothetically ("might," "could be modified"). The test is whether a specific
organ-function relationship is being asserted, not whether the language is indicative vs.
conditional.

Guard needed: must not catch "Natural selection will never produce anything injurious"
(junk, 4/5 already wrong → NDT). That passage states a CONSTRAINT on NS in general, with no
specific organ-function pair. The distinction: organ-function hypothesis (NDT) vs.
general principle about NS behavior (junk).

---

### Three variants

**v7c_mini_gen1a — DT fix only**

Targeted at: passages 1, 2, 5 (Agassiz DT → junk)

Change: In the DT category description, add one sentence after the existing definition:
> "This includes passages that argue the zoological classification, ordering, or systematic
> arrangement of animal groups reflects divine thought, plan, or intention — even when no
> specific organism is named and no explicit creator-language is used. The claim that
> taxonomy or the plan of nature itself has a divine purpose qualifies as divine_teleology."

Justification: Principle-driven — DT is defined by attributing purpose to the divine; it is
not limited to organism-level claims. Arguing that the ORDER of nature reflects God's thought
is exactly the core DT claim Agassiz makes throughout his work.

Violation flag: None. The addition does not reference passage-specific surface features.
It generalizes a principle that covers the Agassiz passages by virtue of their conceptual
content, not their wording.

**v7c_mini_gen1b — NDT hypothetical fix only**

Targeted at: passages 3, 4 (NDT hypothetical → junk)

Change: In the NDT category description, add one sentence:
> "NDT includes passages that describe how specific organs or structures might acquire or
> serve specific functions through natural processes (e.g., 'natural selection might
> specialise this organ for one purpose'). Hypothetical framing ('might,' 'could be
> modified') does not disqualify a passage from NDT — what matters is whether a
> specific organ-function relationship is being posited."

Guard sentence to add (prevents NS-never-produce from flipping to NDT):
> "General statements about what natural selection does or does not produce in principle,
> with no specific organ-function pair, remain junk."

Justification: Principle-driven — NDT is defined by attributing non-divine functional
purpose to organic structures. A hypothetical claim about what a part COULD do for an
organism is still making a teleological claim about that part. The conditionality is about
mechanism, not about whether teleology is asserted.

Violation flag: None. This is a logical extension of NDT's definition, not tailored to
passage surface features.

**v7c_mini_gen1c — Both fixes combined**

Targeted at: passages 1, 2, 3, 4, 5

Applies both changes from gen1a and gen1b simultaneously. Since the two fixes address
independent failure modes (DT taxonomy vs. NDT hypothetical) and neither touches the same
part of the prompt, the combined version should be additive with minimal interference.

Risk: slightly longer prompt may increase junk FP rate (model becomes more permissive on
multiple axes at once). Monitor overall junk recall (currently 86%) for regression.
