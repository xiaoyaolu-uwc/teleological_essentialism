# Prompt Evolution Log

Tracks each iteration of the optimization loop: what was tried, why, and what was learned.

---

## Baseline: v2 (5 runs)

**Accuracy:** 64.9% avg

**Key changes from v1:** Two-step ontological gate; junk as first-pass filter; tightened
divine_teleology (both conditions required); implicit functional definitions allowed;
mixed tags renamed to mixed_ndt-ie / mixed_dt-ie.

**Persistent failure clusters (from v2 5-run analysis):**

| Failure | Count | Correct | Predicted |
|---|---|---|---|
| divine_teleology → junk | 14/45 (31%) | divine_teleology | junk |
| non_divine_teleology → junk | 18/65 (28%) | non_divine_teleology | junk |
| junk → divine_teleology | 8/95 (8%) | junk | divine_teleology |
| junk → non_divine_teleology | 10/95 (11%) | junk | non_divine_teleology |
| junk → internal_essence | 9/95 (9%) | junk | internal_essence |
| internal_essence → non_divine_teleology | 5/30 (17%) | internal_essence | non_divine_teleology |

**Model reasoning diagnosis:**
- divine_teleology misses: model sees "mental purpose of producing beauty and variety" but says
  "not a definition of animals or parts" — implicit/aesthetic divine purpose not recognised
- non_divine_teleology misses: model says "discusses how organs may change function but does not
  define what the animal fundamentally is" — implicit functional characterization dismissed
- "essential features" surface trigger: junk passage about gradation always → internal_essence
- "adapted to conditions" surface trigger: internal_essence passage (Among Polypi) → non_divine_teleology

---

## Iteration 1: v3a / v3b / v3c

**Hypothesis:** Each variant targets one failure cluster independently.

### v3a — Implicit divine purpose
**Change:** Extend divine_teleology to capture indirect/aesthetic divine purpose.
The passage need not use the word "purpose" — it counts if it attributes species/animals
to a divine plan, aesthetic, or end (beauty, harmony, variety as divine goals).
Explicitly note that condition (1) can be satisfied implicitly.

**Targets:** divine_teleology→junk misses (14/45). Should not affect junk FP rate.

### v3b — Organ-function vs. mechanism-function discriminator
**Change:** Add explicit discriminator in non_divine_teleology: the functional claim must
be about what the animal/organ IS or DOES, not about what a mechanism (natural selection,
use-inheritance, growth) does TO animals. "Organs specialised for catching insects" =
characterizes the organ by its function (non_divine_teleology). "Natural selection acts
for the good of each" = characterizes the mechanism, not the animal (junk).

**Targets:** non_divine_teleology→junk misses (18/65) while holding junk precision.

### v3c — Surface trigger suppression
**Change:** Add explicit warnings that "essential features," "adapted to conditions,"
and similar phrases are not sufficient to pass Step 1 or assign a tag. The passage must
use those features as the basis for defining/categorizing animals, not merely observe them.
Also tighten internal_essence: structural characterization by internal plan ≠ "adapted
to conditions" language (which can be junk if no definitional claim is made).

**Targets:** junk→internal_essence FP (9/95); internal_essence→non_divine_teleology (5/30).

---

## Iteration 1 Results

| Variant | Avg Accuracy | DT recall | IE recall | NDT recall | Junk prec |
|---|---|---|---|---|---|
| v2 (baseline) | 64.9% | 67% | 67% | 65% | 71% |
| v3a | 68.0% | 70% | 50% | 69% | 79% |
| v3b | 68.0% | **81%** | 56% | 62% | 77% |
| v3c | **70.1%** | 78% | **72%** | 64% | 77% |

**v3a:** Implicit divine purpose helped DT recall slightly (+3%) but hurt IE recall (50%).
Not worth the regression.

**v3b:** Organ-function discriminator dramatically improved DT recall (67%→81%) — the best
single-change improvement. But hurt IE recall (67%→56%). Still 13 NDT→junk misses.

**v3c:** Surface trigger suppression best overall. IE recall improved (67%→72%), junk→IE FP
dropped from 9 to 2. DT recall improved (67%→78%). NDT recall unchanged (64%).

**Synthesis for v4:** v3c as base + v3b's organ-function discriminator paragraph added to
non_divine_teleology. These target different tags and should stack without conflicting.
v3a dropped — its DT fix is weaker than v3b's and it hurts IE.

**Remaining stubborn failures (all 3/3 in v3c):**
- "we can prove a priori...Ufefulnefs" → divine_teleology (should be junk) — argument that
  usefulness follows necessarily from existence, not that God made animals FOR something
- "The parallelism...gradation" → internal_essence (should be junk) — "essential features"
  still triggers despite surface trigger warning
- "The eye and the hand..." → non_divine_teleology (should be junk) — argument-from-design
  pattern still not blocked
- "Natural selection will never produce..." → non_divine_teleology (should be junk)
- Organ-specialization NDT passages → junk (13/39 NDT still missed)

---

## Iteration 2: v4

**Strategy:** Combine v3c + v3b's NDT change. No new changes — test whether the two fixes
stack cleanly before adding more complexity.

**Result: v4 = 71.4% avg** (up from v3c's 70.1%). DT recall 70%, IE recall 78%, NDT recall 67%, junk recall 81%.
Combination stacked cleanly. New failures emerged: two Agassiz DT passages now consistently junk (3/3).

---

## Iteration 2: v5a / v5b / v5c

**Hypothesis:** Three targeted fixes for the remaining v4 failures.

- v5a: Strengthen argument-from-design in junk gate (targets "eye and hand" junk→NDT)
- v5b: Extend DT to "embodies divine thought" pattern (targets 2 Agassiz DT→junk)
- v5c: Soften NDT discriminator for organ-nature characterizations (targets NDT→junk)

**Results:**

| Variant | Avg | DT recall | IE recall | NDT recall |
|---|---|---|---|---|
| v4 (baseline) | 71.4% | 70% | 78% | 67% |
| v5a | 66.0% | 74% | 50% | 64% |
| v5b | 63.3% | 59% | 50% | 56% |
| v5c | 68.7% | 70% | 78% | 56% |

**All three regressed from v4.** Pattern: adding more text creates unintended interactions.
v5a's IE regression unexplained (argument-from-design text somehow hurt IE recognition).
v5b's DT regression severe — "embodies divine thought" addition confused the model about DT.
v5c's softer NDT discriminator made NDT WORSE (more NDT→junk, not less).

**Learning:** The prompt is now too long/complex for gpt-5.4-mini. Each new clause gets
applied too broadly, creating cross-tag interference. More text ≠ better performance.

---

## Iteration 3: v6a / v6b / v6c

**Strategy:** Simplification over addition.

- v6a: Radical simplification — strip to essentials (~half the length), no discriminator paragraphs
- v6b: v4 + one SHORT argument-from-design bullet (concise vs. v5a's epistemological explanation)
- v6c: Positive-first junk gate — rewrite negative exclusions as a single positive test

**Results:**

| Variant | Avg | DT recall | IE recall | NDT recall | Junk recall |
|---|---|---|---|---|---|
| v4 (best) | 71.4% | 70% | 78% | 67% | 81% |
| v6a | 58.5% | 56% | 61% | 44% | 75% |
| v6b | 68.7% | 78% | **89%** | 62% | 70% |
| v6c | 63.3% | 41% | **89%** | 56% | 77% |

**v6a** proved the v2→v4 structural additions are doing useful work. Stripped prompt loses 13pp.

**v6b**: IE recall jumped to 89% (best seen). But junk recall collapsed to 70% because
the argument-from-design bullet backfired: model read "argument from design involves God
→ divine_teleology." junk→DT false positives spiked from 3 to 8.

**v6c**: Positive-first framing collapsed DT recall to 41%. The "would be false under a
different theory of animals" test is too abstract for the model to apply to DT passages.

**Learning:** IE improvement in v6b/v6c appears to be variance (3 runs too few to confirm).
v4 remains best. The argument-from-design fix needs explicit "→ junk, NOT divine_teleology"
redirect to avoid the DT misfire.

---

## Iteration 4: v7a / v7b / v7c / v7d

**Strategy:** Force better reasoning; precise argument-from-design fix; synthetic examples.

- v7a: Structured CoT — require explicit Step 1 answer in reasoning before tag
- v7b: v4 + argument-from-design bullet with explicit "→ junk, NOT divine_teleology" redirect
- v7c: v4 + 3 abstract junk patterns (divine creation without telos, arg-from-design, mechanism description)
- v7d: v2 + only the two demonstrably useful changes (v3c surface trigger + v4 NDT discriminator)

**Results:**

| Version | Avg | DT recall | IE recall | NDT recall | Junk recall |
|---|---|---|---|---|---|
| v7a (CoT) | 58.2% | 14% | 58% | 63% | 82% |
| v7b (arg-from-design) | 72.1% | 70% | 83% | 67% | 81% |
| **v7c** (3 junk patterns) | **74.8%** | 74% | 56% | 72% | 91% |
| v7d (v2 + useful only) | 70.7% | 89% | 61% | 64% | 77% |

**Winner: v7c (74.8%).** Three abstract junk patterns fixed eye+hand (0/3), partially fixed Ufefulnefs (1/3) and NS-never-produce (1/3). Cost: IE recall 78%→56% (microscopic identity passage 3/3 junk).

**v7a definitively dead:** CoT format collapses DT recall to 14%. Do not revisit.

**Per-target passage (v7c):**
| Passage | v4 | v7c |
|---|---|---|
| Ufefulnefs (junk) | 3/3 | 1/3 |
| NS never produce (junk) | 3/3 | 1/3 |
| eye and hand (junk) | 3/3 | 0/3 ✓ |
| NS specialise organ (NDT) | 3/3 | 3/3 |
| two organs modified (NDT) | 3/3 | 3/3 |
| microscopic identity (IE) | variable | 3/3 (new regression) |

---

## Iteration 5: v8a / v8b / v8c

**Base:** v7c (74.8%). **Targets:** IE recall recovery + NDT stuck pair.

- v8a: v7c + IE safeguard (structural identity across type-members = IE, not junk)
- v8b: v7c + NDT discriminator clarification (grammatical subject doesn't determine tag, result does)
- v8c: v7c + both v8a and v8b combined

**Results:**

| Version | Avg | DT recall | IE recall | NDT recall | Junk recall |
|---|---|---|---|---|---|
| v7c (base) | 74.8% | 74% | 56% | 72% | 91% |
| v8a (IE safeguard) | 66.7% | ~52% | ~88% | ~51% | ~84% |
| v8b (NDT grammatical) | 65.3% | ~33% | ~61% | ~72% | ~84% |
| v8c (both) | 69.4% | ~33% | ~89% | ~74% | ~83% |

**None beat v7c. Abandoning all v8 variants. v7c remains best.**

**Per-target passage:**
| Passage | v7c | v8a | v8b | v8c |
|---|---|---|---|---|
| microscopic identity (IE) | 3/3 | **0/3 ✓** | 3/3 | **0/3 ✓** |
| eye and hand (junk) | 0/3 | 0/3 | 0/3 | 0/3 |
| NS specialise organ (NDT) | 3/3 | 3/3 | 3/3 | 3/3 |
| two organs modified (NDT) | 3/3 | 3/3 | 3/3 | 3/3 |
| NS never produce (junk) | 1/3 | 2/3 | 3/3 NDT | 3/3 NDT |
| Ufefulnefs (junk) | 1/3 | 3/3 | 3/3 | 3/3 |

**What worked:** v8a's IE exception DID fix microscopic identity (0/3) in both v8a and v8c.
**What backfired:**
- v8a's extra IE text caused DT recall to collapse (~52% vs v7c's 74%) — cross-tag interference from added length.
- v8b's NDT clarification ("grammatical subject doesn't matter") broke NS-never-produce: now 3/3 NDT (wrong). The change is too broad — it makes mechanism passages pass as NDT.
- v8b did NOT fix the NDT stuck pair. Root cause: those passages use hypothetical framing ("natural selection MIGHT specialise…") and the model correctly reads that as not committing to what any organ IS.

**Key learning:** The NDT stuck pair is stuck because of HYPOTHETICAL phrasing ("might"), not grammatical subject. The organ isn't described as having a function — it's described as potentially acquiring one. This is a different problem than initially diagnosed. A fix must address the conditional framing directly.

---

## Iteration 6: v9a / v9b / v9c

**Base:** v7c (74.8%). **Targets:** IE recovery (one sentence only), NDT hypothetical fix, DT Agassiz fix.

- v9a: v7c + one-sentence IE exception (structural identity across type = IE)
- v9b: v7c + NDT hypothetical framing ("might specialise" = implicit functional claim → NDT)
- v9c: v7c + DT "embodies divine intellect" note

**Results:**

| Version | Avg | DT recall | IE recall | NDT recall | Junk recall |
|---|---|---|---|---|---|
| v7c (base) | 74.8% | 74% | 56% | 72% | 91% |
| v9a (IE sentence) | 68.0% | ~37% | ~89% | ~64% | ~84% |
| v9b (NDT hypothetical) | 71.4% | ~59% | 50% | ~79% | ~84% |
| v9c (DT embodies) | 65.3% | ~89% | ~67% | ~69% | ~62% |

**None beat v7c. All abandoned. v7c remains best.**

**Per-target passage:**
| Passage | v7c | v9a | v9b | v9c |
|---|---|---|---|---|
| NS specialise organ (NDT) | 3/3 | 3/3 | **2/3** | 3/3 |
| two organs modified (NDT) | 3/3 | 3/3 | **2/3** | 3/3 |
| microscopic identity (IE) | 3/3 | **0/3 ✓** | 3/3 | 3/3 |
| eye and hand (junk) | 0/3 | 0/3 | 0/3 | 0/3 |
| NS never produce (junk) | 1/3 | 2/3 | 2/3 NDT | 2/3 NDT |
| Ufefulnefs (junk) | 1/3 | 3/3 | 3/3 | 3/3 |

**What worked:**
- v9a: IE exception (one sentence) DID fix microscopic identity (0/3). But DT→mixed_dt-ie interference persists even with one sentence. Mechanism confirmed: any IE addition causes model to read DT passages as mixed_dt-ie (looking for structural component). **Stop pursuing IE fix.**
- v9b: FIRST EVER movement on NDT stuck pair (2/3 instead of 3/3). Hypothesis confirmed: hypothetical framing ("might") is the real barrier. But broke NS-never-produce (2/3 NDT). Fix too broad.
- v9c: DT recall jumped (Agassiz passages now mostly correct) but opened 5-6 junk→DT FPs per run. False positives are classification-system passages, not animal-characterization passages. Direction is right, needs guard.

**For v10:**
- v10a: v7c + narrow NDT fix: "organ committed to a specific function" — more precise than v9b, adds organ-specificity test and explicit guard against NS-never-produce pattern.
- v10b: v7c + refined DT extension: v9c's idea + guard that passage must characterize ANIMALS IN THEIR NATURE, not classification methodology.
- v10c: v7c + both v10a and v10b.

---

## Iteration 7: v10a / v10b / v10c

**Results:**

| Version | Avg | DT recall | IE recall | NDT recall | Junk recall |
|---|---|---|---|---|---|
| v7c (base) | 74.8% | 74% | 56% | 72% | 91% |
| v10a (narrow NDT) | 66.0% | ~52% | ~67% | ~59% | ~86% |
| v10b (refined DT) | 73.5% | ~71% | ~61% | ~74% | ~86% |
| v10c (both) | 67.3% | ~33% | ~67% | ~72% | ~87% |

**None beat v7c. All abandoned.**

Per-target: NS specialise/two organs = 3/3 junk across all variants. v10b best overall (73.5%, one run hit 75.5%). NS-never-produce regressed to 3/3 NDT in v10b (DT note about "divine thought about animals" contaminating NDT). Rule-based changes exhausted.

**For v11 — new approach: example-based + new junk pattern:**
- v11a: v7c + synthetic NDT example for process+result pattern (never tried example-based calibration)
- v11b: v7c + junk pattern (d) for "exhibits thought/plan" (parallelism/gradation, stuck since v1, never targeted)
- v11c: both combined

---

## Iteration 8: v11a / v11b / v11c

| Version | Avg | DT | IE | NDT | Junk |
|---|---|---|---|---|---|
| v7c | 74.8% | 74% | 56% | 72% | 91% |
| v11a (NDT mole example) | 70.8% | ~70% | ~56% | ~72% | ~84% |
| v11b (exhibits thought junk) | 70.7% | ~56% | ~50% | ~74% | ~90% |
| v11c (both) | 66.6% | ~33% | ~67% | ~69% | ~91% |

**None beat v7c. CRITICAL FIND: v11b's pattern (d) fixed parallelism/gradation (0/3) and mechanical weapons (0/3) — first fixes ever for both. But DT "exhibits thought" passages caught as junk, and NS-never-produce went 3/3 NDT.**

**For v12 (FINAL):**
- v12a: v7c + refined (d) with God-attribution guard
- v12b: v7c + refined (d) + pattern (e) for NS-never-produce mechanism-constraint
- v12c: v7c + clean 5-pattern rewrite (all learned patterns, concise)

---

## Iteration 9: v12a / v12b / v12c — FINAL

| Version | Avg | DT | IE | NDT | Junk |
|---|---|---|---|---|---|
| v7c (base) | **74.8%** | 74% | 56% | 72% | 91% |
| v12a (refined d) | 68.7% | ~37% | ~67% | ~82% | ~86% |
| v12b (d + e) | 68.7% | ~52% | ~67% | ~67% | ~86% |
| v12c (5-pattern rewrite) | 68.0% | ~37% | ~67% | ~77% | ~89% |

**None beat v7c. Loop complete. v7c (74.8%) is the final best prompt.**

Per-target (v12 best case):
| Passage | v7c | best v12 |
|---|---|---|
| NS specialise organ (NDT) | 3/3 | 3/3 (all) |
| two organs modified (NDT) | 3/3 | 3/3 (all) |
| Ufefulnefs (junk) | 1/3 | 3/3 (all) |
| NS never produce (junk) | 1/3 | 0/3 in v12c |
| eye and hand (junk) | 0/3 | 0/3 (all) |
| parallelism/gradation (junk) | 3/3 | 1/3 in v12c |
| mechanical weapons (NDT) | 2/3 | 2/3 in v12a |

v12c's 5-pattern rewrite fixed NS-never-produce (0/3) and improved parallelism (1/3) but DT recall collapsed massively (3/3 wrong for most DT passages). The rewrite disturbs the whole gate calibration.
