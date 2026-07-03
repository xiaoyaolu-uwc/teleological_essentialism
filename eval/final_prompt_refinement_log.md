# Final Prompt Refinement Log

**Baseline**: f7b — 91.0% (5-run avg on gpt-5.4)  
**Primary goal**: eliminate overfit (eval-set verbatim quotes) while preserving accuracy  
**Secondary goal**: produce a Pareto frontier from most-generalized to most-accurate

---

## Element Registry — Final Verdicts

| ID | Element | Location | Chars | Removed Δ | Verdict |
|----|---------|----------|-------|-----------|---------|
| E1 | Pattern (e) junk bullet | Step 1 bullets | 341 | −0.4pp | **CUTTABLE** |
| E9_JUNK | "70–90% junk" sentence | IMPORTANT GUIDELINES | 50 | ~0pp alone, −5pp with E1 | keep (interaction with E1) |
| E9_CAMP | Camp note | IMPORTANT GUIDELINES | 284 | −4.0pp | **LOAD-BEARING** |
| E3–E8 | All named examples in DT NDT-clarifier block | DT section | 764 | −7.3pp | **LOAD-BEARING** |
| E3–E8 partial | Named examples minus "might with ease" | DT section | ~700 | −5.9pp | **LOAD-BEARING** |
| NDT compress | Compress 1314-char NDT block to ~550 | DT section | 764 | −8.7pp | **LOAD-BEARING** |

---

## Variant Log (gpt-5.4 only — mini results excluded as wrong model)

| Variant | Changes vs f7b | Chars | Acc | Δ vs f7b | Notes |
|---------|---------------|-------|-----|----------|-------|
| f7b | baseline | 8580 | 91.0% | — | winner pre-refinement |
| r1b | −E1 | 8239 | 90.6% | −0.4pp | **WINNER** |
| r1a | −E9_CAMP −E9_JUNK | 8530 | 88.4% | −2.6pp | camp note removal |
| r2a | −E9_CAMP only (5 runs) | 8296 | 87.0% | −4.0pp | confirms camp note load-bearing |
| r2b | compress E1 (shorter) | 8437 | 89.8% | −1.2pp | acceptable but r1b is cleaner |
| r1c | −all named examples | 8219 | 83.7% | −7.3pp | named examples load-bearing |
| r2c | −named examples, keep "might with ease" | 8285 | 85.1% | −5.9pp | still load-bearing |
| r1d | −E9 −E1 −examples | 7828 | 84.4% | −6.6pp | interaction effects |
| r3a | compress NDT block | 7816 | 82.3% | −8.7pp | too lossy |
| r3b | compress NDT + −E1 | 7475 | 84.4% | −6.6pp | too lossy |
| r3c | compress NDT + −E1 + −camp | 7191 | 83.7% | −7.3pp | too lossy |
| c1a_fixed | −E1 −E9_JUNK | 8189 | 85.7% | −5.3pp | interaction: both cuts compound |
| r4a | all eval quotes → synthetic examples | 8624 | 88.4% | −2.6pp | no eval quotes |
| r5c | r4a + abstract NDT block + −E1 | ~8280 | 82.7% | −8.3pp | abstract ceiling |
| r6a | r4a + functional-class rule | 8920 | **90.6%** | **−0.4pp** | **no eval quotes, matches r1b** |
| r6b | r6a + tighter CRITICAL EXCEPTION | ~8920 | 87.8% | −3.2pp | exception over-triggers |
| r6c | r6b − pattern_e | 8913 | 90.2% | −0.8pp | no eval quotes, shorter |

---

## Key Findings

### 1. Pattern (e) is cuttable (E1)
Pattern (e) guards against classification-question passages being labeled DT. On gpt-5.4, the model handles this correctly without the explicit rule (−0.4pp). Model-specific: was catastrophic on gpt-5.4-mini (−21.6pp).

### 2. Camp note is load-bearing despite being a "no-op"
Even though no camp metadata is passed in text-only mode, "Judge each passage on its own content, not the author's camp" anchors the model to passage-specific reasoning (−4.0pp when removed, stable over 5 runs).

### 3. Named examples are category-disambiguation aids, not text-matching shortcuts
The DT section's 1314-char NDT-clarifier block (with examples like "might with ease be modified", "savage instinctive hatred of the queen-bee") cannot be abstracted without large accuracy loss (−7.3pp). However, **replacing all eval-set quotes with synthetic invented examples costs only −2.6pp** (r4a = 88.4%), and adding a functional-class rule recovers all but 0.4pp → r6a = 90.6%.

**Conclusion**: the examples work via genuine principle illustration, not text-matching. Synthetic examples convey the same principle with equal effectiveness once the right rules accompany them.

### 4. Interaction effect between E1 and E9_JUNK
Removing pattern(e) alone = −0.4pp. Removing junk sentence alone ≈ 0pp. Removing both = −5.3pp. Both guard against junk→DT bleeding jointly.

### 5. Abstract-rule ceiling is ~82–83%
Replacing the NDT example block with pure type-level descriptions (r3a, r5c) consistently yields 82–83% regardless of how the abstraction is written. The named-example structure is genuinely load-bearing for boundary cases.

### 6. Functional-class rule closes the synthetic-example gap
r4a's remaining −2.6pp gap vs f7b was mostly driven by the "drooping ears" (domestic animals) passage: a functionally-defined class described by a structural correlate, which the model kept calling junk. Adding the explicit rule "a passage about a functionally-defined class of animals is NDT regardless of structural content" → r6a = 90.6%, recovering nearly all the gap.

### 7. Irreducible failures (fail in both f7b AND r6a, 5/5)
Four passages cannot be fixed by prompt engineering at this accuracy level:
1. "vague sort of notion…mechanical weapons" — NDT→DT (irreducible: model fixates on "weapons")
2. "exhibits thought…could only be the result of thought" — junk→DT (irreducible: looks like DT conclusion)
3. "systems have been considered as the expression of the views" — junk→IE (irreducible: ambiguous passage)
4. "hopeless is the attempt to explain similarity of pattern" — DT→junk (irreducible: label rationale hard to infer)

The remaining 0.4pp gap between r6a and f7b is explained by passage #5 ("mutual dependence…exhibits thought") failing 2/5 in r6a vs 0-1/5 in f7b — a borderline passage, not a systematic failure.

---

## Pareto Frontier (final)

All three prompts contain **zero eval-set verbatim quotes**. All accuracy figures are 5-run averages on gpt-5.4.

| Prompt | Accuracy | Chars | vs f7b | Eval quotes? | What's in it |
|--------|----------|-------|--------|--------------|--------------|
| **r3a** | 82.3% | 7816 | −8.7pp | None | Abstract type-level rules only. No examples of any kind. Best for new datasets where f7b's examples might create false anchors. |
| **r4a** | 88.4% | 8624 | −2.6pp | None | All eval quotes replaced by invented synthetic examples illustrating the same principles. Functionally equivalent abstraction. |
| **r6a** | 90.6% | 8920 | −0.4pp | None | r4a + explicit functional-class rule (NDT via class label, not content). Matches f7b's reference accuracy without text-matching. |
| *(f7b)* | *(91.0%)* | *(8580)* | *—* | *Has quotes* | *Baseline. Contains verbatim eval-set quotes in NDT-clarifier block and pattern (d)/(e) examples.* |

**Recommended deployment order**:
- **New corpus with unknown genre distribution** → r3a (no anchoring to 19th-century biology phrasing)
- **Same corpus, higher generalization priority** → r4a (synthetic examples, category principles only)
- **Same corpus, accuracy priority** → r6a (near-f7b accuracy with zero text-matching risk)

---

## Round Notes

### Round 1 — Low-risk deletions
Variants: r1a (−E9_CAMP −E9_JUNK), r1b (−E1), r1c (−named examples), r1d (all three)  
Finding: r1b safe (−0.4pp); r1c load-bearing (−7.3pp); camp note load-bearing (−4.0pp)

### Round 2 — Disentanglement + compression
Variants: r2a (−E9_CAMP only), r2b (compress E1), r2c (partial examples)  
Finding: camp note alone = −4.0pp stable; named examples still −5.9pp even partial

### Round 3 — Radical NDT block compression
Variants: r3a (compress NDT), r3b (+−E1), r3c (+−camp)  
Finding: abstract type rules lose −8.7pp; r3a = 82.3% is the abstract floor

### Consolidation — c1a_fixed
f7b − E1 − E9_JUNK = 85.7% (−5.3pp). Confirmed interaction effect.

### Round 4 — Synthetic example replacement
r4a: all eval-set quotes → invented synthetic examples. 88.4% (−2.6pp).  
Finding: 2.6pp cost of going fully synthetic; examples work by principle illustration not text-match.

### Round 5 — Improved synthetic examples
r5a (better domestic/d examples): worse (−2.6pp vs r4a). "Displays a regular progression" example broke DT passages.  
r5c (abstract NDT block): 82.7%, confirming abstract ceiling.  
Reverted to r4a as base.

### Round 6 — Functional-class rule
r6a (r4a + functional-class rule): **90.6%** — closes nearly all of the synthetic-example gap.  
r6b (r6a + tighter CRITICAL EXCEPTION): 87.8% — over-triggers, breaks DT passages.  
r6c (r6b − pattern_e): 90.2% — viable alternative to r6a, slightly shorter.
