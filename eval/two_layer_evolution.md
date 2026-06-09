# Two-Layer Prompt Evolution

Tracks each generation of the two-layer pipeline optimization.
Architecture: L1 (binary junk/non-junk) → L2 (DT/NDT/IE on non-junk passages only).

**Loop structure:** Alternating — fix L2, iterate L1 for one generation; then fix L1, iterate L2.
Each layer has its own prompt registry in models/two_layer_prompts.py.
Error analysis produces separate l1_error_analysis.csv and l2_error_analysis.csv.

**Rule:** Every prompt change must be justified by an argument about what the category
definition requires. Flag any passage-specific changes explicitly before making them.

---

## Eval set composition (as of June 2026)
49 passages total. For L1 analysis: junk (19) vs non-junk (30).
For L2 analysis: DT (10), NDT (14), IE (6) — restricted to the 30 non-junk passages.

---

## Design rationale

**Why two layers improves on single-layer:**
The entire single-layer iteration history (9 iterations, 30+ variants) shows that fixing
the junk gate consistently breaks category recall and vice versa — they are coupled in one
prompt, so each change to one axis disturbs the other. Separate prompts allow independent
calibration: L1 is optimized purely for junk recall/precision; L2 is optimized purely for
category distinction on passages that are already known to be substantive.

**L1 prompt design:** Derived from v7c's Step 1 gate logic (the ontological position test +
3 abstract junk patterns). Simplified to binary output: junk / non_junk.

**L2 prompt design:** Derived from v7c with the junk gate entirely stripped. Only sees
non-junk passages. Outputs: divine_teleology / non_divine_teleology / internal_essence.

---

## Bootstrap baseline: l1_v1 + l2_v1 on gpt-5.4-mini (5 runs)

Run dir: `eval/results/two_layer/baseline/`
**Overall: 61.2% avg** (150/245 correct across 5 runs)

### L1 gate: 65.3% accuracy (160/245)

- junk recall: 83% (79/95)
- non_junk recall: 54% (81/150) ← **primary bottleneck**
- 69 FN: non-junk passages gated as junk (never reach L2)
- 16 FP: junk passages leaking through to L2

### L2 category: 47.3% on non-junk gold passages (71/150)

Note: most "error" predictions in L2 analysis are L1 FNs (passages that never reached L2,
predicted_tag=junk appears as "error" in L2-only view). L2 is effectively unmeasurable until
L1 improves — generation 1 focuses on L1 only.

### L1 stuck passages (5/5 wrong):

| Passage snippet | Correct | L1 predicted |
|---|---|---|
| "parallelism between gradation...exhibits thought" | junk | non_junk ×5 |
| "leading features of the animal kingdom" (exposition) | junk | non_junk ×5 |
| "FAMILY II SERPENTIA...true Serpents" | internal_essence | junk ×5 |
| "perfect identity of microscopic structures" | internal_essence | junk ×5 |
| "MODERN classifications...based upon peculiarities of structure" | internal_essence | junk ×5 |
| "nature and foundation of our scientific classifications" | divine_teleology | junk ×5 |
| "organized beings exhibit astonishing independence" | divine_teleology | junk ×5 |
| "drooping ears...domestic animals" | non_divine_teleology | junk ×5 |

L1 stuck (4/5 wrong):

| Passage snippet | Correct | L1 predicted |
|---|---|---|
| "mutual dependence...exhibits thought" | divine_teleology | junk ×4 |
| "Natural selection will never produce anything injurious" | junk | non_junk ×4 |

---

## Generation 1

**Focus: L1 gate** (L2 fixed at l2_v1 until L1 is stable)

### Target passages

Primary FN targets — non-junk being gated as junk (5/5):
1. **"FAMILY II SERPENTIA...true Serpents"** (IE → junk)
2. **"perfect identity of microscopic structures"** (IE → junk)
3. **"MODERN classifications...based upon peculiarities of structure"** (IE → junk)
4. **"drooping ears...domestic animals"** (NDT → junk)

FP targets to guard — junk leaking as non-junk (5/5):
5. **"parallelism between gradation...exhibits thought"** (junk → non_junk)
6. **"leading features of the animal kingdom"** (junk → non_junk)

Success criterion: ≥3 of 4 primary FN targets drop to ≤2/5 wrong on L1,
without junk FP rate worsening (FP must stay ≤20/95 = ≤21%).

---

### Root cause analysis

**FN failure (non-junk gated as junk):** The three IE structural-classification passages
describe structural characteristics as the basis for defining/classifying animal groups. The
L1 model reads structural description as mere enumeration (junk), missing that "classifications
ARE BASED ON structure" is a definitional claim — it asserts structural characteristics DEFINE
animal groups (internal essence). The fix must clarify that a passage claiming structural
properties ARE the basis for animal classification is making a substantive non-junk claim.

Drooping ears NDT: domestic-animal functional observation being gated out — the model
likely misses that attributing a trait to functional/behavioral causation (domestic function)
is a non-junk claim about animal nature.

**FP failure (junk leaking as non-junk):** The parallelism/gradation passage uses "exhibits
thought" (DT-like language) but makes no claim about what defines animals — it observes a
pattern in nature. The short-exposition passage mentions the "animal kingdom" but is
meta-textual (announcing what follows). Both lack a substantive definitional claim.

---

### Three variants

**l1_v2a — FN fix: recognize structural-definition and functional claims as non-junk**

Change: Add to non-junk definition:
> "Non-junk includes: (a) passages asserting that structural or anatomical characteristics
> ARE THE BASIS for defining, identifying, or classifying animal groups — this makes a
> claim about what fundamentally constitutes animals (internal essence); (b) passages
> attributing a specific trait, behavior, or functional characteristic to a class of
> animals and offering a causal explanation, even when the explanation is naturalistic."

Justification: Structural-definition claims ("classified BY their structural peculiarities")
are the core IE claim. The L1 gate should recognize that asserting a CRITERION for defining
animal groups is substantive, not descriptive.

Violation flag: None.

**l1_v2b — FP fix: tighten junk against meta-text and philosophical pattern-observation**

Change: Add to junk definition:
> "Junk includes: (a) meta-textual passages that announce or introduce subsequent content
> without themselves asserting anything about animal nature; (b) passages observing a
> pattern resembling design or order (e.g., a parallel between two natural processes)
> without claiming that the animals involved are defined by or serve any purpose. Using
> the phrase 'exhibits thought' or noting a resemblance to intelligent design, without a
> claim about what defines or explains the animals themselves, is junk."

Justification: Meta-text and philosophical observations are paradigm junk — they make no
claim about animal nature/definition/purpose.

Violation flag: None.

**l1_v2c — Both fixes combined**

Applies both l1_v2a and l1_v2b simultaneously. Independent failure modes → should be
additive. Monitor junk FP rate (≤21% ceiling).
