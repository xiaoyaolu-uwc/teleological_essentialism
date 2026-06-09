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

## Bootstrap baseline

Model: gpt-5.4-mini (default for initial tuning).
L1 starting version: l1_v1 (see models/two_layer_prompts.py)
L2 starting version: l2_v1 (see models/two_layer_prompts.py)

**Next action:** Run 5 eval runs of (l1_v1, l2_v1) to establish bootstrap baseline
and produce initial l1_error_analysis.csv and l2_error_analysis.csv.

---

## Generation 1

*To be filled after bootstrap baseline is established.*
