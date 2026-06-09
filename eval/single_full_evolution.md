# Single-Layer Prompt Evolution — gpt-5.4 (full)

Tracks each generation of the single-layer prompt optimization loop on gpt-5.4 (full model).
Same loop structure as single_mini_evolution.md.

**Rule:** Every prompt change must be justified by an argument about what the category
definition requires. Flag any passage-specific changes explicitly before making them.

**Starting point:** v7c prompt (carried from mini track) as baseline.
gpt-5.4 on v7c: 68.7% avg (71.4%, 69.4%, 65.3% — 3 runs).
Note: v7c was optimized for mini behavior; full model may respond differently to the same
prompt structure. Optimization here starts fresh from v7c as base.

---

## Eval set composition (as of June 2026)
49 passages. mixed_dt-ie and mixed_ndt-ie collapsed into divine_teleology and
non_divine_teleology respectively.
- junk: 19
- non_divine_teleology: 14
- divine_teleology: 10
- internal_essence: 6

---

## Baseline: v7c on gpt-5.4 (3 runs, 68.7% avg)

Result files: eval/results/v7c_gpt54/

**Known from mini history that applies here:**
- v7c's 3 abstract junk patterns are load-bearing structure — do not remove
- CoT format (v7a) is dead — never revisit
- IE-specific additions cause DT interference — avoid

**What is unknown for full model:**
- Whether the full model's failure modes mirror mini's or differ
- Whether the full model responds better or worse to length/complexity
- Per-class recall breakdown for full model on v7c (need 5-run analysis to establish)

**Next action:** Run 5 eval runs on gpt-5.4 with v7c to establish proper baseline with
per-class breakdown, before designing generation 1 variants.

---

## Generation 1

*To be filled after 5-run baseline is established.*
