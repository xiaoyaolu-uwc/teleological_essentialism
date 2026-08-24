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

## Baseline: v7c on gpt-5.4 (5 runs, new eval set)

Run dir: `eval/results/single_54/baseline/`
**Overall: 69.8% avg** (342/490 correct across 5 runs)

Per-class recall:
- divine_teleology: 24% (12/50) ← **catastrophic**
- internal_essence: 87% (26/30)
- junk: 95% (90/95)
- non_divine_teleology: 61% (43/70)

The full model's failure profile is entirely different from mini's: near-perfect junk recall
but catastrophic DT recall. Mini has balanced failures; full almost never calls a junk passage
DT/NDT but calls most DT passages junk.

### Stuck passages (5/5 wrong):

| Passage snippet | Correct | Predicted |
|---|---|---|
| "order and arrangement of our studies are...indisputable" | divine_teleology | junk ×5 |
| "nature and foundation of our scientific classifications" | divine_teleology | junk ×5 |
| "Wyman...blind fish...created for a special object" | divine_teleology | junk ×5 |
| "organized beings exhibit...astonishing independence of physical conditions" | divine_teleology | junk ×5 |
| "mutual dependence of animal/vegetable kingdoms...exhibits thought" | divine_teleology | junk ×5 |
| "Owen...hopeless to explain similarity of pattern on principle of utility" | divine_teleology | junk ×5 |
| "man is not only the last and highest...last term of one of these natural series" | divine_teleology | junk ×5 |
| "merely mechanical weapons...vague sort of notion" | non_divine_teleology | divine_teleology ×4, junk ×1 |
| "drooping ears...domestic animals" | non_divine_teleology | junk ×5 |
| "natural selection might easily specialise...one part...one function" | non_divine_teleology | junk ×5 |
| "one of the two organs might with ease be modified and perfected" | non_divine_teleology | junk ×5 |
| "instinctive hatred of the serpent" | non_divine_teleology | junk ×5 |

---

## Generation 1

### Target passages

Primary (rescue DT recall from 24%):
1. **"order and arrangement of our studies"** (DT → junk)
2. **"nature and foundation of our scientific classifications"** (DT → junk)
3. **"mutual dependence...exhibits thought"** (DT → junk)
4. **"man is not only the last and highest"** (DT → junk)

Secondary:
5. **"natural selection might easily specialise"** (NDT → junk)
6. **"drooping ears...domestic animals"** (NDT → junk)

Success criterion: DT recall rises from 24% → ≥50% (≥25/50 correct across 5 runs),
without junk recall falling below 85%.

---

### Root cause analysis

**DT catastrophe:** All 7 DT→junk failures share one trait: they argue divine teleology at a
SYSTEMIC or TAXONOMIC level — the plan of nature, the order of classification, ecological
interdependence — rather than naming a specific creature with a specific divine purpose. The
full model requires an explicit organism-level divine design claim and rejects abstract/
philosophical DT arguments as junk. This is a miscalibration: Agassiz's central thesis IS
that zoological classification reflects God's thought; claiming the whole plan of nature is
divinely ordered is a stronger DT claim than "this wing was made for flying."

Principle basis for fix: Extend DT to include claims that the ORDER, PLAN, or ARRANGEMENT
of the natural world reflects divine purpose, even when framed taxonomically or ecologically
and without naming a specific organism. This is conceptually DT: it attributes nature's
structure to divine intention.

Guard needed: "Observing a pattern that resembles order or design, without asserting a
divine source, remains junk." This prevents over-broad DT classification.

**NDT hypothetical + domestic function:** Same issues as mini track — hypothetical NS
constructions and domestic-animal functional observations. Same principle applies.

**Key asymmetry vs. mini:** Full model's junk gate is near-perfect (95%), so loosening it
carries more absolute FP risk. Strategy: strengthen DT's positive definition rather than
weaken the junk gate description.

---

### Three variants

**v7c_full_gen1a — DT systemic/taxonomic extension**

Targeted at: passages 1–4

Add to DT definition:
> "DT includes passages arguing that the plan, ordering, or systematic arrangement of the
> natural world reflects divine thought or purpose — even when no specific organism is
> named and no explicit creator-language appears. Claims that zoological classification,
> the hierarchical ordering of taxa, or the interdependence of organisms expresses divine
> intention are divine_teleology."
> "A passage that merely observes a pattern resembling order in nature, without asserting
> a divine source or purpose, remains junk."

Justification: Principle-driven — DT's scope should cover the full range of divine purpose
claims, including taxonomic/systemic ones. The current prompt implicitly restricts DT to
organism-level claims, which excludes canonical 19th-century natural theology.

Violation flag: None.

**v7c_full_gen1b — NDT hypothetical + domestic function fix**

Targeted at: passages 5, 6

Add to NDT definition:
> "NDT includes: (a) passages attributing specific functions to specific organs via natural
> processes, including hypothetical framing ('might acquire,' 'could be modified for');
> (b) passages noting that domesticated animals have traits correlated with their domestic
> function, where the function defines or distinguishes the animal. General principles
> about what natural selection does or does not produce (with no specific organ-function
> pair named) remain junk."

Justification: Same principle as mini gen1b.

Violation flag: None.

**v7c_full_gen1c — Both fixes combined**

Applies both gen1a and gen1b changes simultaneously.

Risk: The full model responds to prompt changes sharply (near-perfect junk gate). Any
loosening carries higher absolute FP risk. Monitor junk recall closely — floor is 85%.
