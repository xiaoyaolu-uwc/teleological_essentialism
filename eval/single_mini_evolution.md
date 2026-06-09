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

## Generation 1 (new loop)

*To be filled when next iteration begins.*
