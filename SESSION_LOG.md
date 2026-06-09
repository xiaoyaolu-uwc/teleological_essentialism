# Session Log — Teleological Essentialism Classifier

Running log of work sessions. Add a new entry at the top for each session.
Purpose: allow any future session to pick up exactly where the last left off.

---

## Session: June 8, 2026

### What was decided

**Goal:** Build a classification system reaching 90%+ on the golden test set
(49-row `data/evaluation_set.csv`), without overfitting to specific passages.
Then use it to label 6000+ passages from the anchor corpus, and fine-tune BERT.

**Three parallel tracks:**
1. **Single-layer mini** — gpt-5.4-mini, single prompt, picking up from v7c (74.8% avg)
2. **Single-layer full** — gpt-5.4 (full model), single prompt, starting from v7c baseline
3. **Two-layer** — L1 binary junk gate + L2 category classifier, gpt-5.4-mini, starting fresh

**Loop structure (all tracks):**
- 5 runs per variant (increased from 3)
- Each generation: identify target passages → write 3 variants → run eval → check if targets improved → promote winner
- Rule: changes justified by category-definition logic, not passage-specific surface features
- Flag principle violations explicitly in the evolution file before making them

**Key decisions:**
- `mixed_dt-ie` → `divine_teleology`, `mixed_ndt-ie` → `non_divine_teleology` (eval set collapsed)
- Author metadata NOT passed to model during eval or inference (text-only)
- No held-out test set split — rely on principle-driven prompting discipline instead

### What was built this session

| File | Status | Notes |
|---|---|---|
| `data/evaluation_set.csv` | Modified | mixed tags collapsed (2 rows changed) |
| `models/prompts.py` | Modified | VALID_TAGS_V2 stripped of mixed; user templates updated |
| `models/two_layer_prompts.py` | New | L1_PROMPT_VERSIONS (l1_v1) + L2_PROMPT_VERSIONS (l2_v1) |
| `models/two_layer.py` | New | TwoLayerModel class |
| `eval/evaluate.py` | Modified | --method two-layer, --l1-version, --l2-version flags; default runs=5 |
| `eval/analyze_errors.py` | Modified | --two-layer flag; L1 and L2 separate analyses |
| `eval/single_mini_evolution.md` | New | Full history from prompt_evolution.md seeded in |
| `eval/single_full_evolution.md` | New | gpt-5.4 baseline noted (68.7% avg on v7c, 3 runs) |
| `eval/two_layer_evolution.md` | New | Bootstrap design rationale documented |
| `SESSION_LOG.md` | New | This file |

### Current state of each track

**Single-layer mini**
- Current best: v7c, 74.8% avg (3 runs — re-run with 5 to establish proper baseline before gen 1)
- Evolution file: `eval/single_mini_evolution.md`
- Results dir: `eval/results/v7c/`
- Stuck passages: NS specialise organ (NDT), two organs modified (NDT), Ufefulnefs (junk), NS never produce (junk), parallelism/gradation (junk), microscopic identity (IE)
- Hard constraints: no CoT format, no IE-specific additions, no broad NDT hypothetical text

**Single-layer full**
- Current state: v7c on gpt-5.4, 68.7% avg (3 runs only — need 5-run baseline)
- Evolution file: `eval/single_full_evolution.md`
- Results dir: `eval/results/v7c_gpt54/`
- **Next action: run 5 evals of v7c on gpt-5.4 to establish proper baseline + per-class breakdown**

**Two-layer**
- Current state: l1_v1 + l2_v1 defined but not yet run
- Evolution file: `eval/two_layer_evolution.md`
- **Next action: run 5 evals of (l1_v1, l2_v1) on gpt-5.4-mini to establish bootstrap baseline**

### Next actions (in order)

1. Run 5-run baselines for all three tracks before starting generation loops:
   ```bash
   # Single-layer mini (re-establish with 5 runs)
   python3 eval/evaluate.py --runs 5 --run-dir v7c_5run --version v7c --model gpt-5.4-mini
   python3 eval/analyze_errors.py --run-dir v7c_5run

   # Single-layer full (establish proper baseline)
   python3 eval/evaluate.py --runs 5 --run-dir v7c_full_5run --version v7c --model gpt-5.4
   python3 eval/analyze_errors.py --run-dir v7c_full_5run

   # Two-layer bootstrap
   python3 eval/evaluate.py --method two-layer --l1-version l1_v1 --l2-version l2_v1 \
       --runs 5 --run-dir two_layer_bootstrap --model gpt-5.4-mini
   python3 eval/analyze_errors.py --run-dir two_layer_bootstrap --two-layer
   ```

2. Read the three error analyses and seed generation 1 for each track.
