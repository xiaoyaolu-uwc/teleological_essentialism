# HANDOFF — DELETE THIS FILE AFTER READING

## What we are doing

Optimizing a text classifier prompt for historical biology passages. The pipeline classifies passages into: `divine_teleology`, `non_divine_teleology`, `internal_essence`, `junk`, `mixed_ndt-ie`, `mixed_dt-ie`.

The eval harness lives at `eval/evaluate.py`. Run it as:
```bash
python3 eval/evaluate.py --runs 3 --run-dir <version_name>
```
Then analyze:
```bash
python3 eval/analyze_errors.py --run-dir <version_name> --no-csv
```

All prompt versions live in `models/prompts.py` in `PROMPT_VERSIONS`. The current best is **v4** (71.4% avg) but **v7c** just ran and hit **74.8% avg** (75.5%, 71.4%, 77.6% across 3 runs) — making it the new best, unconfirmed pending full analysis.

`eval/prompt_evolution.md` contains the full iteration log. Read it.

---

## Current state

**Iteration 4 (v7a/b/c/d) ran but disk access died before analysis.** Results from stdout:

| Version | Run1 | Run2 | Run3 | Avg | Target |
|---|---|---|---|---|---|
| v7a | failed | — | — | — | CoT reasoning format |
| v7b | 73.5% | 71.4% | 71.4% | 72.1% | argument-from-design bullet |
| v7c | 75.5% | 71.4% | 77.6% | **74.8%** | 3 abstract junk patterns |
| v7d | 73.5% | 67.3% | 71.4% | 70.7% | v2 + only useful changes |

**v7c is the new best.** It added three abstract pattern descriptions to the Step 1 junk gate targeting: (a) divine creation without telos, (b) argument-from-design, (c) mechanism description. These are the three junk passages that have been stuck at 3/3 wrong since v1.

---

## What to do next (in order)

### 1. Restore disk access
System Settings → Privacy & Security → Full Disk Access → enable Terminal.

### 2. Run full analysis on v7c
```bash
python3 eval/analyze_errors.py --run-dir v7c --no-csv
```
Check whether the 3 target junk passages (Ufefulnefs, NS never produce, eye and hand) actually improved vs v2 baseline (all were 5/5 wrong in v2).

### 3. Re-run v7a (CoT variant — lost to disk failure)
```bash
sed -i '' 's/^PROMPT_VERSION = .*/PROMPT_VERSION = "v7a"/' eval/evaluate.py
python3 eval/evaluate.py --runs 3 --run-dir v7a
```
v7a uses `USER_PROMPT_TEMPLATE_V7A` which requires "Step 1: [yes/no]. Step 2: [tag]" reasoning format. Designed to force explicit gate application.

### 4. Synthesize v8 from v7c + corpus examples
v7c made abstract pattern descriptions. v8 should add **few-shot synthetic examples** from outside the eval set. A subagent was launched to find examples from `data/sentences.csv` but failed (no Bash access). Do this search manually:

```bash
python3 -c "
import csv
rows = list(csv.DictReader(open('data/sentences.csv')))
# Pattern 1: NDT — organ characterized by function (not from eval set)
ndt = [r for r in rows if r.get('scan_tag') == 'non_divine_teleology' 
       and int(r.get('word_count','999')) < 120][:20]
for r in ndt[:5]:
    print(r['author'], '|', r['text'][:200])
    print()
"
```

Search for:
- **NDT example**: organ described as specialised/suited for a use, no God mention, short (<120 words)
- **Junk/DT-boundary**: divine creation language without "for" purpose claim
- **Junk/NDT-boundary**: mechanism description using teleological-sounding language

**Constraint:** No examples from the 49-row eval set (`data/evaluation_set.csv`). Cross-check any candidate text against that file.

### 5. Design v8 variants
Build on v7c as base. Add few-shot examples to the system prompt as a "EXAMPLES" section after the tag definitions. Keep to 2-3 examples maximum. Each example should have: passage text (synthetic or from corpus), correct tag, and one-line explanation of why.

Target passages for v8:
- NS specialise organ (ndt, stuck 3/3 since v1) 
- two organs modified (ndt, stuck 3/3 since v1)
- Ufefulnefs (junk, stuck everywhere)

### 6. Run v8 and analyze per-target methodology
For each variant, record:
- **Target passages** and their v2 baseline wrong count
- **Post-run wrong count** for those passages specifically
- **Overall accuracy change**

---

## Key constraints (user rules)
- **No examples from the 49-row eval set** (`data/evaluation_set.csv`) or close paraphrases
- Examples from `data/sentences.csv` or `data/promising_passages.csv` are allowed
- Synthetic examples are allowed
- Prefer abstract description wins over example-dependent wins
- Stop loop if avg accuracy exceeds 80% over 3 runs, or after 6 total iterations

## Performance history (avg accuracy)
v1: ~66% → v2: 64.9% → v3c: 70.1% → v4: 71.4% → **v7c: 74.8%** (new best, unconfirmed)

## The 10 permanently stuck passages (wrong in every version so far)
See `eval/prompt_evolution.md` for the full table. The stuck ones:
- Junk mislabeled: Ufefulnefs, parallelism/gradation, NS never produce, eye and hand
- NDT mislabeled as junk: NS specialise organ, two organs modified  
- Mixed never correctly identified: man last+highest, domestic/drooping ears, microscopic identity (variable)
- Also: mechanical weapons (ndt) — partially fixed by v4 to 1/3

---
*Delete this file after reading.*
