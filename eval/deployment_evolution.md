# Deployment Prompt Evolution Log

Tracks iteration on the deployment prompt — the prompt used to label the full
`sentences.csv` for BERT fine-tuning. Unlike the classification eval loop,
this prompt must produce **five fields per passage**: `tag`, `extract`,
`reasoning`, `confidence`. Accuracy on the 49-row eval set must not regress
from the r6a baseline (90.6%).

**Baseline**: r6a — 90.6% accuracy on eval set (gpt-5.4, 5-run avg).  
**Model for deployment**: gpt-5.4 (same as r6a).  
**Eval script**: `eval/evaluate_deployment.py`  
**Analysis script**: `eval/analyze_deployment.py`  
**Results dir**: `eval/results/deployment/`

## What we check each iteration

**Automated (from analyze_deployment.py):**
- Overall accuracy vs r6a baseline (floor: 88%)
- Per-class recall (DT, NDT, IE, junk)
- Confidence calibration: accuracy within each confidence bucket — we want high-confidence predictions to actually be correct
- Extract word-count distribution, short/long outliers

**Manual review (every iteration):**
- **Extract correctness**: is the extract verbatim (modulo OCR fixes)? Does it actually contain the key claim?
- **Extract relevance**: for non-junk, does it quote the sentence that drives the label, or something generic? For junk, does it quote sentences around the animal mention?
- **OCR correction scope**: are corrections limited to character-level artifacts, or is the model rephrasing?
- **Reasoning quality**: does reasoning point to specific evidence and explain why this tag vs. adjacent categories? Or is it circular (restating the label)?
- **Confidence calibration (qualitative)**: do high-confidence predictions feel justified? Do the model's stated uncertainties match the actual difficulty of the passage?
- **JSON formatting**: any parse errors, missing fields, truncation?

---

## d_v1

**Prompt file**: `models/deployment_prompts.py`  
**Built on**: r6a system prompt (unchanged) + new five-field user template  
**Run dir**: `eval/results/deployment/d_v1/`  
**Run command**:
```bash
python3 eval/evaluate_deployment.py --version d_v1 --model gpt-5.4 --run-dir deployment/d_v1
python3 eval/analyze_deployment.py --run-dir deployment/d_v1
```

### Key design choices in d_v1

- Tag comes first in the JSON object — anchors the classification decision before extract/reasoning are written, reducing confirmation-bias risk.
- Extract instruction: verbatim, 1–3 sentences, OCR word-level correction only.
- Junk extract policy: quote 1–3 sentences around the animal mention, same length target as non-junk (so junk isn't identifiable by shorter output).
- Reasoning: 1–2 sentences, must point to specific evidence and distinguish from adjacent categories.
- Confidence: 0.0–1.0 with explicit calibration guidance (surprised to be wrong → ≥0.85; genuine toss-up → ≤0.65).
- Batch size: 5 (down from 10 in eval harness) to reduce truncation risk with longer per-row output.

### Results

**Accuracy: 85.7% (42/49)** — 4.9pp below r6a baseline (90.6%).

| Class | Recall | N |
|---|---|---|
| divine_teleology | 90% | 9/10 |
| internal_essence | 100% | 6/6 |
| junk | 79% | 15/19 |
| non_divine_teleology | 86% | 12/14 |

Confidence calibration: 50–75% bucket at 20% accuracy (5 rows) — well below confidence.
Over-extraction: 13/49 extracts >80 words.

### Wrong rows (7)

| True | Pred | Conf | Snippet | Diagnosis |
|---|---|---|---|---|
| NDT | DT | 0.66 | "law of growth...purpose is fulfilled" | Low-conf, borderline |
| junk | DT | 0.97 | "parallelism...established by a thinking being" | Permanently stuck passage |
| junk | NDT | 0.70 | "NS will never produce anything injurious" | Known stuck; general NS principle |
| junk | DT | 0.88 | "fixed for each species...comprehensive thoughts" | Hard Agassiz meta-passage |
| junk | IE | 0.72 | "existence in nature of distinct species" | Meta-commentary read as IE |
| **DT** | **IE** | **0.72** | **"created by fiat of Almighty...general plan of structure"** | **Format-caused: extract anchored on structural language, missed divine language** |
| **NDT** | **junk** | **0.80** | **"domestic animal...drooping ears...disuse of muscles"** | **Format-caused: extract focused on mechanism, missed functionally-defined class rule** |

### Manual review notes

Extracts: 13/49 >80 words, but otherwise verbatim. OCR corrections minimal and correct.
Reasoning: substantive for correct rows. Wrong rows show extract anchoring the wrong reading.
The two format-caused failures (rows 6, 7) confirm the hypothesis: extract field pulls model
toward the wrong textual anchor BEFORE reasoning is written.

### Next actions

d_v2: fix the two format-caused failures with tag-anchored extract guidance + length tightening.

---

## d_v2

**Prompt file**: `models/deployment_prompts.py`, key: `"d_v2"`  
**Changes vs d_v1**: tag-anchored per-class extract guidance; functionally-defined class reminder; "typically 20–60 words" length guidance.  
**Run dir**: `eval/results/deployment/d_v2/`  
**Run command**:
```bash
python3 eval/run_deployment_eval.py --version d_v2 --model gpt-5.4 --batch-size 5 \
    --out eval/results/deployment/d_v2/eval_gpt-5.4_d_v2.csv
python3 eval/analyze_deployment.py --run-dir deployment/d_v2
```

### Results

**Accuracy: 89.8% (44/49)** — 0.8pp below r6a baseline (90.6%). Within 1-run variance.

| Class | Recall | N |
|---|---|---|
| divine_teleology | 100% | 10/10 |
| internal_essence | 100% | 6/6 |
| junk | 79% | 15/19 |
| non_divine_teleology | 93% | 13/14 |

Confidence calibration: 50–75% bucket at 67% (3 rows); 75–90% at 82% (11 rows); 90%+ at 94% (35 rows). Well-calibrated.
Over-extraction: 6/49 extracts >80 words (down from 13).

### Wrong rows (5)

| True | Pred | Conf | Snippet | Diagnosis |
|---|---|---|---|---|
| NDT | DT | 0.68 | "law of growth...purpose is fulfilled" | Low-conf, borderline, recurring |
| junk | DT | 0.97 | "parallelism...established by a thinking being" | Permanently stuck |
| junk | DT | 0.83 | "Are they the devices of the human mind..." | Rhetorical question; correct in d_v1, new regression |
| junk | DT | 0.93 | "fixed for each species...comprehensive thoughts" | Hard Agassiz meta-passage |
| junk | DT | 0.82 | "systems considered as expression of man's understanding...not as devised by Supreme Intelligence" | Pattern (e): reports others' view, model reads as positive assertion |

Fixed vs d_v1: blind fish (DT→IE), drooping ears (NDT→junk), NS-never-produce (junk→NDT). Net gain: +2 correct.

### Manual review notes

**Extract quality (correct rows)**: on-point, 20–60 words typical, OCR corrections correct and minimal. Reasoning points to specific textual evidence, not circular. Good.

**Junk extracts**: sentence(s) around animal reference, similar length to non-junk. Policy working.

**Remaining failures**: all hard/borderline Agassiz passages at the junk/DT boundary — meta-discussion with divine language that pattern (d)/(e) should catch but model overrides. These same passages were problematic throughout the classification loop. Not format-caused.

**One new regression** ("Are they the devices..."): the tag-anchored extract guidance pointed the model toward the divine language in this rhetorical-question passage, pulling it to DT. Acceptable trade-off — overall +2 correct.

### Decision: d_v2 is NOT final — one more iteration for extract quality

One issue remained: extracts could quote abstract sentences with no animal referent (e.g. "Be it so: this law of growth, if it exist, is but itself an instrument whereby purpose is fulfilled" — no animal named). Adding a hard animal-referent constraint before running the full 12,913-row corpus.

---

## d_v3

**Prompt file**: `models/deployment_prompts.py`, key: `"d_v3"`  
**Change vs d_v2**: added HARD REQUIREMENT that extract must include at least one sentence where an animal, animals, or animal part is explicitly named. If the key claim appears in an animal-free sentence, include that sentence AND the nearest sentence naming the animal/part.  
**Run dir**: `eval/results/deployment/d_v3/`  
**Run command**:
```bash
python3 eval/run_deployment_eval.py --version d_v3 --model gpt-5.4 --batch-size 5 \
    --out eval/results/deployment/d_v3/eval_gpt-5.4_d_v3.csv
python3 eval/analyze_deployment.py --run-dir deployment/d_v3
```

### Results

**Accuracy: 91.8% (45/49)** — 1.2pp ABOVE r6a baseline (90.6%). Best result in any single run.

| Class | Recall | N |
|---|---|---|
| divine_teleology | 90% | 9/10 |
| internal_essence | 100% | 6/6 |
| junk | **95%** | 18/19 |
| non_divine_teleology | 86% | 12/14 |

Junk recall: 79% (d_v1) → 79% (d_v2) → **95% (d_v3)**. The animal-referent constraint forced context that helped the model correctly apply patterns (d) and (e).

Confidence calibration: 75–90% bucket at 79% (14 rows); 90%+ at 97% (33 rows). Well-calibrated.
Over-extraction: 10/49 extracts >80 words — acceptable.

### Wrong rows (4)

| True | Pred | Conf | Snippet | Diagnosis |
|---|---|---|---|---|
| NDT | junk | 0.84 | "mechanical weapons, organs of attack...law of growth" | Changed from NDT→DT in d_v2; still wrong, different error |
| DT | junk | 0.88 | "birth of new Species...mental purpose of producing beauty" | **Regression from d_v2** — was correct; "Species" not read as animal referent |
| junk | DT | 0.95 | "parallelism...established by a thinking being" | Permanently stuck |
| NDT | junk | 0.78 | "one of the two organs might with ease be modified" | **Regression from d_v2** — was correct; lower confidence |

Fixed vs d_v2: 3 junk→DT Agassiz passages now correctly junk. 2 regressions introduced. Net: +1 correct.

### Manual review notes

**Extract quality**: animal-referent constraint working — abstractions now paired with animal-naming context. No extracts quote a principle in isolation from its animal subject.

**Remaining failures**: permanently-stuck parallelism passage; two borderline passages that flipped from d_v2 (both have plausible junk readings and lower confidence); one recurring hard NDT passage.

### Decision: d_v3 is the FINAL deployment prompt

91.8% > 90.6% r6a baseline. Junk recall 95%. Extract quality improved by the animal-referent constraint. Full corpus run uses d_v3.

---
