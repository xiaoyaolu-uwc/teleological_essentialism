# Teleological Essentialism

A pipeline that reads historical scientific texts and measures **how they explain
animals** — by divine purpose, by naturalized function, or by internal structure —
so that shifts in explanatory style can be tracked across the 18th and 19th centuries.

Research context and taxonomy: [`PROJECT_DESC.md`](PROJECT_DESC.md).

---

## What this repo produces

A **two-stage classifier** that labels each sentence of a text as one of:

| Label | Meaning |
|---|---|
| **Divine teleology (DT)** | the animal is characterized by a purpose ordained or designed by a creator |
| **Non-divine teleology (NDT)** | characterized by the function or end it serves, with no designer — adaptation, ecological role |
| **Internal essence (IE)** | characterized purely by internal structure, composition, or type, with no reference to purpose |
| *junk* | makes no definitional claim about an animal at all — discarded |

Measured on **16 books the model never saw during training**, it performs as follows.

**Getting the proportions right.** For a text it has never read, the model's estimate
of each category's share of the explanatory sentences lands this close to the truth
**90% of the time**:

| Category | Within | Average error |
|---|---|---|
| Divine teleology | **±8.5 points** | 2.6 pts |
| Internal essence | **±15 points** | 5.6 pts |
| Non-divine teleology | **±16 points** | 6.0 pts |

There is **no systematic bias** in any category — every mean signed error is within
1.5 points of zero, so the model is noisy around the right answer rather than
tilted in a fixed direction.

**Getting the ranking right.** Ordering is considerably more reliable than
magnitude, which matters because the research questions are comparative:

| True gap between two texts | Ordering recovered correctly |
|---|---|
| 20+ points | **99%** |
| 10–20 points | 90% |
| 5–10 points | 84% |
| under 2 points | 54% — no better than chance |

Overall 90.4% across all 354 comparisons (95% CI 83.5–95.8%). Within a single book,
the full DT/NDT/IE ordering is recovered in **14 of 16 books**.

**Read that as:** a difference of 20 points or more between two periods is solid,
10–20 points is likely, and anything under 5 points should not be interpreted.

Full numbers, caveats, and the honest error bars:
[`docs/PROPORTION_EVAL_RESULTS.md`](docs/PROPORTION_EVAL_RESULTS.md).

---

## Setup

```bash
pip install -r requirements.txt
```

Only two of those packages are needed to **reproduce every published number** —
the analysis path is deliberately dependency-free. To recompute all the claims
above from the committed predictions, with no GPU and no ML stack:

```bash
python3 evaluation/evaluate_proportions.py
```

Training and inference need the full install plus a GPU. `.env` holds
`OPENAI_API_KEY` for the labelling step (see `.env.example`).

---

## Where everything lives

| Path | Role |
|---|---|
| `corpus/` | text → passages → animal-relevant sentences |
| `labelling/` | the GPT-5.4 labeller and its prompts — produces ground truth |
| `models/labels.py` | the label spaces. **No torch**, so analysis runs anywhere |
| `models/data.py` | loading, fold splits, scoring functions. Also torch-free |
| `models/lora/train.py` | trains either cascade stage, for any fold |
| `evaluation/` | the 5 live scripts — see [`evaluation/README.md`](evaluation/README.md) |
| `scripts/` | SLURM wrappers, job submission, run validation |
| `docs/` | current plan and results; `docs/history/` for superseded phases |
| `archive/` | finished phases, kept runnable |

---

# Part 1 — How the model is trained

## 1. Training data

16 anchor texts spanning **1691–1876**, chosen to represent each philosophical
camp: Paley, Ray, Derham and Kirby for divine teleology; Darwin, Gray and Lamarck
for naturalized function; Owen and Cuvier for structural accounts; Huxley and
Haeckel for mechanism.

`corpus/find_animal_chunks.py` splits these into ~300-word chunks and keeps only
those mentioning an animal or animal part (a ~170-term keyword list in
`config/config.py`). `corpus/extract_sentences.py` refines those into
sentence-level rows.

**Result: 12,873 rows** in `data/sentences_train.csv` — the training and evaluation
corpus for everything that follows.

## 2. Establishing ground truth

**The problem:** nobody had labelled those 12,873 rows, and hand-labelling them
was not feasible.

**The solution:** an LLM labeller that agrees with human judgement **91% of the
time** on a deliberately difficult 49-row hand-labelled set
(`data/evaluation_set.csv`, labelled by xiaoyao with input from David).

**How it works.** `labelling/run_full_deployment.py` sends each passage to GPT-5.4
with a long prompt that defines each category precisely and walks the model through
a decision procedure. The prompt lives in
[`labelling/deployment_prompts.py`](labelling/deployment_prompts.py) (version
`d_v3`), built on the category definitions in
[`labelling/prompts.py`](labelling/prompts.py).

**How the prompt was built.** Iteratively, and empirically:

1. Write a prompt; run it against the 49-row hand-labelled set.
2. Examine *exactly* which rows it got wrong and how often.
3. Infer the structural misunderstanding behind the errors — not "it missed this
   one", but "it treats any mention of a creator as divine teleology, even when the
   passage grounds the animal's character in structure".
4. Add a rule targeting that specific misunderstanding.
5. Repeat until agreement passed 90%.

The full trail is in [`docs/history/final_prompt_refinement_log.md`](docs/history/final_prompt_refinement_log.md)
and [`docs/history/deployment_evolution.md`](docs/history/deployment_evolution.md).

**Evaluation integrity.** Because the prompt was tuned *against* the 49-row set,
there was a real risk of overfitting to it — so no verbatim text from those rows
was ever placed in the prompt. When a candidate prompt (`f7b`) scored 91.0% partly
by quoting eval-set passages as examples, it was **rejected** in favour of `r6a`
at 90.6%, which uses synthetic examples instead. Slightly lower score, no leakage.

The resulting labels become the `deploy_tag` column. **Every number in this repo
is measured against these labels**, so the ~9% residual human–LLM disagreement is a
real floor on all of it. That caveat is stated wherever results appear.

## 3. Why fine-tune a small model at all

The GPT-5.4 labeller is accurate but **far too expensive to deploy**. The full
scanning corpus is on the order of **1 million rows** — roughly 300–500 million
tokens, or **$6,000–$10,000** per pass. That is not a pipeline you can iterate on.

So the frontier model is used *once*, to teach a small one. We fine-tune
**Qwen3-0.6B** — a model about a thousand times smaller — to imitate its labels.
It then runs locally on a single rented GPU: the same million rows take roughly
half a day and cost a few dollars of electricity.

### What LoRA fine-tuning is

Retraining all 600 million parameters of a model needs serious hardware. **LoRA
(Low-Rank Adaptation)** freezes the original model and trains a small set of extra
weights alongside it — here **9.2 million parameters, about 1.5% of the total**.

The analogy: rather than rewriting a textbook, you write margin notes. The book is
unchanged; the notes adapt it to your purpose. The notes are small enough to train
on modest hardware, quick to swap, and you can keep several sets for different
tasks against one underlying model.

### Why two stages, not one

**Because 65% of the corpus is junk**, and that lopsidedness smothers the
distinction we actually care about. In a single 4-way classifier, junk dominates
the training signal and the three real categories get squeezed together.

So the work is split:

| Stage | Job | Why it is separate |
|---|---|---|
| **1. Junk gate** | junk vs. everything else | Handles the easy, high-volume filtering. ~62% of rows are discarded here and never reach stage 2. |
| **2. Three-way classifier** | DT vs. NDT vs. IE | Trained *only* on real rows, so it sees a near-balanced problem and can learn the fine distinctions without junk drowning them out. |

Splitting them raised divine-teleology recall from 0.30–0.40 to 0.80–0.90 —
the largest single improvement in the project. The evidence is in
[`docs/history/bert_cascade_evolution.md`](docs/history/bert_cascade_evolution.md).

Both stages are trained by the same script,
[`models/lora/train.py`](models/lora/train.py), differing only in `--stage`:

```bash
# stage 1
python3 models/lora/train.py --run-name gate_fold4 --stage junk_gate \
    --holdout-fold fold4 --prompt-variant A_structured \
    --lora-r 32 --lora-alpha 64 --epochs 4 --text-column text --max-length 640

# stage 2
python3 models/lora/train.py --run-name s2_fold4 --stage nonjunk_3way \
    --holdout-fold fold4 --prompt-variant S2_structured \
    --lora-r 32 --lora-alpha 64 --epochs 4 --text-column text --max-length 640
```

**The adopted configuration**, chosen by a 16-run search over stage design, LoRA
rank, and prompt (two seeds each, ranked by proportion error rather than accuracy —
see `evaluation/compare_stage2_search.py`):

| Setting | Value | Why |
|---|---|---|
| Base model | Qwen3-0.6B | small enough to run the full corpus locally |
| LoRA rank / alpha | 32 / 64 | rank 32 beat rank 16 in the stage-2 search, matching the earlier junk-gate sweep |
| Epochs | 4 | more did not help |
| `max_length` | **640** | at 384, **27.8% of rows were being silently truncated** |
| Input column | raw `text` | the only column that exists for an unread book |

Two findings from that search are worth carrying forward. Giving stage 2 its own
junk class — so it could reject junk the gate leaked — **failed badly**
(macro-F1 0.59 vs 0.80): it has to relearn junk rejection from the 65%-junk corpus,
reintroducing the exact problem the cascade exists to remove. And the prompt
mattered far more for the *category mix* than for raw accuracy, which is why
selection was done on mix error.

---

# Part 2 — How the evaluation numbers were produced

Each question below is answered twice: one sentence for the general reader, then
the technical detail beneath it.

## Why can't you just test the model on the texts you trained it on?

> **Short answer:** because it would score well by memorising, and we need to know
> how it does on a book it has never read.

The unit that matters is the **book**, not the sentence. The deployment scenario is
"hand the model a new book and trust its proportions", so the evaluation has to
mirror that.

We use **leave-one-out by work**. The 16 books are split into 6 folds
([`evaluation/folds.json`](evaluation/folds.json)); for each fold, a fresh gate and
stage-2 model are trained on the ~13 books *outside* that fold and then run on the
books inside it. Every book therefore receives a prediction from a model that never
saw a single sentence of it — 16 out-of-sample books from only 12 trainings.

The folds are balanced on row count, and the five divine-teleology-rich books are
deliberately spread across five different folds so that no fold's training set is
starved of the rarest category.

```bash
bash scripts/submit_folds.sh          # 12 jobs, one parallel burst
python3 scripts/validate_folds.py     # confirm each run's config before using it
```

## How do you turn predictions into a confidence claim?

> **Short answer:** we compare the model's category mix to the true mix for each of
> the 16 books, and the spread of those 16 errors *is* the error bar.

[`evaluation/run_cascade_folds.py`](evaluation/run_cascade_folds.py) runs each
fold's two models over its held-out books and writes a single file,
`evaluation/results/proportions/per_row_predictions.csv`, with one row per sentence:

```
work, year, true_tag, gate_prob_nonjunk, gate_pred, s2_pred, s2_p_*
```

**Everything downstream is arithmetic on that one file** — no model ever runs again.
It is committed, which is why the claims can be reproduced without a GPU.

[`evaluation/evaluate_proportions.py`](evaluation/evaluate_proportions.py) then, for
each of the 16 books:

- **true mix** = counts of `true_tag` over non-junk rows, normalised
- **predicted mix** = counts of `s2_pred` over rows the gate kept, normalised
- **signed error** = predicted − true, per category

That yields a **16 × 3 matrix of signed errors**. Every headline claim is a summary
of that matrix:

| Claim | How it is computed |
|---|---|
| Bias (+0.3 / −1.5 / +1.2) | mean of each column |
| ±90% band | 1.645 × the column's standard deviation |
| "14 of 16 books" | per book, does the predicted category order match the true order |
| "90% of pairs" | all 120 book-pairs × 3 categories (minus ties = 354): does the sign of the true difference match the sign of the predicted difference, binned by the size of the true gap |
| 95% CI on that | resample the **16 books** with replacement 4,000×, recompute the whole rate each time |

Note the last row. The 354 comparisons are **not independent** — they come from 16
books, each appearing in 15 pairs. A textbook interval that assumes independence
gives 87–93%, which is too narrow; resampling books gives **83.5–95.8%**, and that
is the figure quoted.

## Which stage is responsible for the remaining error?

> **Short answer:** both, about equally — and that is a change from earlier, when
> the junk gate was clearly the weak link.

Stage 2 is deliberately run on **every** held-out row, not just the rows the gate
kept. That costs a little inference and buys three counterfactuals from one pass:

| Variant | Meaning | Mean error |
|---|---|---|
| Perfect gate | true non-junk rows → stage 2's label | 5.3 pts |
| Perfect stage 2 | gate decides; survivors credited their true label | 4.9 pts |
| **End-to-end** | both real | **7.1 pts** |

They partly cancel (7.1 < 5.3 + 4.9). The practical conclusion: **the gate is no
longer where the headroom is.** Threshold calibration was tested and bought
+0.15 points (nothing); a 2-seed gate ensemble improved gate-internal metrics but
left the end-to-end result flat. Both negative results are documented rather than
buried, in [`docs/PROPORTION_EVAL_RESULTS.md`](docs/PROPORTION_EVAL_RESULTS.md).

## How do I know the scoring code itself is right?

> **Short answer:** it is tested, and the tests run in a tenth of a second.

```bash
python3 -m pytest tests/ -q      # 16 tests
```

[`tests/test_evaluate_proportions.py`](tests/test_evaluate_proportions.py) covers
the mix arithmetic, the three stage-attribution variants, the ranking logic, the
gap bucketing, and — importantly — that the cluster bootstrap cannot silently
regress to a narrower interval than the one it replaced.

---

## Reproducing the whole thing

| Step | Command | Needs |
|---|---|---|
| Recompute every published claim | `python3 evaluation/evaluate_proportions.py` | nothing |
| Regenerate the figures | `python3 evaluation/plot_proportions.py` | matplotlib |
| Regenerate predictions | `python3 evaluation/run_cascade_folds.py --text-column text --max-length 640` | GPU + adapters |
| Retrain all 12 fold models | `bash scripts/submit_folds.sh` | SLURM cluster |
| Re-label the corpus from scratch | `python3 labelling/run_full_deployment.py` | OpenAI API key, ~$50 |

---

## Not built yet

**Corpus generation** — the scanning corpus this model is meant to be pointed at.
`corpus/bhl/` holds the first pass at discovering candidate texts in the
Biodiversity Heritage Library. The decade-by-decade sampling design, the download
and cleaning pipeline, and the actual scan are still to come. This section will be
filled in when that lands.

## Status

The model and its evaluation are complete. The known next lever is **stage 2**,
not the gate — the gate has been tuned to a plateau, with the negative results to
prove it. Method and every locked decision:
[`docs/PROPORTION_EVAL_PLAN.md`](docs/PROPORTION_EVAL_PLAN.md).
