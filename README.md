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

# Part 3 — The scan

## The corpus

The scanning corpus was assembled from two sources and is fully reproducible
from the scripts in `corpus/`.

**Biodiversity Heritage Library.** `corpus/bhl/build_bhl_metadata.py` downloads
BHL's public metadata tables and filters 324,356 items down to English,
public-domain, animal-biology candidates. `corpus/bhl/build_scan_pool.py` then
selects the works actually scanned: monographs, textbooks and essays only,
1750–1929, dropping serial runs longer than ten volumes.

**Internet Archive.** BHL yields only 44 distinct works for 1750–1799, too few
to characterise the period that carries the divine-teleology peak.
`corpus/ia/build_ia_candidates.py` supplements it with 353 further works,
deduplicated against BHL.

Filtering decisions that materially changed the corpus, all made after
inspecting samples rather than in the abstract:

| Excluded | Why |
|---|---|
| Catalogues, journals, serials | Not authored scientific prose |
| Archival material and reports | 407 of one ornithologist's private letters were catalogued as individual books |
| Periodical-tagged items | 14% of the pool, invisible to the genre field |
| Non-English titles tagged ENG | BHL's language code is per-title and unreliable |
| Multi-volume runs over 10 | One serial would otherwise carry the weight of 80 books |

**Result: 2,599 volumes downloaded, 2,575 usable, 2,088 distinct works.**

## Cleaning

`corpus/clean_scan_texts.py` applies generic OCR repairs at scale, since the
anchor texts' hand-tuned cleanup does not generalise to 2,600 volumes:

- long-s as the character `ſ`, substituted directly
- long-s misread as `f`, repaired only when the token is not a word and the
  `f`→`s` form is — so "moft" becomes "most" while "Fringilla" is untouched
- long-s misread as a brace or pipe mid-word
- hyphenation broken across line ends, rejoined
- OCR hard-wrapping, unwrapped back into paragraphs

The dictionary used for the `f`→`s` test is the system word list plus word forms
harvested from the already-cleaned post-1800 anchor texts, because the system
list holds base forms only and misses most inflections.

Effect is concentrated exactly where it should be. Median dictionary-hit rate:

| Period | Before | After |
|---|---:|---:|
| 1750–1799 | 0.845 | **0.917** |
| 1800–1829 | 0.911 | 0.927 |
| 1870–1889 | 0.940 | 0.945 |
| 1910–1929 | 0.944 | 0.950 |

## Sentences and inference

`corpus/build_scan_sentences.py` reuses the anchor pipeline unchanged —
`chunk_text` and `extract_from_chunk` — so scanned text reaches the model by
exactly the path the training data did. Output is one CSV per volume, which
makes the inference run resumable and traceable back to a source book.

A cap of 1,200 rows per volume trims 2.42M rows to **1,709,417**. Sampling rows
within a book is unbiased for that book's mix, and the resulting sampling error
is far below the model's own per-book error.

`evaluation/run_scan_inference.py` runs the deployment cascade over those files.
It differs from the fold script in three ways, all because this is deployment:
one gate and one stage-2 model trained on all 16 anchor works (`--no-holdout`),
stage 2 run only on gate survivors, and predictions written to their own files
so a retrained model cannot destroy the input.

Practical notes for anyone re-running this on a 16GB card:

- Training needs `--batch-size 4 --grad-accum-steps 4` to reproduce the adopted
  effective batch of 16. There is no LR scheduler and sample weights are
  uniform, so accumulation is equivalent to one large batch.
- `models/lora/model.py` pins fp32 explicitly. transformers 5 loads Qwen3 in
  bf16 by default, which would silently change the numerics of an
  already-validated config; the fold adapters are fp32.
- Inference batches by token budget, not row count. The prompt template is a
  fixed prefix of several hundred tokens and must be counted.

Measured throughput on an A4000: **17.3 rows/s**, about 24 hours for the corpus.

## Results

Figures are pre-rendered in `reports/figures/` and collected in
`reports/teleological_essentialism_figures.pdf`. Regenerate with:

```bash
python3 reports/build_work_bucket_counts.py
python3 reports/make_figures.py
```

Error bars throughout are a cluster bootstrap over **works** within each period,
matching the method used in the proportion evaluation. Works, not sentences, are
the unit of observation.

Headline proportions, all animal sentences:

| Period | Works | DT | NDT | IE |
|---|---:|---:|---:|---:|
| 1750–1799 | 326 | 4.2 | 80.8 | 15.2 |
| 1800–1829 | 125 | 2.2 | 62.1 | 35.7 |
| 1830–1849 | 158 | 1.6 | 51.3 | 47.1 |
| 1850–1869 | 169 | 1.6 | 58.9 | 39.4 |
| 1870–1889 | 349 | 0.6 | 56.0 | 43.4 |
| 1890–1909 | 499 | 0.4 | 61.9 | 37.7 |
| 1910–1929 | 343 | 0.3 | 63.8 | 35.9 |

**Divine teleology declines steadily and the decline survives every cut** we
tried: within each subfield separately, after standardising for subfield
composition, and restricted to whole-animal sentences. It is the one clear
trend in the data.

**NDT and IE are largely subfield properties rather than period properties.**
Ecology sits at 65–81% NDT throughout; palaeontology and embryology at 56–76%
IE. Standardising the subfield mix flattens the pooled NDT and IE lines to
roughly 55–63 and 36–44 across the whole period.

Three cuts of the same data are provided because they answer different
questions:

| File | Contents |
|---|---|
| `data/scan/scan_by_work.csv` | one row per work, raw counts |
| `data/scan/scan_by_period.csv` | aggregated to seven period bins |
| `data/scan/scan_by_subfield_period.csv` | subfield × period |
| `data/scan/scan_by_parts.csv` | period × subfield × whole/part |
| `data/scan/scan_by_work_bucket.csv` | per work, split whole/part — the bootstrap input |

All hold **counts only**. Proportions, rankings and intervals are functions of
those counts, and a stored derived value is one that can go stale.

---

# Part 4 — How far to trust the scan

## The validation

The proportion evaluation in Part 2 was measured on 16 argumentative treatises.
The scanning corpus is mostly descriptive natural history, so those numbers do
not transfer automatically. To check, we sampled **30 sentences per category per
period bin — 630 rows** — and relabelled them with the same GPT-5.4 `d_v3`
prompt, which matched the 49-row hand-labelled set on 45 of 49 sentences
(91.8%; its parent prompt `r6a` averaged 90.6% over five runs).

```bash
python3 evaluation/validate_scan_with_gpt.py --model openai/gpt-5.4 --batch-size 10
```

Agreement with the teacher, by the model's own label:

| Model label | n | Agrees | → junk | → other category |
|---|---:|---:|---:|---:|
| Divine teleology | 210 | 54.3% | 29.0% | 16.7% |
| Non-divine teleology | 210 | 54.3% | 37.1% | 8.6% |
| Internal essence | 210 | 47.1% | 46.2% | 6.7% |

**The accuracy is even across the three categories**, which is the property the
proportions depend on. A classifier that is equally wrong in every category
still recovers the right mix; one that is wrong in a lopsided way does not.

Two things worth reading carefully:

- The dominant error is **gate leakage**, not category confusion. Between 29%
  and 46% of what reaches stage 2 is junk by the teacher's judgement. Confusion
  *among* the three real categories is minor — NDT→DT is zero.
- Per-period agreement wobbles between 27% and 67%, but with 30 rows per cell
  the 95% sampling range is roughly ±18 points, so nearly all of that scatter is
  noise. Divine teleology in the last two bins is the one value that sits near
  the edge, and since it is a decline on a base already below 1%, it moves
  nothing material.

## The open question

Everything above measures agreement with the teacher, not with truth. The
teacher was checked against 49 hand-labelled sentences from the anchor corpus; these sentences are
well outside that distribution. **Whether the teacher is still right here has
not been checked, and that check is the precondition for trusting the trends.**

`data/scan/validation_scored.csv` is built for exactly that review: 630 rows,
each with the sentence, our model's label, the teacher's label, and the
teacher's stated reasoning. Reading it is the next step for anyone using these
results.

Inspection of the sampled sentences shows the failure modes are behavioural
description ("they roost in the reeds"), practical husbandry ("cattle must be
young"), and human-use utility ("useful as domestic animals") being read as
functional explanation. Excluding applied and agricultural works entirely moves
the 1910–1929 NDT figure by only 3.9 points, so it is not the main cause.

## Status

The scan is complete and its outputs are in the repository. What remains is a
judgement call, not a computation: verify the teacher's labels on the 630-row
sample, and decide from there whether the proportions carry the weight the
argument needs.
