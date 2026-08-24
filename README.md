# Teleological Essentialism

A pipeline that reads historical scientific texts and measures **how they
explain animals** — by divine purpose, by naturalized function, or by internal
structure — and tracks how that mix shifts across the 18th and 19th centuries.

Research context, taxonomy, and timeline: [`PROJECT_DESC.md`](PROJECT_DESC.md).
What the model can and cannot be trusted to do:
[`docs/PROPORTION_EVAL_RESULTS.md`](docs/PROPORTION_EVAL_RESULTS.md).

---

## The pipeline

```
        texts on disk
             │
   ┌─────────▼─────────┐
   │ 1. corpus/        │  chunk texts, keep animal-relevant passages,
   │                   │  split into sentence-level rows
   └─────────┬─────────┘
             │  data/sentences.csv
   ┌─────────▼─────────┐
   │ 2. labelling/     │  GPT-5.4 assigns DT / NDT / IE / junk
   │                   │  → the training + evaluation ground truth
   └─────────┬─────────┘
             │  data/sentences_train.csv   (deploy_tag = the label)
   ┌─────────▼─────────┐
   │ 3. models/lora/   │  fine-tune two Qwen3-0.6B LoRA adapters per fold:
   │                   │    junk gate     (junk vs non_junk)
   │                   │    stage 2       (DT vs NDT vs IE)
   └─────────┬─────────┘
             │  models/checkpoints/lora/{gate,s2}_fold*
   ┌─────────▼─────────┐
   │ 4. evaluation/    │  run the cascade on held-out books, then turn the
   │                   │  per-row predictions into proportion + ranking claims
   └───────────────────┘
             │  evaluation/results/proportions/
             ▼
      docs/PROPORTION_EVAL_RESULTS.md
```

**Two stages, because junk is 65% of the corpus.** A single 4-way classifier
lets junk dominate the gradient and smother the distinction that matters. A
gate strips junk first; stage 2 then separates the three real categories on a
near-balanced problem. See
[`docs/history/bert_cascade_evolution.md`](docs/history/bert_cascade_evolution.md).

---

## Layout

| Directory | What lives there |
|---|---|
| `config/` | paths, text metadata, animal/thematic keyword lists |
| `corpus/` | text → passages → sentences; plus BHL scanning-corpus discovery |
| `labelling/` | the LLM labelling pass that produces ground truth |
| `models/` | label spaces, data loading, and the LoRA trainer |
| `evaluation/` | the live evaluation workflow — 5 scripts, nothing else |
| `scripts/` | SLURM wrappers and job-submission helpers |
| `docs/` | current plan + results; `docs/history/` for superseded phases |
| `archive/` | code from finished phases, kept runnable — see `archive/README.md` |
| `data/` | corpus CSVs and cleaned texts |

### `models/`

- `labels.py` — the label spaces. **Deliberately free of torch**, so the whole
  analysis path runs on a laptop with no ML stack installed.
- `data.py` — loading, fold splits, and the scoring functions
  (`category_mix_metrics`, `gate_proportion_metrics`). Also torch-free.
- `torch_utils.py` — device, eval loop, class weights, plain dataset.
- `lora/train.py` — trains any stage (`junk_gate`, `nonjunk_3way`, `full4way`)
  for any fold. One trainer for both cascade stages.

---

## Running it

**Recompute every published claim** from the committed predictions. No GPU, no
torch, a few seconds:

```bash
python3 evaluation/evaluate_proportions.py
```

**Regenerate predictions** from the trained adapters (needs a GPU):

```bash
python3 evaluation/run_cascade_folds.py --text-column text --max-length 640
```

**Train one fold** (SLURM):

```bash
sbatch scripts/train.slurm --run-name s2_fold4 --stage nonjunk_3way --holdout-fold fold4 --prompt-variant S2_structured --lora-r 32 --lora-alpha 64 --epochs 4 --seed 42 --text-column text --max-length 640
```

**Train all six folds**, both stages, as one parallel burst:

```bash
bash scripts/submit_folds.sh
```

---

## Current status

The evaluation is complete. On 16 held-out books there is no meaningful
proportion bias in any category, and category *ordering* is far more reliable
than category *magnitude* — which is what the research questions actually
depend on. Numbers, caveats, and the honest error bars are in
[`docs/PROPORTION_EVAL_RESULTS.md`](docs/PROPORTION_EVAL_RESULTS.md); the
method and every locked decision are in
[`docs/PROPORTION_EVAL_PLAN.md`](docs/PROPORTION_EVAL_PLAN.md).

Known next lever: **stage 2**, not the gate. The gate has been tuned to a
plateau — threshold calibration and seed ensembling were both tested and are
documented as negative results.
