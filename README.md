# Teleological Essentialism

A computational pipeline for classifying how historical scientific texts conceptualize animals — by divine purpose, naturalized function, internal structure, or mechanism. The end goal is a fine-tuned BERT classifier that can scan a large corpus of historical biology texts and track how explanatory style shifts over time.

For full project context, research questions, and timeline, see `PROJECT_DESC.md`.

---

## Repo Structure

```
config/
    config.py               shared paths, text metadata, keyword lists

extraction/
    find_animal_chunks.py   step 1: chunk texts → passages.csv
    extract_sentences.py    step 2: passages.csv → sentences.csv

labelling/
    scan_passages.py        step 3: LLM classification of sentences.csv
    prompts.py              versioned prompt templates

eval/
    evaluate.py             run a model+prompt against the evaluation set
    models/                 model adapters (OpenAI, BERT stub)
    results/                output CSVs: eval_{model}_{prompt_version}.csv

training/                   (future) fine-tune BERT on labelled sentences

data/
    evaluation_set.csv      49 hand-labelled passages, used to score models
    sentences.csv           ~12,900 passages with LLM labels
    passages.csv            ~6,100 raw chunks (step 1 output)
    promising_passages.csv  non-junk subset extracted from sentences.csv
    texts/
        raw_texts/          16 original OCR text files
        clean_texts/        16 cleaned text files
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```
OPENAI_API_KEY=sk-...
SCAN_MODEL=gpt-5.4-mini
SCAN_BATCH_SIZE=10
```

---

## Data Pipeline

Run steps in order. Each step reads the previous step's output.

**Step 1 — Chunk texts into passages**
```bash
python3 extraction/find_animal_chunks.py
```
Reads the 16 cleaned texts in `data/texts/clean_texts/`, splits into ~300-word chunks, keeps only those mentioning animals. Writes `data/passages.csv` (~6,100 rows).

**Step 2 — Extract sentence-level passages**
```bash
python3 extraction/extract_sentences.py
```
Reads `data/passages.csv`, refines each chunk to focused sentence-level passages by pulling seed sentences (animal mentions) plus thematically relevant neighbours. Writes `data/sentences.csv` (~12,900 rows).

**Step 3 — LLM classification**
```bash
python3 labelling/scan_passages.py --chunks 100 --parallel 5
python3 labelling/scan_passages.py --report          # check progress
python3 labelling/scan_passages.py --extract         # write promising_passages.csv
```
Reads `data/sentences.csv`, sends passages to the model in parallel batches, writes `scan_tag` and `scan_reasoning` back into the same file in-place. Re-runnable — picks up where it left off.

---

## Evaluation Harness

Tests a model and prompt version against the 49 hand-labelled passages in `data/evaluation_set.csv`.

**To run:**
```bash
python3 eval/evaluate.py                          # single run
python3 eval/evaluate.py --runs 5                 # batch of 5 runs
python3 eval/evaluate.py --run-dir my-experiment  # isolate output in a subdirectory
```

Use `--run-dir` to keep results from different configurations separate. The subdirectory is created under `eval/results/` if it doesn't exist.

**To switch model or prompt**, edit the two variables at the top of `eval/evaluate.py`:
```python
MODEL          = "gpt-5.4-mini"
PROMPT_VERSION = "v1"
```

**Output:**
- `eval/results/[run-dir/]eval_{model}_{prompt_version}.csv` — one row per passage: text, correct label, your rationale, predicted label, model reasoning
- CLI summary: overall accuracy, per-class precision/recall/F1, confusion matrix
- On `--runs > 1`: cross-run error analysis written to `eval/results/[run-dir/]analysis/error_analysis.csv`

**To analyse errors across runs:**
```bash
python3 eval/analyze_errors.py --run-dir my-experiment   # reads all CSVs in that subdir
python3 eval/analyze_errors.py                           # reads all CSVs in eval/results/
python3 eval/analyze_errors.py eval/results/run-a/*.csv  # explicit files
```
