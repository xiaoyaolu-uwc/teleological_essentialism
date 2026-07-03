# Data Directory

Reference for every file in `data/`. Update the changelog at the bottom whenever a file is created, modified, or replaced.

---

## CSVs

### passages.csv
**6,136 rows.** The primary chunked passage database, produced by `extraction/find_animal_chunks.py`. Each row is a ~300-word paragraph-aware chunk from one of the 16 anchor texts, filtered to chunks containing at least one animal-related keyword (~170-term list). This is the upstream source from which `sentences.csv` is derived.

Columns: `work, author, year, camp, camp_label, chunk_number, total_chunks, word_count, animal_keywords, text`

No LLM labels. This file is not directly used for BERT training — `sentences.csv` is.

---

### sentences.csv
**12,913 rows.** Sentence-level passages derived from `passages.csv` by `extraction/extract_sentences.py`, which refines each chunk down to focused sentence-level units. Contains LLM labels from an early scan run (pre-eval-harness, using an older prompt).

Columns: `work, author, year, camp, camp_label, source_chunk, word_count, animal_keywords, text, scan_complete, scan_tag, scan_reasoning`

Current label distribution (from early scan — **not used for training, superseded**):
- junk: 6,697
- internal_essence: 3,253
- non_divine_teleology: 2,051
- divine_teleology: 879
- error / unknown: 30

The `scan_tag` column here reflects an early labeling pass with an older prompt (pre-v7c). These labels are **not the training labels** — see changelog for when a new labeling run produces the authoritative training column.

---

### promising_passages.csv
**6,193 rows.** A filtered view of `sentences.csv` keeping only non-junk predictions from the early scan (`divine_teleology`, `non_divine_teleology`, `internal_essence`, `unknown`). Same schema as `sentences.csv`. Used for manual review and prompt development; not an authoritative training file.

---

### evaluation_set.csv
**49 rows.** The golden hand-labeled evaluation set used by `eval/evaluate.py`. Rows were sampled from `sentences.csv`, manually labeled by xiaoyao with input from David, and frozen. This file should not be modified except to correct genuine labeling errors (document any changes in the changelog below).

Columns extend `sentences.csv` with: `agree with david?, correct_tag, xy_rationale, scan_complete, scan_tag, scan_reasoning, tag_quality, david_rationale`

Correct-tag distribution: junk 19, non_divine_teleology 14, divine_teleology 10, internal_essence 6.

Note: two rows previously had `mixed_dt-ie` / `mixed_ndt-ie` tags; these were collapsed to `divine_teleology` and `non_divine_teleology` respectively in June 2026 (see changelog).

---

## texts/

### texts/raw_texts/
16 raw OCR downloads from Internet Archive. One per anchor author/work (two volumes for Haeckel and Kirby). Plus `cleanup.md` (OCR quality audit, per-text findings, correction patterns) and `reference_index.md` (text catalogue with archive IDs).

### texts/clean_texts/
16 cleaned counterparts, produced by the OCR cleanup pipeline documented in `cleanup.md`. These are the files `find_animal_chunks.py` reads. Do not edit manually — regenerate via the cleanup script if corrections are needed.

**Anchor corpus:** 14 texts / 16 files, ~14MB, spanning 1691–1876.

Authors: Agassiz, Argyll, Cuvier, Darwin, Derham, Gray, Haeckel (×2), Huxley, Kirby (×2), Lamarck, Mivart, Owen, Paley, Ray.

---

## Changelog

Entries are newest-first. Add an entry whenever a file is created, modified, or replaced, with enough detail that the reason and scope of the change is clear to a future reader.

---

### 2026-06-23 — evaluation_set.csv: mixed tags collapsed

Two rows in `evaluation_set.csv` had `correct_tag` values of `mixed_dt-ie` and `mixed_ndt-ie`. These were collapsed to `divine_teleology` and `non_divine_teleology` respectively, to align with the decision to remove mixed tags from `VALID_TAGS_V2` in `models/prompts.py`. The eval harness and all downstream results use the collapsed labels from this point forward.

---

### 2026-06-23 — sentences.csv: re-labeled with prompt r6a (f7b no-overfit variant) [PLANNED — update with actuals after run]

**Prompt:** `r6a` (see `eval/final_prompt_refinement_log.md`). Synthetic examples replacing eval-set quotes; no overfit risk. Accuracy on eval set: 90.6% (5-run avg on gpt-5.4). Selected over f7b (91.0%) to avoid overfitting to the 49-row eval set.

**Model:** gpt-5.4

**Script:** `labelling/scan_passages.py`

**Output:** new `label` column added to `sentences.csv` (or written to a new file — update this entry with the actual output path).

**Label distribution:** _[fill in after run]_

**Purpose:** produce the authoritative training labels for BERT fine-tuning.

---

### ~2026-04 — sentences.csv: initial scan with early prompt

First LLM labeling pass over all 12,913 rows using a pre-v7c prompt (exact version not recorded). Labels written to `scan_tag` / `scan_reasoning` columns. Distribution: junk 6,697 / IE 3,253 / NDT 2,051 / DT 879 / error+unknown 30. These labels were used for prompt development and manual review only, not for BERT training.

---

### ~2026-04 — promising_passages.csv: created as filtered view of sentences.csv

Rows where `scan_tag` ∈ {divine_teleology, non_divine_teleology, internal_essence, unknown} extracted from `sentences.csv`. 6,193 rows. Used for manual review during prompt iteration.

---

### ~2026-03 — passages.csv, sentences.csv: created

`passages.csv` produced by `extraction/find_animal_chunks.py` — 6,136 animal-relevant chunks from 16 clean texts.
`sentences.csv` produced by `extraction/extract_sentences.py` — 12,913 sentence-level passages derived from `passages.csv`.
