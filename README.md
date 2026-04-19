# Teleological Essentialism — Project README

## The Thesis

People have historically understood animals through their **purpose or function** — "what is this animal *for*?" — rather than through their internal structure or essence. This is **teleological essentialism**. The hypothesis is that the Baconian scientific revolution didn't simply eliminate teleology from biology. Instead, it produced a more complicated landscape where purpose, structure, and mechanism remained in competition, with teleology persisting in transformed forms — moving from overt divine design to naturalized functional explanation. Biology, in particular, never fully gave up purpose-like explanation, even as physics did.

If this is right, then the explanatory categories the lab studies experimentally — purpose, structure/type, and mechanism — are psychologically real enough to organize both contemporary cognition and major episodes in the history of science.

## The Lab

This is **David Rose's lab**. Shaun is also involved. I (**xiaoyao**) am the research assistant building the computational pipeline.

## The Two Studies

### Study 1: The Historical Corpus (my focus)

A computational/historical study using a fine-tuned text classifier (likely BERT) to scan historical scientific texts about animals. Four research questions:

1. **Do explicitly anti-teleological figures still use teleological or purpose-like explanation when talking about animals and animal parts?** A writer may denounce teleology philosophically but still rely on purpose-like description when actually discussing anatomy. Detecting that dissociation between rhetorical stance and explanatory practice is a key goal.
2. **Does biology retain teleological description longer than physics and related domains?**
3. **Does teleology disappear, or does it change form?** Working hypothesis: transformation rather than decline.
4. **Can we track the difference between purpose/function explanations and structure/type explanations across the history of biology?** This connects to the broader project by distinguishing function-based explanation (Camp A) from structural archetype explanation (Camp B, closer to Lockean/Platonic essentialism).

### Study 2: The Behavioral Link (future, not my current focus)

Take cleanly classified passages from Study 1 and test whether modern readers recover the same explanatory distinctions. Possible designs: a classification task (readers label passages by explanatory type), a similarity/clustering task, or a link to the lab's existing yes/no work on whether animals are purpose-bearing. The point is to show the explanatory categories aren't just artifacts of intellectual history but are psychologically real.

### Ideal Outcome

Study 1 shows anti-teleological science produced a complicated landscape where purpose, structure, and mechanism competed, with teleology persisting in transformed form. Study 2 shows those distinctions map onto psychologically real categories modern readers still recognize.

## The Classification Taxonomy

The classifier needs to distinguish passages along these lines at minimum:

| Camp | Label | What it means | Anchor authors |
|------|-------|---------------|----------------|
| A | Divine teleology | Animals/parts exist for God-given purposes; design proves Creator | Paley, Kirby, Ray, Derham |
| A' | Naturalized/evolutionary teleology | Purpose reframed as adaptation, ecological role; includes ecological views | Gray, Agassiz |
| B | Structuralist/archetype | Animals explained by relation to ideal structural plan (closer to Lockean essentialism) | Owen |
| C | Mechanistic/materialist | Animals explained by material causes; no inherent purpose | Huxley, Haeckel, Lamarck |
| Mixed | Boundary cases | Authors whose practice contradicts their rhetoric | Darwin, Mivart |

David asked that under "non-divine teleology" we code for more specific views including ecological explanations — not just a binary divine/non-divine split. He also noted that the structuralist category might be dropped if the taxonomy proves too fine-grained.

The classifier should work on descriptions of **whole animals and animal parts** — not just animals generally. Note: the model may not initially pick up on animal-parts explanations if trained mainly on whole-animal descriptions, so retraining may be necessary.

## My Role and Build Order

### Completed Work

**1. Research the history of science — DONE.** Identified key figures from ~1660 onward, their philosophical commitments, their animal-related publications. 25+ figures across four camps, organized temporally. Output: `teleology_history_catalogue.md`

**2. Build an anchor corpus — DONE.** Downloaded the strongest texts from each camp (Paley, Agassiz, Kirby, Gray as teleological anchors; Haeckel, Huxley, Lamarck as mechanistic; Owen as structuralist; Darwin and Mivart as boundary cases). Current corpus: 14 texts (16 files with multi-volume works), ~14MB, 1691–1876. Downloaded from Internet Archive.

**3. Clean the texts — DONE.** OCR cleanup pipeline handling three quality tiers (near-clean 19th-century texts, double-spaced/hyphenated texts, and pre-1800 texts with long-s typography). ~250 correction patterns. Output: 16 cleaned files in `Data/reference_texts/clean_texts/`

**4. Chunk and detect animal-relevant passages — DONE.** Pipeline chunks texts into ~300-word paragraph-aware segments, filters to passages containing animal references (~170-term keyword system), tags with metadata. Result: **6,136 animal-relevant passages**. Can export to CSV or load into Postgres. Output: `Data/chunk_and_load.py`

### Timeline

Week 1 starts March 30, 2026. "End of Week X" means the task is finished by that Friday.

**End of Week 3 (April 17): Labeled training corpus ready.**
- Build a custom LLM-judge pipeline to label all ~6,700 passages with the nuanced taxonomy: divine teleology, naturalized/evolutionary teleology (including ecological), structuralist/archetype, mechanistic, and none (for passages that mention animals in passing without explanatory content).
- Also produce a `reduced_categories` column collapsing to 4 labels: religious teleology, non-religious teleology, mechanistic, none. This is a trivial Python mapping and serves as a fallback if the nuanced categories don't have enough samples per class for reliable training.
- Consult Stanford history/philosophy of natural science professors to validate the categorizations, confirm what trend we should expect to see, and identify additional figures and texts — especially for the physics comparison corpus.
- Human-validate a sample of the LLM labels.

**End of Week 5 (May 1): BERT model trained and biology corpus ready for analysis.**
- Fine-tune BERT on the labeled passages. Regularly consult data science faculty on model architecture, training strategy, and legal issues around pulling corpus data at scale.
- Assemble the broader biology analysis corpus (beyond the anchor texts) for the model to scan.
- Initial model run: produce first results on the biology corpus.

**End of Week 7 (May 15): Validated results and mechanistic-figure analysis.**
- Validate model results, produce graphical displays of how teleological vs. mechanistic language shifts over time in the biology corpus.
- Build a catalogue of explicitly mechanistic thinkers and run the classifier on their texts to detect where they still describe animals or animal parts teleologically. This directly addresses research question (1).

**End of Week 9 (May 29): Physics comparison corpus assembled.**
- Assemble a comparable physics corpus for training and analysis. This enables the biology-vs-physics comparison for research question (2).

**Weeks 9–10: Extended analysis and write-up.**
- Run the classifier on the physics corpus and compare trajectories.
- Supply cleanly classified passages for potential Study 2 design.

### People to Consult

- **History/philosophy of natural science professors at Stanford**: validate categorizations, expected trends, additional figures and texts (especially physics corpus)
- **Data science faculty**: model training, architecture choices, legal questions about large-scale corpus scanning
- **Stanford library**: 7 texts in the catalogue need institutional access

## Technical Architecture

Everything runs locally on a laptop except model training, which uses a cloud GPU. Total infrastructure cost for the 10-week project: ~$60–75.

### Where things live

**Laptop (primary):** Postgres database, all raw and cleaned text files, the chunked passages, the trained model weights, inference/scanning, analysis, and visualization. The scanning corpus (500–2,000 texts, estimated 500MB–2GB of cleaned text, 50K–200K chunked passages) is small enough to store and process locally. Download scripts should use checkpointing (a manifest tracking each text's archive ID, download status, cleanup status, chunking status) so bulk downloads can resume after interruption.

**Google Colab Pro ($12/month):** GPU runtime for BERT training iterations. BERT-base (110M parameters) fine-tunes in ~20 minutes on an A100. Colab Pro provides 100 compute units/month; an A100 burns ~15 CU/hr, so this budget covers many training iterations. Workflow: push data to Colab, train, pull model weights back to laptop. Google's official Colab MCP Server enables Claude to create notebooks, write code cells, execute training, and retrieve results directly — so the training loop can be orchestrated from a Cowork session.

**LLM API (~$50 credit):** For the passage-labeling pipeline (Weeks 1–3). ~6,700 passages at ~300 words each = 3–5M input tokens per labeling run. Claude Sonnet or GPT-4o-mini both work. Budget covers multiple labeling passes as the taxonomy is refined.

**HathiTrust Data Capsule (free, requires Stanford institutional application):** Only needed if HathiTrust is used for the large-scale scanning corpus. Their Data Capsule is a VM they provide with texts pre-loaded; analysis runs inside it and texts cannot be downloaded out. Application can take time, so apply early if planning to use it.

### What to request from the lab

- **Colab Pro subscription** ($12/month) or confirmation that the lab has a GPU allocation on Stanford's Sherlock cluster.
- **LLM API key** (Anthropic or OpenAI) with ~$50 credit, or confirmation the lab already has one.
- **Stanford library access** for the 7 texts in the catalogue that aren't freely digitized.
- **HathiTrust Data Capsule application** if HathiTrust is the chosen source for the large scanning corpus (requires institutional sponsorship).
- Introduction to a **history/philosophy of science professor** for taxonomy validation and physics corpus sourcing.
- Introduction to a **data science faculty member** for model training guidance and legal questions about corpus scanning.

### The trained model

A fine-tuned BERT-base model is ~440MB. After training on Colab, the model weights are downloaded to the laptop and inference runs locally. Classifying 200K passages takes ~10–20 minutes on CPU (faster on an M-series Mac with MPS acceleration). No server or cloud hosting needed for inference unless the model needs to be shared with other lab members — in which case a free-tier HuggingFace Spaces instance can serve it.

### Software stack

All free and pip-installable: `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, `matplotlib`/`seaborn` for visualization. Postgres for structured passage storage and querying. Python `urllib.request` for bulk downloads from Internet Archive and BHL APIs.

### Corpus storage structure

```
corpus/
├── anchor_texts/          ← 16 training texts (already assembled)
├── biology_scanning/      ← bulk downloaded biology texts for model to scan
├── physics_scanning/      ← bulk downloaded physics texts
└── download_logs/         ← manifest tracking download/cleanup/chunking status per text
```

The anchor texts (~14MB) are the training corpus. The scanning corpus (biology + physics) is assembled in Weeks 5–9 via scripted bulk downloads from Internet Archive, Biodiversity Heritage Library, and potentially HathiTrust. Raw text files live on disk; only the chunked and filtered passages go into Postgres.

### Training data requirements

For the **nuanced taxonomy** (6–8 categories: divine teleology, naturalized/evolutionary teleology, structuralist, mechanistic, mixed, none): aim for 500–1,000 labeled examples per class, so 4,000–8,000 total. The current corpus has 6,136 animal-relevant passages, but distribution is uneven (structuralist_essentialist has only 155). Minority classes may need oversampling or category collapse.

For the **reduced taxonomy** (4 categories: religious teleology, non-religious teleology, mechanistic, none): 300–500 labeled examples per class, so 1,200–2,000 total. Much more achievable with the current corpus. Categories are more linguistically distinct, so the classifier should learn faster.

Recommended approach: label with the nuanced taxonomy first, then produce a `reduced_categories` column as a trivial Python mapping. Train on reduced categories initially; split non-religious teleology later if the model performs well and there's enough data.

## Data Pipeline

The pipeline runs in three sequential steps, each script producing input for the next.

**Step 1 — Chunk and filter** (`Data/find_animal_chunks.py`)
Reads the 16 cleaned reference texts, splits them into ~300-word paragraph-aware chunks, and keeps only chunks that mention animals. Writes `passages.csv` (~6,100 rows).
```bash
python3 Data/find_animal_chunks.py
```

**Step 2 — Extract sentence-level passages** (`Data/extract_sentences.py`)
Reads `passages.csv` and refines it to focused sentence-level passages. For each chunk, finds "seed" sentences (those mentioning an animal) and pulls in neighboring sentences only if they contain thematic content (purpose, function, design, definition, mechanism). Writes `sentences.csv` (~12,900 rows).
```bash
python3 Data/extract_sentences.py
```

**Step 3 — LLM classification scan** (`Data/scan_passages.py`)
Reads `sentences.csv` and sends passages to OpenAI in parallel batches, classifying each as `divine_teleology`, `non_divine_teleology`, `internal_essence`, or `junk`. Writes scan results back into `sentences.csv` in-place (adds `scan_complete`, `scan_tag`, `scan_reasoning` columns). Run multiple times — picks up where it left off.
```bash
python3 Data/scan_passages.py --chunks 100 --parallel 5
python3 Data/scan_passages.py --report        # check progress
python3 Data/scan_passages.py --extract       # write promising_passages.csv
```

Shared configuration (text metadata, animal keyword lists, thematic keyword lists) lives in `Data/corpus_config.py` and is imported by Steps 1 and 2.

Set `OPENAI_API_KEY`, `SCAN_MODEL`, and `SCAN_BATCH_SIZE` in a `.env` file at the repo root (see `.env.example` — never commit `.env` itself).

## Current File Structure

```
teleological_essentialism/
├── README.md                              ← this file
├── teleology_history_catalogue.md         ← historical research on key figures
├── .env                                   ← secrets + config (not committed)
├── .env.example                           ← template for .env
├── requirements.txt
├── scripts/
│   ├── corpus_config.py                   ← shared metadata, keyword lists, regex patterns, PATHS
│   ├── find_animal_chunks.py              ← Step 1: chunk texts → passages.csv
│   ├── extract_sentences.py               ← Step 2: passages.csv → sentences.csv
│   └── scan_passages.py                   ← Step 3: LLM classification of sentences.csv
└── Data/
    ├── passages.csv                       ← Step 1 output (~6,100 chunks)
    ├── sentences.csv                      ← Step 2 output (~12,900 passages, scan results added here)
    ├── promising_passages.csv             ← Step 3 extract output (non-junk only)
    └── texts/
        ├── raw_texts/                     ← 16 original OCR text files + reference_index.md
        └── clean_texts/                   ← 16 cleaned text files + cleanup.md (OCR audit notes)
```

## Technical Notes

- Internet Archive blocks direct `curl` to djvu.txt files. Use Python `urllib.request` with a browser User-Agent header.
- Pre-1800 OCR requires special handling for long-s typography ("firft" → "first"). The cleanup notes are in `Data/reference_texts/clean_texts/cleanup.md` (~250 correction patterns).
- Genre comparability is a known concern: pre-1665 texts are a different genre from post-Philosophical Transactions journal articles. Teleological language may drop partly due to genre change, not just philosophical change.
- `scan_passages.py` routes `gpt-5.x` models to the Responses API (`/v1/responses`) and older models to Chat Completions. Update the routing condition in `call_openai()` if adding a new model family.
