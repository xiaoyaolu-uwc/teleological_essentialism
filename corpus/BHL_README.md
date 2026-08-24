# Corpus Discovery

This directory is for building the large historical biology scanning corpus.

The first target source is the Biodiversity Heritage Library (BHL), because it
has broad historical zoology/natural-history coverage and downloadable metadata.

## Layout

```text
corpus_discovery/
  bhl/
    raw/       downloaded BHL metadata tables (`.txt.gz`)
    derived/   filtered candidate metadata and strata summaries
    build_bhl_metadata.py
```

## First pass

```bash
python3 corpus_discovery/bhl/build_bhl_metadata.py
```

The script downloads BHL TSV metadata tables if they are missing, filters for
likely biology/zoology candidates, assigns provisional strata, and writes:

- `corpus_discovery/bhl/derived/bhl_biology_candidates.csv`
- `corpus_discovery/bhl/derived/strata_counts_period_genre.csv`
- `corpus_discovery/bhl/derived/strata_counts_period_subfield.csv`
- `corpus_discovery/bhl/derived/strata_counts_period_genre_subfield.csv`
- `corpus_discovery/bhl/derived/metadata_summary.md`

The strata labels are deliberately provisional. The goal of this pass is to see
how many candidate texts exist in each bucket before deciding the final sampling
design.
