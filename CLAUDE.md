# Working conventions

## The scanning corpus is not versioned until a download run finishes

`corpus/download_texts.py` bulk-fetches the BHL/IA scan pool into
`data/texts/scan_raw/`. A full run is **2,665 volumes, roughly 2GB**, and it takes
hours. Three paths are gitignored while that runs:

| Path | Status |
|---|---|
| `data/texts/scan_raw/` | **Never tracked.** Regenerable from `corpus/bhl/derived/scan_pool.csv`. |
| `data/texts/download.log` | Never tracked. Run log. |
| `data/texts/scan_download_status.csv` | **Ignored during a run, tracked at the end.** |

**The rule: do not track the download state until the very end, then push it.**

`scan_download_status.csv` is the resume manifest — it rewrites every few seconds
while workers are fetching, so tracking it mid-run produces constant meaningless
churn. But it is also the record of *which volumes failed and why*, which is corpus
provenance worth keeping. So it stays ignored until the run completes, and then it
gets committed once as a frozen snapshot and pushed.

When a run finishes:

1. Confirm no `download_texts.py` process is alive.
2. Remove the `data/texts/scan_download_status.csv` line from `.gitignore`.
3. `git add -f data/texts/scan_download_status.csv` and commit it as the frozen
   manifest for that run, noting the volume count and failure count in the message.
4. Push. `data/texts/scan_raw/` stays untracked permanently.

**Never run `git add -A` while a download is in progress.** It will try to hash
hundreds of megabytes of raw OCR into `.git/objects`. This has already happened once
(2026-08-26), leaving 59 interrupted `tmp_obj_*` files behind. Stage explicit paths.

## Generated data that *is* tracked

`corpus/bhl/derived/` and `evaluation/results/proportions/per_row_predictions*.csv`
are committed on purpose — every published number must be recomputable without a GPU.
Note the cost: `bhl_biology_candidates.csv` is 23MB and fully rewrites whenever
`build_bhl_metadata.py` changes its columns, so a one-column edit shows up as a
~100k-line diff. That is expected, not a mistake.

Run `git gc` occasionally; the repo tends to accumulate loose objects with no packfile.
