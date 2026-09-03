#!/usr/bin/env python3
"""Turn cleaned scanning-corpus volumes into one sentence CSV per volume.

Reuses the anchor-corpus logic unchanged -- chunk_text from
find_animal_chunks.py and extract_from_chunk from extract_sentences.py -- so
scanned text reaches the model through exactly the path the training data did.

One CSV per volume rather than one giant file: inference over thousands of
volumes will be interrupted, and per-volume files make the run resumable,
parallelisable, and traceable back to a source book.
"""

import argparse
import csv
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "corpus"))

from find_animal_chunks import chunk_text  # noqa: E402
from extract_sentences import extract_from_chunk  # noqa: E402
from config.config import ANIMAL_PATTERN  # noqa: E402

CLEAN_DIR = ROOT / "data/texts/scan_clean"
OUT_DIR = ROOT / "data/scan/sentences"
MANIFEST = ROOT / "data/scan/sentence_manifest.csv"

FIELDS = [
    "uid", "work_id", "source", "year", "period_bin", "title",
    "sent_id", "source_chunk", "word_count", "animal_keywords", "text",
]


def load_metadata() -> dict[str, dict]:
    """uid -> the pool metadata we need to carry onto every sentence row."""
    meta: dict[str, dict] = {}
    bhl = ROOT / "corpus/bhl/derived/scan_pool.csv"
    for row in csv.DictReader(bhl.open()):
        meta[f"bhl_{row['item_id']}"] = {
            "work_id": row["work_id"], "source": "BHL", "year": row["year"],
            "period_bin": row["period_bin"], "title": row["title"][:200],
        }
    ia = ROOT / "corpus/ia/derived/ia_supplement_candidates.csv"
    for row in csv.DictReader(ia.open()):
        meta[f"ia_{row['identifier']}"] = {
            "work_id": row["work_key"], "source": "IA", "year": row["year"],
            "period_bin": "1750_1799", "title": row["title"][:200],
        }
    return meta


def process(job: tuple[Path, dict, int]) -> dict:
    path, info, max_rows = job
    text = path.read_text(encoding="utf-8", errors="replace")
    rows, chunks_kept, sent_id = [], 0, 0
    for chunk_index, chunk in enumerate(chunk_text(text)):
        if not ANIMAL_PATTERN.search(chunk):
            continue
        chunks_kept += 1
        for passage, keywords in extract_from_chunk(chunk):
            sent_id += 1
            rows.append({
                "uid": path.stem, **info,
                "sent_id": sent_id,
                "source_chunk": chunk_index,
                "word_count": len(passage.split()),
                "animal_keywords": ";".join(keywords) if isinstance(keywords, (list, tuple, set)) else str(keywords),
                "text": passage,
            })
    # A few volumes yield many thousands of rows and would dominate the
    # inference budget. Sampling rows within a volume is unbiased for that
    # volume's category mix, and the sampling error at this size is far below
    # the model's own per-book error, so the cap costs nothing that matters.
    n_before = len(rows)
    if max_rows and len(rows) > max_rows:
        rows = sorted(random.Random(hash(path.stem) & 0xFFFFFFFF).sample(rows, max_rows),
                      key=lambda r: r["sent_id"])

    if rows:
        out = OUT_DIR / f"{path.stem}.csv"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return {
        "uid": path.stem, "work_id": info["work_id"], "source": info["source"],
        "year": info["year"], "period_bin": info["period_bin"],
        "animal_chunks": chunks_kept, "n_sentences": len(rows),
        "n_sentences_before_cap": n_before,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-quality", type=float, default=0.60)
    ap.add_argument("--max-rows-per-volume", type=int, default=1200,
                    help="0 disables the cap")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()

    usable = None
    manifest = ROOT / "data/texts/scan_clean_manifest.csv"
    if manifest.exists():
        usable = {
            r["uid"] for r in csv.DictReader(manifest.open())
            if float(r["quality_after"]) >= args.min_quality
        }

    jobs = []
    for path in sorted(CLEAN_DIR.glob("*.txt")):
        if usable is not None and path.stem not in usable:
            continue
        if path.stem in meta:
            jobs.append((path, meta[path.stem], args.max_rows_per_volume))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"building sentences for {len(jobs)} volumes", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(process, jobs, chunksize=4))

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["n_sentences"] for r in rows)
    empty = sum(1 for r in rows if r["n_sentences"] == 0)
    print(f"{total:,} sentence rows across {len(rows) - empty} volumes ({empty} empty)")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
