#!/usr/bin/env python3
"""
LLM-Assisted Passage Scanner
=============================

Sends passages from sentences.csv to OpenAI in batches, asking the model
to classify each passage by how it conceptualizes animals (teleological,
internal essence, or junk).

Usage:
    # Scan 50 chunks, 5 at a time in parallel:
    python3 scan_passages.py --chunks 50

    # Override parallelism:
    python3 scan_passages.py --chunks 100 --parallel 10

    # Extract non-junk passages to a separate file after scanning:
    python3 scan_passages.py --extract

    # Check progress:
    python3 scan_passages.py --report

    # Dry run — show what would be sent without calling the API:
    python3 scan_passages.py --chunks 1 --dry-run

Environment (.env at repo root):
    OPENAI_API_KEY   — required for scanning
    SCAN_MODEL       — model to use (default: gpt-4o-mini)
    SCAN_BATCH_SIZE  — passages per API call (default: 10)

Outputs:
    - Updates Data/reference_texts/sentences.csv in-place
      (adds scan_complete, scan_tag, scan_reasoning columns)
    - --extract produces Data/promising_passages.csv (non-junk rows only)
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

from corpus_config import PATHS

# ---------------------------------------------------------------------------
# Load .env from repo root
# ---------------------------------------------------------------------------
if PATHS["env_file"].exists():
    with open(PATHS["env_file"]) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PASSAGES_CSV  = PATHS["sentences_csv"]
PROMISING_CSV = PATHS["promising_csv"]

# ---------------------------------------------------------------------------
# Classification taxonomy — sent to the LLM as instructions
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are helping classify historical scientific passages about animals for a \
research project studying how animals have been defined across the history of biology.

Your task: for each passage, decide whether it contains a substantive \
explanation or definition of what an animal (or animal part) IS, decide what \
the underlying way in which they conceptualize an animal is.

Many passages mention animals, but reveal nothing about how the author conceptualizes
the animal. This is the vast majority of content. They should be categorized as junk.

For passages that DO contain substantive explanatory content about animals \
or animal parts, classify them using one of these tags:

  non_divine_teleology — An animal/part is defined by the functions it has / the purpose it serves. \
This seldom arises in its explicit form (e.g. "this animal exists to control pests"), although do \
capture explicit claims that 'animals have purpose X' when they come up. But also accept implicit claims \
which define/categorize animals based on any function/purpose they have. A common one is when \
animals are defined/categorized by interaction modes they have with the environment — \
think something like "bees are defined by their making of honey."

  divine_teleology — Animals/their parts are defined by the purpose they serve, and this purpose \
is *specifically divine*. Look for anything that says animals serve some plan of God.

  internal_essence — Animals/parts are defined by some internal structure they have. Notice the core \
distinction between this and the teleological explanations above — both of those define the animal based \
on some external interaction it has with the environment. This is the opposite: the animal is defined \
by some features or mechanisms it has independently of the external environment.

  junk — Passage mentions animals but contains no real explanatory or \
definitional content. Sentences which mention animals in passing, taxonomic catalogs, publication info, \
tables of contents, geographic descriptions, narrative anecdotes without explanatory \
substance, OCR artifacts.

IMPORTANT GUIDELINES:
- Be fairly generous with what counts as "explanatory content" — a passage \
doesn't need to be a philosophical treatise. If it characterizes what an \
animal or animal part is FOR, what it IS structurally, or HOW it works \
mechanistically, that counts.
- But be strict about junk. A passage that just lists species names, \
describes where an animal lives without explaining why, or discusses \
publishing logistics is junk.
- The "camp" metadata attached to each passage reflects the AUTHOR's overall \
intellectual position, not necessarily what THIS specific passage does. An \
author tagged as "divine_teleology" may have plenty of junk passages or \
even mechanistic ones. Judge each passage on its own content.
- Do NOT try to balance the distribution of tags. Most passages SHOULD be \
junk — probably 70-90% of them. The non-junk tags will be rare. Tag each \
passage independently based on what it actually says, not on what other \
passages in the same batch were tagged.
"""

USER_PROMPT_TEMPLATE = """\
Classify each of the following {n} passages.

You MUST return a JSON array of exactly {n} objects, one per passage, in the \
same order as the passages appear below. Each object MUST have exactly these \
three fields:
  "id"        — the passage id shown in the header (copy it exactly as an integer)
  "tag"       — one of: divine_teleology, non_divine_teleology, internal_essence, junk
  "reasoning" — SHORT (under one sentence). Start with your certainty \
(e.g. "Confident:", "Unsure:") then a quick justification pointing to \
specific evidence in the text.

Example output for 2 passages:
[
  {{"id": 0, "tag": "junk", "reasoning": "Confident: just lists species names and publication dates."}},
  {{"id": 1, "tag": "divine_teleology", "reasoning": "Confident: 'designed by the Creator for the benefit of' directly invokes divine purpose."}}
]

Return ONLY the JSON array. No other text, no wrapper keys.

{passages_block}
"""

PASSAGE_TEMPLATE = """\
--- Passage {id} ---
Author: {author} ({year})
Text:
{text}
"""


# ---------------------------------------------------------------------------
# CSV I/O helpers
# ---------------------------------------------------------------------------
def read_passages(path: Path) -> list[dict]:
    """Read the passages CSV, preserving all fields."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_passages(path: Path, rows: list[dict], fieldnames: list[str]):
    """Write rows back to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# OpenAI API call
# ---------------------------------------------------------------------------
def call_openai(batch: list[dict], model: str = "gpt-4o-mini") -> list[dict]:
    """Send a batch of passages to OpenAI and parse the response.

    Uses the Responses API (/v1/responses) for gpt-5.x models,
    and Chat Completions for older models.

    Returns a list of dicts: [{"id": ..., "tag": ..., "reasoning": ...}, ...]
    """
    import openai

    passages_block = "\n".join(
        PASSAGE_TEMPLATE.format(
            id=p["_batch_id"],
            author=p["author"],
            year=p["year"],
            text=p["text"][:1500],  # truncate to keep token costs predictable
        )
        for p in batch
    )

    user_msg = USER_PROMPT_TEMPLATE.format(n=len(batch), passages_block=passages_block)
    client = openai.OpenAI()

    # Responses API (gpt-5.x and newer)
    if model.startswith("gpt-5"):
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_PROMPT,
                input=user_msg,
            )
        except openai.APIStatusError as e:
            raise RuntimeError(f"Responses API error {e.status_code}: {e.message}")

        raw = None
        for block in response.output:
            if block.type == "message":
                for part in block.content:
                    if part.type == "output_text":
                        raw = part.text
                        break
            if raw:
                break
        if raw is None:
            raise RuntimeError(
                f"Could not find output_text in Responses API reply: {response}"
            )
    else:
        # Chat Completions API (gpt-4o, gpt-4o-mini, etc.)
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={"type": "json_object"},
            )
        except openai.APIStatusError as e:
            raise RuntimeError(f"Chat Completions API error {e.status_code}: {e.message}")
        raw = response.choices[0].message.content.strip()

    # Parse — model should return a JSON array, but might wrap it
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse model response as JSON: {e}\nRaw response:\n{raw}")

    if isinstance(parsed, dict):
        for key in ("results", "passages", "classifications"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]

    return parsed


# ---------------------------------------------------------------------------
# Result matching helper
# ---------------------------------------------------------------------------
def _apply_results(rows, batch_indices, results):
    """Match API results back to rows. Returns count matched."""
    results_by_id = {}
    for r in results:
        if "id" in r:
            results_by_id[str(r["id"])] = r
    can_positional = len(results) == len(batch_indices)

    matched = 0
    for pos, idx in enumerate(batch_indices):
        result = results_by_id.get(str(idx))
        if result is None and can_positional:
            result = results[pos]
        if result:
            tag = result.get("tag", "junk").strip().lower()
            rows[idx]["scan_tag"] = tag
            rows[idx]["scan_reasoning"] = result.get("reasoning", "")
            rows[idx]["scan_complete"] = "TRUE"
            matched += 1
        else:
            rows[idx]["scan_tag"] = "error"
            rows[idx]["scan_reasoning"] = "No result returned by API for this passage"
            rows[idx]["scan_complete"] = "TRUE"
            matched += 1
    return matched


# ---------------------------------------------------------------------------
# Main scanning logic
# ---------------------------------------------------------------------------
def scan(
    rows: list[dict],
    num_chunks: int,
    batch_size: int,
    model: str,
    dry_run: bool,
    parallel: int = 5,
    save_path: Path = None,
    fieldnames: list[str] = None,
) -> int:
    """Scan unscanned passages in batches. Sends `parallel` API calls simultaneously."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unscanned = [i for i, r in enumerate(rows) if r.get("scan_complete", "") != "TRUE"]
    if not unscanned:
        print("All passages have already been scanned.")
        return 0

    total_to_scan = min(num_chunks * batch_size, len(unscanned))
    print(f"Unscanned passages: {len(unscanned)}")
    print(f"Will scan: {total_to_scan} ({num_chunks} chunks × {batch_size} per chunk, {parallel} in parallel)")

    # Build all batches upfront
    all_batches = []
    for chunk_i in range(num_chunks):
        start = chunk_i * batch_size
        if start >= len(unscanned):
            break
        end = min(start + batch_size, len(unscanned))
        batch_indices = unscanned[start:end]
        batch = []
        for idx in batch_indices:
            row_copy = dict(rows[idx])
            row_copy["_batch_id"] = str(idx)
            batch.append(row_copy)
        all_batches.append((chunk_i, batch_indices, batch))

    if dry_run:
        for chunk_i, batch_indices, batch in all_batches:
            print(f"\n[Chunk {chunk_i + 1}] Would send {len(batch)} passages:")
            for p in batch:
                print(f"  Row {p['_batch_id']}: {p['author']} — {p['text'][:80]}...")
        return sum(len(b) for _, _, b in all_batches)

    scanned = 0

    # Process in waves of `parallel` chunks
    for wave_start in range(0, len(all_batches), parallel):
        wave = all_batches[wave_start:wave_start + parallel]
        chunk_range = f"{wave[0][0]+1}–{wave[-1][0]+1}"
        print(f"\n{'—'*50}")
        print(f"Sending chunks {chunk_range} ({len(wave)} in parallel) to {model}...")

        futures = {}
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            for chunk_i, batch_indices, batch in wave:
                future = executor.submit(call_openai, batch, model)
                futures[future] = (chunk_i, batch_indices)

            for future in as_completed(futures):
                chunk_i, batch_indices = futures[future]
                try:
                    results = future.result()
                    matched = _apply_results(rows, batch_indices, results)
                    scanned += matched
                    print(f"  Chunk {chunk_i + 1}: {matched}/{len(batch_indices)} matched")
                except Exception as e:
                    print(f"  Chunk {chunk_i + 1} ERROR: {e}")
                    for idx in batch_indices:
                        rows[idx]["scan_tag"] = "error"
                        rows[idx]["scan_reasoning"] = f"API error: {str(e)[:200]}"
                        rows[idx]["scan_complete"] = "TRUE"
                    scanned += len(batch_indices)

        # Save after each wave
        if save_path and fieldnames:
            for r in rows:
                r.pop("_batch_id", None)
            write_passages(save_path, rows, fieldnames)
            done = sum(1 for r in rows if r.get("scan_complete") == "TRUE")
            print(f"  Saved. Progress: {done}/{len(rows)}")

    for r in rows:
        r.pop("_batch_id", None)
    return scanned


# ---------------------------------------------------------------------------
# Extract mode
# ---------------------------------------------------------------------------
def extract(rows: list[dict], fieldnames: list[str]):
    """Write non-junk rows to promising_passages.csv."""
    promising = [r for r in rows if r.get("scan_tag", "") not in ("junk", "error", "")]
    print(f"Total scanned: {sum(1 for r in rows if r.get('scan_complete') == 'TRUE')}")
    print(f"Promising passages: {len(promising)}")
    write_passages(PROMISING_CSV, promising, fieldnames)
    print(f"Written to: {PROMISING_CSV}")


# ---------------------------------------------------------------------------
# Progress report
# ---------------------------------------------------------------------------
def report(rows: list[dict]):
    """Print a summary of scanning progress."""
    from collections import Counter

    total = len(rows)
    scanned = sum(1 for r in rows if r.get("scan_complete") == "TRUE")
    unscanned = total - scanned
    tags = Counter(r.get("scan_tag", "") for r in rows if r.get("scan_complete") == "TRUE")

    print(f"\n{'='*50}")
    print(f"SCAN PROGRESS: {scanned}/{total} passages scanned ({unscanned} remaining)")
    print(f"{'='*50}")
    if tags:
        print("Tag distribution:")
        for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
            pct = count / scanned * 100
            print(f"  {tag:30s} {count:5d}  ({pct:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="LLM-assisted passage scanner for teleological essentialism project"
    )
    parser.add_argument(
        "--chunks", type=int, default=5,
        help="Number of batches to process in this run (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=int(os.environ.get("SCAN_BATCH_SIZE", 10)),
        help="Passages per API call (default: SCAN_BATCH_SIZE from .env, or 10)"
    )
    parser.add_argument(
        "--parallel", type=int, default=5,
        help="Number of API calls to send simultaneously (default: 5)"
    )
    parser.add_argument(
        "--model", type=str, default=os.environ.get("SCAN_MODEL", "gpt-4o-mini"),
        help="OpenAI model to use (default: SCAN_MODEL from .env, or gpt-4o-mini)"
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Extract non-junk passages to promising_passages.csv"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print scan progress report only"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be sent without calling the API"
    )
    args = parser.parse_args()

    if not PASSAGES_CSV.exists():
        print(f"Error: {PASSAGES_CSV} not found.", file=sys.stderr)
        sys.exit(1)

    rows = read_passages(PASSAGES_CSV)
    original_fieldnames = list(rows[0].keys()) if rows else []

    # Ensure scan columns exist
    scan_fields = ["scan_complete", "scan_tag", "scan_reasoning"]
    fieldnames = list(original_fieldnames)
    for sf in scan_fields:
        if sf not in fieldnames:
            fieldnames.append(sf)
            for r in rows:
                r.setdefault(sf, "")

    if args.report:
        report(rows)
        return

    if args.extract:
        extract(rows, fieldnames)
        return

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        print("Set it with: export OPENAI_API_KEY='sk-...'", file=sys.stderr)
        sys.exit(1)

    scanned = scan(rows, args.chunks, args.batch_size, args.model, args.dry_run,
                   parallel=args.parallel, save_path=PASSAGES_CSV, fieldnames=fieldnames)

    report(rows)


if __name__ == "__main__":
    main()
