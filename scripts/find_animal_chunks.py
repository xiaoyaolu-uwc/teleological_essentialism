#!/usr/bin/env python3
"""
find_animal_chunks.py
=====================
Reads the cleaned reference texts, chunks them into ~300-word paragraph-aware
passages, filters to those that mention animals, and exports to passages.csv.

Usage:
    python3 find_animal_chunks.py
    python3 find_animal_chunks.py --output path/to/passages.csv
    python3 find_animal_chunks.py --dry-run   # print stats only, no CSV
"""

import os
import csv
import sys
import json
import argparse

from corpus_config import TEXT_METADATA, ANIMAL_PATTERN, get_animal_keywords

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR  = os.path.join(REPO_ROOT, "Data", "reference_texts", "clean_texts")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "Data", "reference_texts", "passages.csv")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text, target_words=300):
    """
    Split text into ~target_words passages using paragraph boundaries.
    Keeps the last paragraph of each chunk as overlap into the next.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = []
    current_wc = 0

    for para in paragraphs:
        para_wc = len(para.split())

        if current_wc + para_wc > target_words * 1.3 and current:
            chunks.append('\n\n'.join(current))
            # carry the last paragraph forward as overlap
            current = [current[-1]] if para_wc < target_words else []
            current_wc = len(current[0].split()) if current else 0

        current.append(para)
        current_wc += para_wc

        if para_wc > target_words * 2:
            chunks.append('\n\n'.join(current))
            current = []
            current_wc = 0

    if current:
        tail = '\n\n'.join(current)
        if len(tail.split()) > 50:
            chunks.append(tail)

    return chunks


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_chunks(clean_dir):
    """
    For each clean text, chunk it and keep chunks that mention animals.
    Returns a list of row dicts ready for CSV export.
    """
    records = []
    files = sorted(f for f in os.listdir(clean_dir) if f.endswith('_clean.txt'))

    for fname in files:
        stem = fname.replace('_clean.txt', '')
        if stem not in TEXT_METADATA:
            print(f"  WARNING: no metadata for {stem}, skipping")
            continue

        meta = TEXT_METADATA[stem]
        with open(os.path.join(clean_dir, fname), encoding='utf-8') as f:
            text = f.read()

        chunks = chunk_text(text)
        kept = 0

        for i, chunk in enumerate(chunks):
            if not ANIMAL_PATTERN.search(chunk):
                continue
            records.append({
                'work':          meta['title'],
                'author':        meta['author'],
                'year':          meta['year'],
                'camp':          meta['camp'],
                'camp_label':    meta['camp_label'],
                'chunk_number':  i + 1,
                'total_chunks':  len(chunks),
                'word_count':    len(chunk.split()),
                'animal_keywords': json.dumps(get_animal_keywords(chunk)),
                'text':          chunk,
            })
            kept += 1

        pct = kept / len(chunks) * 100 if chunks else 0
        print(f"  {meta['author']}: {len(chunks)} chunks, {kept} with animals ({pct:.0f}%)")

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Extract animal-relevant chunks from cleaned texts')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output CSV path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print stats only, do not write CSV')
    args = parser.parse_args()

    print(f"Reading from: {CLEAN_DIR}\n")
    records = extract_chunks(CLEAN_DIR)

    from collections import Counter
    print(f"\n--- SUMMARY ---")
    print(f"Total passages: {len(records)}")
    print(f"Avg word count: {sum(r['word_count'] for r in records) / len(records):.0f}")
    print("\nBy camp:")
    for camp, n in sorted(Counter(r['camp_label'] for r in records).items()):
        print(f"  {camp}: {n}")

    if args.dry_run:
        return

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'work', 'author', 'year', 'camp', 'camp_label',
            'chunk_number', 'total_chunks', 'word_count', 'animal_keywords', 'text'
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote {len(records)} passages to: {args.output}")


if __name__ == '__main__':
    main()
