#!/usr/bin/env python3
"""
extract_sentences.py
====================
Reads passages.csv (produced by find_animal_chunks.py) and extracts
focused sentence-level passages, writing them to sentences.csv.

For each chunk, it finds "seed" sentences — those that mention an animal —
then pulls in neighboring sentences only if they contain thematic content
(purpose, function, design, definition, mechanism, etc.), not simply because
they also mention an animal.

Usage:
    python3 extract_sentences.py
    python3 extract_sentences.py --input path/to/passages.csv
    python3 extract_sentences.py --output path/to/sentences.csv
    python3 extract_sentences.py --context-window 1   # tighter context
    python3 extract_sentences.py --dry-run
"""

import os
import re
import csv
import sys
import json
import argparse
from collections import Counter

from corpus_config import ANIMAL_PATTERN, get_animal_keywords, contains_thematic

REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT  = os.path.join(REPO_ROOT, "Data", "reference_texts", "clean_texts", "passages.csv")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "Data", "reference_texts", "sentences.csv")


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations that should not trigger sentence splits
_ABBREV = r'\b(Mr|Mrs|Dr|Prof|Vol|No|Fig|pp|vs|etc|viz|i\.e|e\.g|St|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'

def split_sentences(text):
    """
    Split text into sentences. Protects common abbreviations so they don't
    cause false splits. Returns a list of stripped sentence strings.
    """
    protected = re.sub(_ABBREV + r'\.', r'\1<DOT>', text, flags=re.IGNORECASE)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', protected)
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Sentence-level extraction
# ---------------------------------------------------------------------------

def extract_from_chunk(text, context_window=2):
    """
    Given a single chunk of text, extract focused sentence passages.

    Seeds = sentences that mention an animal keyword.
    Neighbors = sentences within `context_window` positions of a seed.
      - Included only if they contain thematic content.
      - NOT included just because they mention another animal.

    Consecutive included sentences are merged into a single passage.
    Returns a list of (passage_text, animal_keywords) tuples.
    """
    sentences = split_sentences(text)
    n = len(sentences)
    if n == 0:
        return []

    is_seed = [bool(ANIMAL_PATTERN.search(s)) for s in sentences]

    include = [False] * n
    for i, seed in enumerate(is_seed):
        if not seed:
            continue
        include[i] = True
        for delta in range(1, context_window + 1):
            for j in [i - delta, i + delta]:
                if 0 <= j < n and not is_seed[j] and contains_thematic(sentences[j]):
                    include[j] = True

    passages = []
    i = 0
    while i < n:
        if include[i]:
            start = i
            while i < n and include[i]:
                i += 1
            passage_text = ' '.join(sentences[start:i])
            passages.append((passage_text, get_animal_keywords(passage_text)))
        else:
            i += 1

    return passages


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process(input_path, output_path, context_window, dry_run):
    with open(input_path, newline='', encoding='utf-8') as f:
        chunks = list(csv.DictReader(f))

    print(f"Input: {input_path} ({len(chunks)} chunks)")

    records = []
    for row in chunks:
        passages = extract_from_chunk(row['text'], context_window=context_window)
        for passage_text, kws in passages:
            if len(passage_text.split()) < 15:
                continue
            records.append({
                'work':          row['work'],
                'author':        row['author'],
                'year':          row['year'],
                'camp':          row['camp'],
                'camp_label':    row['camp_label'],
                'source_chunk':  row['chunk_number'],
                'word_count':    len(passage_text.split()),
                'animal_keywords': json.dumps(kws),
                'text':          passage_text,
            })

    print(f"\n--- SUMMARY ---")
    print(f"Total sentence passages: {len(records)}")
    if records:
        print(f"Avg word count: {sum(r['word_count'] for r in records) / len(records):.0f}")
        print("\nBy camp:")
        for camp, n in sorted(Counter(r['camp_label'] for r in records).items()):
            print(f"  {camp}: {n}")

    if dry_run:
        return

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'work', 'author', 'year', 'camp', 'camp_label',
            'source_chunk', 'word_count', 'animal_keywords', 'text'
        ])
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote {len(records)} sentences to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract focused sentence passages from passages.csv'
    )
    parser.add_argument('--input',  default=DEFAULT_INPUT,
                        help=f'Input CSV (default: passages.csv)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output CSV (default: sentences.csv)')
    parser.add_argument('--context-window', type=int, default=2,
                        help='Sentences to look left/right of each seed (default: 2)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print stats only, do not write CSV')
    args = parser.parse_args()

    process(args.input, args.output, args.context_window, args.dry_run)


if __name__ == '__main__':
    main()
