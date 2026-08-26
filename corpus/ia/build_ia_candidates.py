#!/usr/bin/env python3
"""Discover Internet Archive texts to supplement the thin 1750-1799 BHL bin.

BHL yields only 51 distinct works for 1750-1799, too few to characterise the
period that carries the divine-teleology peak. IA holds far more 18th-century
English natural history, so this script queries it, applies the same kind of
filtering used on BHL, and removes anything we already have from BHL.

Dedup note: IA metadata does not reliably distinguish a later *edition* of a
work from a later *volume* of it, so we collapse on normalised title+creator
and keep one representative. That undercounts available text and never
double-weights a work, which is the safe direction for proportion estimates.
"""

import argparse
import csv
import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from title_filters import excluded as title_excluded

ROOT = Path(__file__).resolve().parent
DERIVED = ROOT / "derived"
BHL_POOL = ROOT.parent / "bhl" / "derived" / "scan_pool.csv"
CTX = ssl.create_default_context(cafile=certifi.where())
UA = {"User-Agent": "Mozilla/5.0 (Stanford teleology corpus research)"}

ANIMAL_TITLE_TERMS = (
    '("natural history" OR zoology OR birds OR insects OR quadrupeds OR fishes '
    'OR animals OR entomology OR ornithology OR conchology OR "comparative anatomy")'
)

# Titles that match an animal term but are not animal-biology prose.
TITLE_EXCLUDE = [
    "gardening", "planting", "husbandry", "agriculture", "botany", "flora",
    "herbal", "garden", "vegetable kingdom", "cookery", "farrier",
    "catalogue", "sale", "auction", "sermon", "poem", "poems",
]

NON_ENGLISH_MARKERS = [
    " des ", " du ", " sur ", " dans ", " les ", " une ", " der ", " die ",
    " und ", " zur ", " naturgeschichte", " della ", " historia ",
]

ARTICLES = re.compile(r"^(the|a|an) ")


def search(query: str, rows: int = 100, max_pages: int = 20) -> list[dict]:
    docs: list[dict] = []
    for page in range(1, max_pages + 1):
        params = {
            "q": query, "rows": rows, "page": page, "output": "json",
            "sort[]": "year asc",
            "fl[]": ["identifier", "title", "creator", "year", "collection"],
        }
        url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params, doseq=True)
        batch = json.load(
            urllib.request.urlopen(urllib.request.Request(url, headers=UA), context=CTX, timeout=90)
        )["response"]["docs"]
        if not batch:
            break
        docs.extend(batch)
    return docs


def norm_title(title: str) -> str:
    title = re.sub(r"\[.*?\]", " ", title.lower())
    title = re.sub(r"[^a-z0-9 ]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = ARTICLES.sub("", title)
    return " ".join(title.split()[:8])


def surname(creator: str) -> str:
    if not creator:
        return ""
    return re.sub(r"[^a-z]", "", creator.split(",")[0].lower())


def work_key(title: str, creator: str) -> str:
    return f"{surname(creator)}|{norm_title(title)}"


def excluded(title: str) -> bool:
    lower = f" {title.lower()} "
    return (
        any(term in lower for term in TITLE_EXCLUDE)
        or any(marker in lower for marker in NON_ENGLISH_MARKERS)
        or title_excluded(title)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-min", type=int, default=1750)
    ap.add_argument("--year-max", type=int, default=1799)
    args = ap.parse_args()

    DERIVED.mkdir(parents=True, exist_ok=True)
    query = (
        f"mediatype:texts AND year:[{args.year_min} TO {args.year_max}] "
        f"AND language:(English) AND title:{ANIMAL_TITLE_TERMS}"
    )
    docs = search(query)
    steps = [("IA search hits", len(docs))]

    def first(value):
        return value[0] if isinstance(value, list) else (value or "")

    records = [
        {
            "source": "IA",
            "identifier": d["identifier"],
            "title": first(d.get("title")),
            "creators": first(d.get("creator")),
            "year": str(first(d.get("year")) or ""),
            "collections": ";".join(d.get("collection") or []),
            "item_text_url": f"https://archive.org/download/{d['identifier']}/{d['identifier']}_djvu.txt",
        }
        for d in docs
    ]

    records = [r for r in records if not excluded(r["title"])]
    steps.append(("after title exclusions", len(records)))

    seen: set[str] = set()
    deduped = []
    for r in sorted(records, key=lambda x: x["year"]):
        key = work_key(r["title"], r["creators"])
        if key in seen:
            continue
        seen.add(key)
        r["work_key"] = key
        deduped.append(r)
    steps.append(("after collapsing to distinct works", len(deduped)))

    bhl_keys = {
        work_key(row["title"], row["creators"])
        for row in csv.DictReader(BHL_POOL.open())
    }
    final = [r for r in deduped if r["work_key"] not in bhl_keys]
    steps.append(("after removing BHL overlap", len(final)))

    out = DERIVED / "ia_supplement_candidates.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(final[0].keys()))
        w.writeheader()
        w.writerows(final)

    for label, n in steps:
        print(f"{label:<40}{n:>7,}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
