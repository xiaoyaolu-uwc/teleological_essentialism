#!/usr/bin/env python3
"""Select the final scanning pool from the BHL public-domain candidates.

build_bhl_metadata.py describes every candidate. This script decides which of
them we actually download and scan, and writes the resulting manifest.

The unit of analysis is the *work*, not the item. BHL lists one item per
physical volume, so a multi-volume work appears many times. We keep every
volume (it is all real text) but stamp each row with `work_id` so that
downstream aggregation can weight by work. Title runs longer than
MAX_VOLUMES_PER_WORK are serial publications rather than authored books, and
are dropped outright.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from title_filters import excluded as title_excluded

DERIVED = Path(__file__).resolve().parent / "derived"
KEEP_GENRES = {"monograph_or_book", "textbook_manual", "popular_or_essay"}
YEAR_MIN, YEAR_MAX = 1750, 1929
MAX_VOLUMES_PER_WORK = 10
PERIOD_BINS = [
    (1750, 1799), (1800, 1829), (1830, 1849), (1850, 1869),
    (1870, 1889), (1890, 1909), (1910, 1929),
]


def bin_for(year: int) -> str:
    for lo, hi in PERIOD_BINS:
        if lo <= year <= hi:
            return f"{lo}_{hi}"
    return "out_of_range"


def main() -> None:
    rows = list(csv.DictReader(open(DERIVED / "bhl_english_public_domain_candidates.csv")))
    steps = [("english public-domain candidates", len(rows))]

    rows = [r for r in rows if r["genre"] in KEEP_GENRES]
    steps.append(("after genre filter", len(rows)))

    rows = [r for r in rows if not title_excluded(r["title"])]
    steps.append(("after shared title exclusions", len(rows)))

    rows = [r for r in rows if r["non_english_title"] == "0"]
    steps.append(("after non-English title check", len(rows)))

    rows = [r for r in rows if r["year"] and YEAR_MIN <= int(r["year"]) <= YEAR_MAX]
    steps.append((f"after {YEAR_MIN}-{YEAR_MAX} window", len(rows)))

    volumes = Counter(r["title_id"] for r in rows)
    rows = [r for r in rows if volumes[r["title_id"]] <= MAX_VOLUMES_PER_WORK]
    steps.append((f"after dropping runs over {MAX_VOLUMES_PER_WORK} volumes", len(rows)))

    for r in rows:
        r["work_id"] = r["title_id"]
        r["period_bin"] = bin_for(int(r["year"]))

    out = DERIVED / "scan_pool.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    works = defaultdict(set)
    items = Counter()
    for r in rows:
        works[r["period_bin"]].add(r["work_id"])
        items[r["period_bin"]] += 1

    with (DERIVED / "scan_pool_counts.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["period_bin", "works", "items"])
        for lo, hi in PERIOD_BINS:
            key = f"{lo}_{hi}"
            w.writerow([key, len(works[key]), items[key]])

    for label, n in steps:
        print(f"{label:<44}{n:>7,}")
    print(f"\nwrote {out}")
    print(f"\n{'period':<14}{'works':>8}{'items':>8}")
    for lo, hi in PERIOD_BINS:
        key = f"{lo}_{hi}"
        print(f"{key:<14}{len(works[key]):>8}{items[key]:>8}")
    print(f"{'TOTAL':<14}{sum(len(v) for v in works.values()):>8}{sum(items.values()):>8}")


if __name__ == "__main__":
    main()
