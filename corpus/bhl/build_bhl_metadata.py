#!/usr/bin/env python3
"""Build provisional BHL biology-corpus metadata and strata counts.

This script uses BHL's public S3-hosted TSV exports. It does not download OCR;
it only builds the candidate metadata needed to decide sampling strata.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import ssl
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

try:
    import certifi
except ImportError:  # pragma: no cover - optional local dependency
    certifi = None


BASE_URL = "https://bhl-open-data.s3.amazonaws.com/data"
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
DERIVED_DIR = ROOT / "derived"

TABLES = {
    "title": f"{BASE_URL}/title.txt.gz",
    "item": f"{BASE_URL}/item.txt.gz",
    "subject": f"{BASE_URL}/subject.txt.gz",
    "creator": f"{BASE_URL}/creator.txt.gz",
}

BIOLOGY_TERMS = {
    "biology",
    "zoology",
    "animal",
    "animals",
    "fauna",
    "natural history",
    "comparative anatomy",
    "anatomy",
    "physiology",
    "embryology",
    "morphology",
    "evolution",
    "adaptation",
    "taxonomy",
    "systematics",
    "classification",
    "ornithology",
    "mammalogy",
    "ichthyology",
    "entomology",
    "herpetology",
    "malacology",
    "conchology",
    "paleontology",
    "palaeontology",
}

STRONG_ANIMAL_TERMS = {
    "zoology",
    "animal",
    "animals",
    "fauna",
    "ornithology",
    "mammalogy",
    "ichthyology",
    "entomology",
    "herpetology",
    "malacology",
    "conchology",
    "paleontology",
    "palaeontology",
    "fossil",
    "fossils",
    "insect",
    "insects",
    "bird",
    "birds",
    "fish",
    "fishes",
    "mammal",
    "mammals",
    "reptile",
    "reptiles",
    "amphibian",
    "amphibians",
    "mollusk",
    "mollusca",
    "mollusc",
    "molluscs",
    "crustacea",
    "spider",
    "spiders",
    "vertebrate",
    "vertebrates",
    "invertebrate",
    "invertebrates",
    "coleoptera",
    "lepidoptera",
    "diptera",
    "hymenoptera",
    "aves",
    "reptilia",
    "amphibia",
}

BROAD_ANIMAL_BIOLOGY_TERMS = {
    "comparative anatomy",
    "embryology",
    "morphology",
    "evolution",
    "adaptation",
    "paleontology",
    "palaeontology",
    "fossil",
    "fossils",
}

EXCLUDE_TERMS = {
    "botany",
    "plants",
    "flora",
    "herbarium",
    "agriculture",
    "horticulture",
    "gardening",
}

SUBFIELD_RULES = [
    (
        "taxonomy_systematics",
        [
            "taxonomy",
            "systematics",
            "classification",
            "species",
            "genera",
            "genus",
            "catalogue",
            "catalog",
            "check-list",
            "checklist",
            "monograph of",
            "revision of",
        ],
    ),
    (
        "natural_history_ecology_behavior",
        [
            "natural history",
            "fauna",
            "habits",
            "behavior",
            "behaviour",
            "distribution",
            "ecology",
            "bionomics",
            "field notes",
    "field notebook",
    "field book",
    "field-book",
        ],
    ),
    (
        "comparative_anatomy_morphology",
        [
            "comparative anatomy",
            "anatomy",
            "morphology",
            "osteology",
            "structure",
            "organ",
            "limb",
            "skull",
            "skeleton",
        ],
    ),
    (
        "physiology",
        [
            "physiology",
            "respiration",
            "circulation",
            "digestion",
            "nervous system",
            "function",
        ],
    ),
    (
        "embryology_development",
        [
            "embryology",
            "embryo",
            "development",
            "larva",
            "larvae",
            "metamorphosis",
        ],
    ),
    (
        "evolution_adaptation",
        [
            "evolution",
            "adaptation",
            "selection",
            "descent",
            "origin of species",
            "darwin",
        ],
    ),
    (
        "paleontology",
        [
            "paleontology",
            "palaeontology",
            "fossil",
            "fossils",
            "extinct",
        ],
    ),
]


def download(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "teleological-essentialism-corpus-discovery/0.1"},
    )
    context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(request, timeout=120, context=context) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(destination)


def rows(path: Path) -> Iterable[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def first_present(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value.strip()
    return ""


def year_from_text(*values: str) -> int | None:
    for value in values:
        if not value:
            continue
        match = re.search(r"(16|17|18|19|20)\d{2}", value)
        if match:
            year = int(match.group(0))
            if 1600 <= year <= 2026:
                return year
    return None


def period_for(year: int | None) -> str:
    if year is None:
        return "unknown"
    if year < 1700:
        return "pre_1700"
    if year < 1750:
        return "1700_1749"
    if year < 1800:
        return "1750_1799"
    if year < 1830:
        return "1800_1829"
    if year < 1850:
        return "1830_1849"
    if year < 1900:
        return f"{year // 10 * 10}_{year // 10 * 10 + 9}"
    if year < 1950:
        return f"{year // 10 * 10}_{year // 10 * 10 + 9}"
    return "1950_plus"


ARCHIVAL_TERMS = [
    "correspondence",
    "letters from",
    "letters to",
    "letters of",
    "field notes",
    "field notebook",
    "field book",
    "field-book",
    "diary of",
    "diaries",
    "notebooks",
    "papers of",
    "scrapbook",
]

REPORT_TERMS = [
    "annual report",
    "report of",
    "report on",
    "report upon",
    "reports of",
    "reports on",
]

# Cheap function-word test. BHL's LanguageCode is per-title and unreliable:
# French and German titles appear tagged ENG.
NON_ENGLISH_MARKERS = [
    " des ", " du ", " sur ", " dans ", " et ", " les ", " une ", " methodique",
    " der ", " die ", " und ", " ueber ", " zur ", " naturgeschichte",
    " di ", " della ", " nella ", " sobre ", " historia ",
]


def looks_non_english(title: str) -> bool:
    padded = f" {title.lower()} "
    return any(marker in padded for marker in NON_ENGLISH_MARKERS)


def genre_for(title_text: str, item_text: str, subjects_text: str = "") -> str:
    subjects_lower = subjects_text.lower()
    # Checked before anything else: a "Periodicals" subject heading marks a
    # serial run, whatever the title looks like.
    if "periodicals" in subjects_lower:
        return "serial_or_periodical"
    if "juvenile" in subjects_lower:
        return "juvenile"
    text = f"{title_text} {item_text}".lower()
    # Checked first: archival material and institutional reports are catalogued
    # as ordinary titles in BHL but are not authored scientific prose.
    if any(term in text for term in ARCHIVAL_TERMS + REPORT_TERMS):
        return "archival_or_report"
    if any(term in text for term in ["journal", "proceedings", "transactions", "annals", "bulletin", "memoirs"]):
        return "journal_or_proceedings_volume"
    if any(term in text for term in ["manual", "handbook", "text-book", "textbook", "introduction to"]):
        return "textbook_manual"
    if any(term in text for term in ["catalogue", "catalog", "check-list", "checklist", "list of"]):
        return "catalogue_list"
    if any(term in text for term in ["popular", "essay", "lectures", "addresses"]):
        return "popular_or_essay"
    return "monograph_or_book"


def subfield_for(text: str) -> str:
    lower = text.lower()
    scores: list[tuple[int, str]] = []
    for subfield, terms in SUBFIELD_RULES:
        score = sum(1 for term in terms if term in lower)
        if score:
            scores.append((score, subfield))
    if not scores:
        return "general_or_unknown"
    scores.sort(reverse=True)
    return scores[0][1]


def is_likely_biology(title: str, subjects: list[str]) -> tuple[bool, str]:
    text = f"{title} {' '.join(subjects)}".lower()
    include_hits = sorted(term for term in BIOLOGY_TERMS if term in text)
    exclude_hits = sorted(term for term in EXCLUDE_TERMS if term in text)

    # Require a positive animal-biology signal. "Natural history" by itself is
    # too broad in BHL metadata and often comes from institution names.
    strong_animal_signal = any(term in text for term in STRONG_ANIMAL_TERMS)
    broad_signal = any(term in text for term in BROAD_ANIMAL_BIOLOGY_TERMS)
    if include_hits and (strong_animal_signal or (broad_signal and not exclude_hits)):
        return True, ";".join(include_hits)
    return False, ";".join(include_hits)


def read_subjects(path: Path) -> dict[str, list[str]]:
    by_title: dict[str, list[str]] = defaultdict(list)
    for row in rows(path):
        title_id = first_present(row, ["TitleID", "titleid"])
        subject = first_present(row, ["SubjectText", "Subject", "subject"])
        if title_id and subject:
            by_title[title_id].append(subject)
    return by_title


def read_creators(path: Path) -> dict[str, list[str]]:
    by_title: dict[str, list[str]] = defaultdict(list)
    for row in rows(path):
        title_id = first_present(row, ["TitleID", "titleid"])
        name = first_present(row, ["FullName", "Name", "CreatorName", "creator"])
        if title_id and name:
            by_title[title_id].append(name)
    return by_title


def read_titles(path: Path) -> dict[str, dict[str, str]]:
    title_rows: dict[str, dict[str, str]] = {}
    for row in rows(path):
        title_id = first_present(row, ["TitleID", "titleid"])
        if title_id:
            title_rows[title_id] = row
    return title_rows


def build_candidates() -> list[dict[str, str]]:
    subjects_by_title = read_subjects(RAW_DIR / "subject.txt.gz")
    creators_by_title = read_creators(RAW_DIR / "creator.txt.gz")
    titles = read_titles(RAW_DIR / "title.txt.gz")

    candidates: list[dict[str, str]] = []
    seen_items: set[str] = set()

    for item in rows(RAW_DIR / "item.txt.gz"):
        item_id = first_present(item, ["ItemID", "itemid"])
        title_id = first_present(item, ["TitleID", "titleid"])
        if not item_id or item_id in seen_items or not title_id:
            continue
        seen_items.add(item_id)

        title_row = titles.get(title_id, {})
        short_title = first_present(title_row, ["ShortTitle", "FullTitle", "Title", "SortTitle"])
        full_title = first_present(title_row, ["FullTitle", "Title", "ShortTitle", "SortTitle"])
        title = full_title or short_title
        subjects = subjects_by_title.get(title_id, [])
        keep, biology_hits = is_likely_biology(title, subjects)
        if not keep:
            continue

        year = year_from_text(
            first_present(item, ["Year"]),
            first_present(item, ["VolumeInfo", "Volume"]),
            first_present(title_row, ["PublicationDate", "StartYear", "EndYear"]),
            title,
        )
        item_text = " ".join(
            [
                first_present(item, ["VolumeInfo", "Volume"]),
                first_present(item, ["CallNumber"]),
                first_present(item, ["InstitutionName", "HoldingInstitution"]),
            ]
        )
        classification_text = " ".join([title, " ".join(subjects), item_text])

        candidates.append(
            {
                "source": "BHL",
                "item_id": item_id,
                "title_id": title_id,
                "title": title,
                "creators": "; ".join(creators_by_title.get(title_id, [])),
                "language": first_present(title_row, ["LanguageCode"]),
                "year": "" if year is None else str(year),
                "period": period_for(year),
                "genre": genre_for(title, item_text, " ".join(subjects)),
                "non_english_title": "1" if looks_non_english(title) else "0",
                "subfield": subfield_for(classification_text),
                "biology_hits": biology_hits,
                "subjects": "; ".join(subjects),
                "volume_info": first_present(item, ["VolumeInfo", "Volume"]),
                "barcode": first_present(item, ["BarCode", "Barcode"]),
                "item_url": first_present(item, ["ItemURL", "ItemUrl"]),
                "item_text_url": first_present(item, ["ItemTextURL", "ItemTextUrl"]),
                "copyright_status": first_present(item, ["CopyrightStatus"]),
                "rights_statement": first_present(item, ["RightsStatement"]),
                "license_type": first_present(item, ["LicenseType"]),
                "rights_holder": first_present(item, ["RightsHolder"]),
            }
        )
    return candidates


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_count_csv(path: Path, records: list[dict[str, str]], fields: list[str]) -> None:
    counts = Counter(tuple(record[field] for field in fields) for record in records)
    rows_out = [
        {**{field: key[i] for i, field in enumerate(fields)}, "count": str(count)}
        for key, count in sorted(counts.items())
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["count"])
        writer.writeheader()
        writer.writerows(rows_out)


def is_public_domain(record: dict[str, str]) -> bool:
    rights_text = " ".join(
        [
            record.get("copyright_status", ""),
            record.get("rights_statement", ""),
            record.get("license_type", ""),
        ]
    ).lower()
    return "public domain" in rights_text


def write_summary(path: Path, records: list[dict[str, str]]) -> None:
    periods = Counter(record["period"] for record in records)
    genres = Counter(record["genre"] for record in records)
    subfields = Counter(record["subfield"] for record in records)
    languages = Counter(record["language"] or "unknown" for record in records)
    dated = [int(record["year"]) for record in records if record["year"]]
    english_records = [record for record in records if record["language"] == "ENG"]
    english_public_domain = [record for record in english_records if is_public_domain(record)]

    def table(counter: Counter[str]) -> str:
        lines = ["| bucket | count |", "|---|---:|"]
        for bucket, count in sorted(counter.items()):
            lines.append(f"| {bucket} | {count} |")
        return "\n".join(lines)

    text = [
        "# BHL Biology Metadata Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Candidate item rows: {len(records):,}",
        f"English candidate item rows: {len(english_records):,}",
        f"English public-domain candidate item rows: {len(english_public_domain):,}",
        "",
    ]
    if dated:
        text.extend(
            [
                f"Year range: {min(dated)}-{max(dated)}",
                "",
            ]
        )
    text.extend(
        [
            "## Period Counts",
            "",
            table(periods),
            "",
            "## Genre Counts",
            "",
            table(genres),
            "",
            "## Subfield Counts",
            "",
            table(subfields),
            "",
            "## Language Counts",
            "",
            table(Counter(dict(languages.most_common(20)))),
            "",
            "## Notes",
            "",
            "- These are provisional metadata-derived labels, not final scholarly classifications.",
            "- `journal_or_proceedings_volume` means a bound volume, not an article-level unit.",
            "- Subfield labels are keyword heuristics from title, subject, and item metadata.",
            "- The next methodological step is to inspect sparse or overlarge buckets and revise the strata.",
        ]
    )
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true", help="redownload BHL TSV files")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in TABLES.items():
        destination = RAW_DIR / f"{name}.txt.gz"
        print(f"Ensuring {destination}")
        download(url, destination, force=args.force_download)

    print("Building candidate metadata")
    candidates = build_candidates()
    write_csv(DERIVED_DIR / "bhl_biology_candidates.csv", candidates)
    english_public_domain = [
        record for record in candidates if record["language"] == "ENG" and is_public_domain(record)
    ]
    write_csv(DERIVED_DIR / "bhl_english_public_domain_candidates.csv", english_public_domain)
    write_count_csv(DERIVED_DIR / "strata_counts_period_genre.csv", candidates, ["period", "genre"])
    write_count_csv(DERIVED_DIR / "strata_counts_period_subfield.csv", candidates, ["period", "subfield"])
    write_count_csv(
        DERIVED_DIR / "strata_counts_period_genre_subfield.csv",
        candidates,
        ["period", "genre", "subfield"],
    )
    write_count_csv(
        DERIVED_DIR / "english_public_domain_counts_period_genre.csv",
        english_public_domain,
        ["period", "genre"],
    )
    write_count_csv(
        DERIVED_DIR / "english_public_domain_counts_period_subfield.csv",
        english_public_domain,
        ["period", "subfield"],
    )
    write_count_csv(
        DERIVED_DIR / "english_public_domain_counts_period_genre_subfield.csv",
        english_public_domain,
        ["period", "genre", "subfield"],
    )
    write_summary(DERIVED_DIR / "metadata_summary.md", candidates)
    print(f"Wrote {len(candidates):,} candidate rows to {DERIVED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
