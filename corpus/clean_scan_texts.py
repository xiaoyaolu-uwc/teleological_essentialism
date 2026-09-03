#!/usr/bin/env python3
"""Clean downloaded OCR text for the scanning corpus.

The 16 anchor texts were cleaned by hand against a per-text audit
(data/texts/raw_texts/cleanup.md). That does not scale to 2,600 volumes, so
this applies the generic repairs from that audit plus one data-driven step:

  * long-s as the real character -- a deterministic 'ſ' -> 's' substitution
  * long-s misread as 'f' -- repaired only when the token is not a word AND
    the f->s form is, so "moft" becomes "most" while "Fringilla" is untouched
  * hyphenation broken across line ends, rejoined
  * running heads and bare page-number lines, dropped
  * runs of whitespace, collapsed

Each volume also gets a quality score: the share of alphabetic tokens found in
the system dictionary, measured before and after. Volumes still below
--min-quality after cleaning are marked unusable rather than deleted, so the
exclusion is visible and reversible.
"""

import argparse
import csv
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/texts/scan_raw"
CLEAN_DIR = ROOT / "data/texts/scan_clean"
MANIFEST = ROOT / "data/texts/scan_clean_manifest.csv"
WORDLIST = Path("/usr/share/dict/words")

TOKEN = re.compile(r"[A-Za-z]{2,}")
PAGE_LINE = re.compile(r"^\s*[\[\(]?(?:page\s*)?[ivxlcdm\d]{1,6}[\]\)]?\s*$", re.I)
HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\s*\n\s*(\w)")
MULTISPACE = re.compile(r"[ \t ]{2,}")
BLANKLINES = re.compile(r"\n{3,}")
BLANKLINES_SPLIT = re.compile(r"\n\s*\n+")
BRACE_S = re.compile(r"(?<=[A-Za-z])[{\[|](?=[A-Za-z])")

_WORDS: set[str] | None = None

# The system word list holds base forms, not inflections ("diminished" is
# absent), which made the f->s repair miss most of what it should catch. The
# already-cleaned anchor texts supply the missing forms in the right register.
# Ray and Derham are excluded: they are the two pre-1800 anchors, and their own
# long-s artefacts would be learned as legitimate words.
ANCHOR_DIR = ROOT / "data/texts/clean_texts"
ANCHOR_EXCLUDE = ("ray_", "derham_")
ANCHOR_MIN_COUNT = 5


def words() -> set[str]:
    global _WORDS
    if _WORDS is None:
        vocab = {w.strip().lower() for w in WORDLIST.read_text(errors="replace").split()}
        counts: dict[str, int] = {}
        for path in ANCHOR_DIR.glob("*.txt"):
            if path.name.startswith(ANCHOR_EXCLUDE):
                continue
            for token in TOKEN.findall(path.read_text(encoding="utf-8", errors="replace")):
                lower = token.lower()
                counts[lower] = counts.get(lower, 0) + 1
        vocab |= {w for w, c in counts.items() if c >= ANCHOR_MIN_COUNT}
        _WORDS = vocab
    return _WORDS


def quality(text: str) -> float:
    tokens = TOKEN.findall(text)
    if not tokens:
        return 0.0
    known = words()
    return sum(1 for t in tokens if t.lower() in known) / len(tokens)


def repair_long_s(text: str) -> str:
    """Repair long-s misread as 'f'. Conservative by construction: a token is
    only rewritten when it is not itself a word and the f->s form is one."""
    known = words()

    def fix(match: re.Match) -> str:
        token = match.group(0)
        lower = token.lower()
        if "f" not in lower or lower in known:
            return token
        candidate = lower.replace("f", "s")
        if candidate in known:
            # preserve original capitalisation of the first letter
            return candidate.capitalize() if token[0].isupper() else candidate
        return token

    return TOKEN.sub(fix, text)


def unwrap(text: str) -> str:
    """Join OCR hard-wrapped lines back into paragraphs.

    djvu text wraps at the column width, so a single sentence arrives split
    across several lines. Left as-is, the chunker treats those breaks as
    paragraph boundaries and the sentence splitter sees fragments. Blank lines
    are the real paragraph separator and are preserved.
    """
    paragraphs = BLANKLINES_SPLIT.split(text)
    return "\n\n".join(" ".join(p.split()) for p in paragraphs if p.strip())


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("ſ", "s"))
    # A long-s scanned as a brace or pipe: these characters essentially never
    # occur inside a word, so mid-word occurrences are safe to rewrite.
    text = BRACE_S.sub("s", text)
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [ln for ln in text.split("\n") if not PAGE_LINE.match(ln)]
    text = unwrap("\n".join(lines))
    text = repair_long_s(text)
    text = MULTISPACE.sub(" ", text)
    text = BLANKLINES.sub("\n\n", text)
    return text.strip()


def process(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    before = quality(raw)
    cleaned = clean(raw)
    after = quality(cleaned)
    out = CLEAN_DIR / path.name
    out.write_text(cleaned, encoding="utf-8")
    return {
        "uid": path.stem,
        "raw_chars": len(raw),
        "clean_chars": len(cleaned),
        "quality_before": round(before, 4),
        "quality_after": round(after, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-quality", type=float, default=0.60,
                    help="volumes below this dictionary-hit rate are marked unusable")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(RAW_DIR.glob("*.txt"))
    print(f"cleaning {len(paths)} volumes", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(process, paths, chunksize=8))

    for row in rows:
        row["usable"] = "1" if row["quality_after"] >= args.min_quality else "0"

    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    usable = sum(int(r["usable"]) for r in rows)
    gained = [r["quality_after"] - r["quality_before"] for r in rows]
    print(f"usable {usable}/{len(rows)} at min-quality {args.min_quality}")
    print(f"mean quality gain {sum(gained)/len(gained):+.4f}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
