#!/usr/bin/env python3
"""Aggregate scan predictions with optional filters, for the analysis pass.

Two filters, both motivated by inspecting what the model actually labelled:

  --exclude-applied   drops husbandry, veterinary, economic-entomology and
                      similar practical works. Their prose ("cattle must be
                      young", "by the shoulder the horse does his work") reads
                      as functional to the classifier but is not explanatory
                      claim-making about animals.
  --bucket whole|part restricts to rows that do or do not mention an animal
                      part, using the vocabulary in split_animal_parts.py.
"""

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C = ["divine_teleology", "non_divine_teleology", "internal_essence"]
P = ["1750_1799", "1800_1829", "1830_1849", "1850_1869",
     "1870_1889", "1890_1909", "1910_1929"]

# Deliberately narrow: it must catch practical manuals without swallowing
# legitimate zoology. "economic"/"injurious" appear as BHL subject headings on
# applied works specifically, which is why they are included.
# Matched against the TITLE only, on phrases that mark a practical manual.
# An earlier subject-based version excluded Darwin's "Variation of Animals and
# Plants under Domestication" and Catesby's "Natural History of Carolina",
# which are precisely the works this study needs. Specific phrases, not
# keywords: "cattle" or "fisheries" alone appear in serious zoology.
APPLIED = re.compile(
    r"farm animal|feeds? and feeding|husbandry|veterinar|stock ?rais|"
    r"poultry|dairy|economic (zoology|entomology|ornithology|biology)|"
    r"agricultural (zoology|entomology)|injurious insect|sanitary entomology|"
    r"insect pests?|pests? of|care and management|horse ?manship|"
    r"breeding and (care|management)|manual of farm|domestic(ated)? (fowl|poultry)|"
    r"cattle (feeding|breeding|raising)|sheep (farming|husbandry)|bee ?keep",
    re.I,
)

TOKEN = re.compile(r"[A-Za-z]+")


def parts_vocab() -> set[str]:
    src = (ROOT / "evaluation/split_animal_parts.py").read_text()
    return set(src.split('PARTS = """')[1].split('"""')[0].split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-applied", action="store_true")
    ap.add_argument("--bucket", choices=["all", "whole", "part"], default="all")
    ap.add_argument("--by-subfield", action="store_true")
    args = ap.parse_args()

    pool = {}
    for r in csv.DictReader((ROOT / "corpus/bhl/derived/scan_pool.csv").open()):
        pool.setdefault(r["work_id"], (r["subfield"], r["title"] + " " + r["subjects"]))
    parts = parts_vocab()

    agg = defaultdict(lambda: defaultdict(int))
    excluded_titles = set()
    for pp in sorted(glob.glob(str(ROOT / "data/scan/predictions/*.csv"))):
        name = Path(pp).name
        sp = ROOT / "data/scan/sentences" / name
        if not sp.exists():
            continue
        srows = list(csv.DictReader(sp.open()))
        prows = list(csv.DictReader(open(pp)))
        if not srows or len(srows) != len(prows):
            continue
        subfield, text = pool.get(srows[0]["work_id"], ("(IA supplement)", srows[0]["title"]))
        if args.exclude_applied and APPLIED.search(srows[0]["title"]):
            excluded_titles.add(srows[0]["title"][:70])
            continue
        for s, p in zip(srows, prows):
            if p["gate_pred"] == "junk" or p["s2_pred"] not in C:
                continue
            if args.bucket != "all":
                is_part = any(t.lower() in parts for t in TOKEN.findall(s["text"]))
                if (args.bucket == "part") != is_part:
                    continue
            key = (s["period_bin"], subfield if args.by_subfield else "ALL")
            agg[key]["n"] += 1
            agg[key][p["s2_pred"]] += 1

    if args.exclude_applied:
        print(f"excluded {len(excluded_titles)} applied works\n")
    groups = sorted({g for _, g in agg}, key=lambda g: -sum(agg[(p, g)]["n"] for p in P))
    for g in groups:
        if args.by_subfield:
            print(f"\n===== {g} =====")
        print(f"{'period':<12}{'rows':>10}{'DT':>7}{'NDT':>7}{'IE':>7}")
        for p in P:
            a = agg.get((p, g))
            if not a or not a["n"]:
                continue
            t = a["n"]
            print(f"{p:<12}{t:>10,}" + "".join(f"{100*a[c]/t:>7.1f}" for c in C))


if __name__ == "__main__":
    main()
