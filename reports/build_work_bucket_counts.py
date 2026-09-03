#!/usr/bin/env python3
"""Per-work category counts, split by whole-animal / animal-part.

The bootstrap in make_figures.py resamples WORKS, so it needs counts at work
level. scan_by_parts.csv is already aggregated to period x subfield and cannot
support that, hence this pass over the per-volume prediction files.
"""

import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C = ["divine_teleology", "non_divine_teleology", "internal_essence"]
TOKEN = re.compile(r"[A-Za-z]+")


def parts_vocab() -> set[str]:
    src = (ROOT / "evaluation/split_animal_parts.py").read_text()
    return set(src.split('PARTS = """')[1].split('"""')[0].split())


def main() -> None:
    parts = parts_vocab()
    subfield = {}
    for r in csv.DictReader((ROOT / "corpus/bhl/derived/scan_pool.csv").open()):
        subfield.setdefault(r["work_id"], r["subfield"])

    rows = defaultdict(lambda: defaultdict(int))
    meta = {}
    for pp in sorted(glob.glob(str(ROOT / "data/scan/predictions/*.csv"))):
        name = Path(pp).name
        sp = ROOT / "data/scan/sentences" / name
        if not sp.exists():
            continue
        srows = list(csv.DictReader(sp.open()))
        prows = list(csv.DictReader(open(pp)))
        if not srows or len(srows) != len(prows):
            continue
        wid = srows[0]["work_id"]
        meta[wid] = (srows[0]["period_bin"], subfield.get(wid, "(IA supplement)"))
        for s, p in zip(srows, prows):
            rows[wid]["n_rows"] += 1
            if p["gate_pred"] == "junk" or p["s2_pred"] not in C:
                continue
            bucket = "part" if any(t.lower() in parts for t in TOKEN.findall(s["text"])) else "whole"
            rows[wid]["all_n"] += 1
            rows[wid][f"all_{p['s2_pred']}"] += 1
            rows[wid][f"{bucket}_n"] += 1
            rows[wid][f"{bucket}_{p['s2_pred']}"] += 1

    fields = ["work_id", "period_bin", "subfield", "n_rows"]
    for b in ("all", "whole", "part"):
        fields += [f"{b}_n"] + [f"{b}_{c}" for c in C]
    out = []
    for wid, c in rows.items():
        period, sub = meta[wid]
        out.append({"work_id": wid, "period_bin": period, "subfield": sub,
                    **{k: c.get(k, 0) for k in fields if k not in ("work_id", "period_bin", "subfield")}})
    dest = ROOT / "data/scan/scan_by_work_bucket.csv"
    with dest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(out)
    print(f"{len(out)} works -> {dest}")


if __name__ == "__main__":
    main()
