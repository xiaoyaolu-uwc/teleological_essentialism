#!/usr/bin/env python3
"""Aggregate per-volume predictions into the master analysis sheet.

Two levels are written, and nothing else:

  scan_by_work.csv    one row per WORK, raw counts only
  scan_by_period.csv  one row per period bin, aggregated over works

Counts, never proportions or rankings. Every quantity the analysis needs --
shares, gaps, orderings, bootstrap intervals -- is a function of these counts,
and a stored derived value is a value that can go stale.

Volumes are aggregated up to `work_id` first. A multi-volume work is one
observation, not nine: the proportion evaluation was calibrated per book, so
the work is the unit that error bars attach to.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SENT_DIR = ROOT / "data/scan/sentences"
PRED_DIR = ROOT / "data/scan/predictions"
OUT_DIR = ROOT / "data/scan"

CATEGORIES = ["divine_teleology", "non_divine_teleology", "internal_essence"]
PERIOD_ORDER = [
    "1750_1799", "1800_1829", "1830_1849", "1850_1869",
    "1870_1889", "1890_1909", "1910_1929",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-explanatory", type=int, default=30,
                    help="works with fewer surviving rows are flagged, not dropped")
    args = ap.parse_args()

    works: dict[str, dict] = defaultdict(lambda: {
        "volumes": 0, "n_rows": 0, "n_junk": 0, "n_explanatory": 0,
        **{f"n_{c}": 0 for c in CATEGORIES},
        "years": [], "titles": [], "sources": set(), "period_bin": "",
    })

    for pred_path in sorted(PRED_DIR.glob("*.csv")):
        sent_path = SENT_DIR / pred_path.name
        if not sent_path.exists() or pred_path.stat().st_size == 0:
            continue
        sent_rows = list(csv.DictReader(sent_path.open()))
        pred_rows = list(csv.DictReader(pred_path.open()))
        if len(sent_rows) != len(pred_rows):
            print(f"WARNING: length mismatch for {pred_path.name}; skipped")
            continue

        info = sent_rows[0]
        work = works[info["work_id"]]
        work["volumes"] += 1
        work["period_bin"] = info["period_bin"]
        work["sources"].add(info["source"])
        work["titles"].append(info["title"])
        if info["year"]:
            work["years"].append(int(info["year"]))

        for pred in pred_rows:
            work["n_rows"] += 1
            if pred["gate_pred"] == "junk":
                work["n_junk"] += 1
                continue
            work["n_explanatory"] += 1
            if pred["s2_pred"] in CATEGORIES:
                work[f"n_{pred['s2_pred']}"] += 1

    work_fields = [
        "work_id", "period_bin", "year", "title", "sources", "volumes",
        "n_rows", "n_junk", "n_explanatory",
        *[f"n_{c}" for c in CATEGORIES], "below_min_explanatory",
    ]
    work_rows = []
    for work_id, w in works.items():
        work_rows.append({
            "work_id": work_id,
            "period_bin": w["period_bin"],
            "year": min(w["years"]) if w["years"] else "",
            "title": sorted(w["titles"], key=len)[0][:200],
            "sources": "/".join(sorted(w["sources"])),
            "volumes": w["volumes"],
            "n_rows": w["n_rows"],
            "n_junk": w["n_junk"],
            "n_explanatory": w["n_explanatory"],
            **{f"n_{c}": w[f"n_{c}"] for c in CATEGORIES},
            "below_min_explanatory": "1" if w["n_explanatory"] < args.min_explanatory else "0",
        })
    work_rows.sort(key=lambda r: (r["period_bin"], str(r["year"])))

    with (OUT_DIR / "scan_by_work.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=work_fields)
        writer.writeheader()
        writer.writerows(work_rows)

    period_fields = ["period_bin", "n_works", "n_works_used", "volumes",
                     "n_rows", "n_junk", "n_explanatory",
                     *[f"n_{c}" for c in CATEGORIES]]
    period_rows = []
    for period in PERIOD_ORDER:
        members = [r for r in work_rows if r["period_bin"] == period]
        used = [r for r in members if r["below_min_explanatory"] == "0"]
        if not members:
            continue
        period_rows.append({
            "period_bin": period,
            "n_works": len(members),
            "n_works_used": len(used),
            "volumes": sum(r["volumes"] for r in used),
            "n_rows": sum(r["n_rows"] for r in used),
            "n_junk": sum(r["n_junk"] for r in used),
            "n_explanatory": sum(r["n_explanatory"] for r in used),
            **{f"n_{c}": sum(r[f"n_{c}"] for r in used) for c in CATEGORIES},
        })

    with (OUT_DIR / "scan_by_period.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=period_fields)
        writer.writeheader()
        writer.writerows(period_rows)

    print(f"{len(work_rows)} works -> {OUT_DIR / 'scan_by_work.csv'}")
    print(f"{len(period_rows)} periods -> {OUT_DIR / 'scan_by_period.csv'}\n")
    print(f"{'period':<12}{'works':>7}{'used':>6}{'expl rows':>11}   share DT/NDT/IE")
    for r in period_rows:
        total = max(r["n_explanatory"], 1)
        shares = "/".join(f"{100*r[f'n_{c}']/total:.1f}" for c in CATEGORIES)
        print(f"{r['period_bin']:<12}{r['n_works']:>7}{r['n_works_used']:>6}"
              f"{r['n_explanatory']:>11,}   {shares}")


if __name__ == "__main__":
    main()
