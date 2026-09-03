#!/usr/bin/env python3
"""Render every figure for the handoff, as PNGs plus one combined PDF.

Error bars are a cluster bootstrap over WORKS within each period bin, matching
the method used in the original proportion evaluation. Works, not sentences,
are the unit of observation: a 1,200-sentence volume is one book, and treating
its sentences as independent would understate the interval badly.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports/figures"
C = ["divine_teleology", "non_divine_teleology", "internal_essence"]
LABEL = {"divine_teleology": "Divine teleology",
         "non_divine_teleology": "Non-divine teleology",
         "internal_essence": "Internal essence"}
COLOR = {"divine_teleology": "#B4451F", "non_divine_teleology": "#1F6FB4",
         "internal_essence": "#3E8E5A"}
P = ["1750_1799", "1800_1829", "1830_1849", "1850_1869",
     "1870_1889", "1890_1909", "1910_1929"]
NICE = {p: p.replace("_", "–") for p in P}
MIN_WORKS = 8
N_BOOT = 2000


def load():
    return list(csv.DictReader((ROOT / "data/scan/scan_by_work_bucket.csv").open()))


def bootstrap(works, bucket, seed=7):
    """Pooled share per category, with a 95% interval from resampling works."""
    tot = sum(int(w[f"{bucket}_n"]) for w in works)
    if tot == 0 or len(works) < 2:
        return None
    point = {c: 100 * sum(int(w[f"{bucket}_{c}"]) for w in works) / tot for c in C}
    rng = random.Random(seed)
    draws = {c: [] for c in C}
    for _ in range(N_BOOT):
        pick = [works[rng.randrange(len(works))] for _ in range(len(works))]
        t = sum(int(w[f"{bucket}_n"]) for w in pick)
        if not t:
            continue
        for c in C:
            draws[c].append(100 * sum(int(w[f"{bucket}_{c}"]) for w in pick) / t)
    lo, hi = {}, {}
    for c in C:
        d = sorted(draws[c])
        lo[c] = d[int(0.025 * len(d))]
        hi[c] = d[int(0.975 * len(d)) - 1]
    return point, lo, hi, tot, len(works)


def series(rows, bucket, subfield=None):
    out = {}
    for p in P:
        ws = [w for w in rows if w["period_bin"] == p and int(w[f"{bucket}_n"]) > 0
              and (subfield is None or w["subfield"] == subfield)]
        if len(ws) < MIN_WORKS:
            continue
        r = bootstrap(ws, bucket)
        if r:
            out[p] = r
    return out


def plot(ax, data, title, subtitle=""):
    xs = [i for i, p in enumerate(P) if p in data]
    for c in C:
        y = [data[P[i]][0][c] for i in xs]
        lo = [data[P[i]][0][c] - data[P[i]][1][c] for i in xs]
        hi = [data[P[i]][2][c] - data[P[i]][0][c] for i in xs]
        ax.errorbar(xs, y, yerr=[lo, hi], marker="o", capsize=3, lw=1.8,
                    ms=5, color=COLOR[c], label=LABEL[c])
    ax.set_xticks(range(len(P)))
    ax.set_xticklabels([NICE[p] for p in P], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of explanatory sentences")
    ax.set_ylim(0, 100)
    ax.grid(alpha=.25, lw=.6)
    ax.set_title(title + ("\n" + subtitle if subtitle else ""), fontsize=11, loc="left")


def annotate_n(ax, data):
    for i, p in enumerate(P):
        if p in data:
            ax.annotate(f"{data[p][4]}", (i, 2), fontsize=6.5, ha="center", color="#666")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load()
    figures = []

    for bucket, name, desc in [
        ("all", "01_proportions_all", "All animal sentences"),
        ("whole", "02_proportions_whole_animal", "Whole-animal sentences only"),
        ("part", "03_proportions_animal_part", "Animal-part sentences only"),
    ]:
        data = series(rows, bucket)
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        plot(ax, data, desc,
             "95% intervals from resampling works; work count at base")
        annotate_n(ax, data)
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=200)
        figures.append(fig)

    # Divine teleology on its own axis: it never exceeds 5% and is unreadable
    # on a 0-100 scale, but it is the clearest trend in the data.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for bucket, style, lab in [("all", "-", "All sentences"),
                               ("whole", "--", "Whole-animal only")]:
        data = series(rows, bucket)
        xs = [i for i, p_ in enumerate(P) if p_ in data]
        y = [data[P[i]][0]["divine_teleology"] for i in xs]
        lo = [y[k] - data[P[i]][1]["divine_teleology"] for k, i in enumerate(xs)]
        hi = [data[P[i]][2]["divine_teleology"] - y[k] for k, i in enumerate(xs)]
        ax.errorbar(xs, y, yerr=[lo, hi], marker="o", capsize=3, lw=1.8, ms=5,
                    ls=style, color=COLOR["divine_teleology"], alpha=1 if bucket == "all" else .55,
                    label=lab)
    ax.set_xticks(range(len(P)))
    ax.set_xticklabels([NICE[p_] for p_ in P], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of explanatory sentences")
    ax.set_ylim(0, 8)
    ax.grid(alpha=.25, lw=.6)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Divine teleology alone, on its own scale\n"
                 "The clearest trend in the data: a steady decline to near zero",
                 fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "04_divine_teleology_detail.png", dpi=200)
    figures.append(fig)

    subs = sorted({w["subfield"] for w in rows},
                  key=lambda s: -sum(int(w["all_n"]) for w in rows if w["subfield"] == s))
    panels = [s for s in subs if len(series(rows, "all", s)) >= 3][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, s in zip(axes.flat, panels):
        plot(ax, series(rows, "all", s), s.replace("_", " "))
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Proportions by subfield, with 95% intervals over works", x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "05_proportions_by_subfield.png", dpi=200)
    figures.append(fig)

    # validation: agreement with the teacher, per category per period
    v = [r for r in csv.DictReader((ROOT / "data/scan/validation_scored.csv").open())
         if r["gpt_tag"] != "error"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for c in C:
        xs, ys = [], []
        for i, p in enumerate(P):
            g = [r for r in v if r["model_pred"] == c and r["period_bin"] == p]
            if g:
                xs.append(i)
                ys.append(100 * sum(1 for r in g if r["gpt_tag"] == c) / len(g))
        ax.plot(xs, ys, marker="o", color=COLOR[c], label=LABEL[c], lw=1.8, ms=5)
    overall = {c: 100 * sum(1 for r in v if r["model_pred"] == c and r["gpt_tag"] == c)
               / max(sum(1 for r in v if r["model_pred"] == c), 1) for c in C}
    # With 30 rows per cell, the 95% sampling interval on a ~50% rate is about
    # +/-18 points. Almost all of the scatter below sits inside that, which is
    # why the per-period wobble should not be read as a real change.
    import math
    band = 1.96 * math.sqrt(0.5 * 0.5 / 30) * 100
    mid = sum(overall.values()) / 3
    ax.axhspan(mid - band, mid + band, color="#999", alpha=.13, zorder=0)
    for c in C:
        ax.axhline(overall[c], color=COLOR[c], ls=":", lw=1, alpha=.6)
    ax.annotate(f"shaded: 95% sampling range for n=30 (±{band:.0f} pts)",
                (0.02, 0.04), xycoords="axes fraction", fontsize=8, color="#555")
    ax.set_xticks(range(len(P)))
    ax.set_xticklabels([NICE[p] for p in P], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% agreement with teacher model")
    ax.set_ylim(0, 100)
    ax.grid(alpha=.25, lw=.6)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Agreement with the teacher model, 30 sentences per category per period\n"
                 "Dotted lines are the overall rate for each category", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "06_validation_agreement.png", dpi=200)
    figures.append(fig)

    with PdfPages(ROOT / "reports/teleological_essentialism_figures.pdf") as pdf:
        for f in figures:
            pdf.savefig(f)
    for f in figures:
        plt.close(f)
    print(f"wrote {len(figures)} figures to {FIG}")
    print(f"combined PDF -> reports/teleological_essentialism_figures.pdf")
    print("\noverall agreement with teacher: " +
          ", ".join(f"{LABEL[c]} {overall[c]:.0f}%" for c in C))


if __name__ == "__main__":
    main()
