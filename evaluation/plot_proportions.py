#!/usr/bin/env python3
"""
plot_proportions.py
===================
Three figures from evaluation/results/proportions/proportion_metrics.json:
  A  predicted vs. true category share, one point per book per category
  B  the blog-style trend: category share by year, true vs. predicted
  C  across-book ranking accuracy as a function of the true gap

Usage: python3 evaluation/plot_proportions.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATS = ["divine_teleology", "non_divine_teleology", "internal_essence"]
SHORT = {"divine_teleology": "Divine teleology", "non_divine_teleology": "Non-divine teleology",
         "internal_essence": "Internal essence"}
COLS = {"divine_teleology": "#B45309", "non_divine_teleology": "#1D4ED8", "internal_essence": "#047857"}


def main():
    r = json.load(open("evaluation/results/proportions/proportion_metrics.json"))
    e2e = r["variants"]["end_to_end"]
    pw = e2e["per_work"]
    works = sorted(pw, key=lambda w: int(pw[w]["year"]))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # --- A: predicted vs true -------------------------------------------------
    ax = axes[0]
    ax.plot([0, 90], [0, 90], color="#9CA3AF", lw=1, ls="--", zorder=1)
    for c in CATS:
        ax.scatter([100 * pw[w]["true_mix"][c] for w in works],
                   [100 * pw[w]["pred_mix"][c] for w in works],
                   s=46, alpha=.85, color=COLS[c], label=SHORT[c], zorder=3, edgecolor="white", lw=.6)
    ax.set_xlabel("True share (%)"); ax.set_ylabel("Predicted share (%)")
    ax.set_title("A. Predicted vs. true category share\n16 held-out books", fontsize=11)
    ax.legend(fontsize=8, frameon=False); ax.set_xlim(-3, 92); ax.set_ylim(-3, 92)

    # --- B: the error bar itself ---------------------------------------------
    # A book-by-book trend line is unreadable here (single books swing wildly --
    # exactly the problem decade-pooling exists to solve), so this panel shows
    # the quantity the blog post actually needs: the distribution of per-book
    # signed errors, whose spread IS the error bar on any reported share.
    ax = axes[1]
    import random
    random.seed(0)
    for i, c in enumerate(CATS):
        errs = [100 * pw[w]["signed_error"][c] for w in works]
        m = 100 * e2e["bias"][c]["mean_signed_error"]
        band = 100 * e2e["bias"][c]["predict_90pct"]
        ax.add_patch(plt.Rectangle((i - .3, m - band), .6, 2 * band,
                                   color=COLS[c], alpha=.16, lw=0))
        ax.plot([i - .3, i + .3], [m, m], color=COLS[c], lw=2.2)
        ax.scatter([i + random.uniform(-.13, .13) for _ in errs], errs,
                   s=34, color=COLS[c], alpha=.9, edgecolor="white", lw=.6, zorder=3)
    ax.axhline(0, color="#374151", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[c].replace(" ", "\n") for c in CATS], fontsize=8.5)
    ax.set_ylabel("Predicted - true share (pp)")
    ax.set_title("B. Error on a new book\nbar = mean bias, band = +/-90%", fontsize=11)

    # --- C: ranking resolving power ------------------------------------------
    ax = axes[2]
    bins = e2e["across_work_ranking"]["_pooled_3cat"]["by_gap"]
    xs = [f"{100*b['gap_lo']:.0f}-{100*b['gap_hi']:.0f}" if b["gap_hi"] < 1 else f"{100*b['gap_lo']:.0f}+"
          for b in bins]
    rates = [100 * b["rate"] for b in bins]
    lo = [100 * (b["rate"] - b["ci_lo"]) for b in bins]
    hi = [100 * (b["ci_hi"] - b["rate"]) for b in bins]
    ax.bar(xs, rates, color="#4F46E5", alpha=.85, width=.62)
    ax.errorbar(xs, rates, yerr=[lo, hi], fmt="none", ecolor="#111827", elinewidth=1.1, capsize=3)
    ax.axhline(50, color="#9CA3AF", ls=":", lw=1)
    for x, v, b in zip(xs, rates, bins):
        ax.text(x, v + 3.5, f"n={b['n']}", ha="center", fontsize=7.5, color="#374151")
    ax.set_ylim(0, 112); ax.set_xlabel("True gap between the two books (pp)")
    ax.set_ylabel("Ordering recovered correctly (%)")
    ax.set_title("C. When can two books be ordered?\n95% Wilson intervals", fontsize=11)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = Path("evaluation/results/proportions/proportion_figures.png")
    plt.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
