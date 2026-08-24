#!/usr/bin/env python3
"""
Tests for the scoring code behind every published confidence claim.

These matter more than they look: docs/PROPORTION_EVAL_RESULTS.md rests
entirely on evaluate_proportions.py being arithmetically right, and the whole
analysis is deterministic given the committed per-row predictions -- so it is
cheaply and exactly testable.

    python3 -m pytest tests/ -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate_proportions import (
    mix, wilson, variant_labels, per_work_errors, bias_table, within_work_ranking,
    across_work_ranking, cluster_bootstrap_ci, binned,
)

DT, NDT, IE, JUNK = "divine_teleology", "non_divine_teleology", "internal_essence", "junk"


def row(true, gate, s2, work="W", year="1800"):
    return {"true_tag": true, "gate_pred": gate, "s2_pred": s2, "work": work, "year": year}


def test_mix_excludes_junk_and_normalises():
    m, n = mix([DT, DT, NDT, JUNK])
    assert n == 3
    assert m[DT] == 2 / 3 and m[NDT] == 1 / 3 and m[IE] == 0.0


def test_mix_returns_none_when_nothing_survives():
    m, n = mix([JUNK, JUNK])
    assert m is None and n == 0


def test_perfect_pipeline_has_zero_error():
    rows = [row(DT, "non_junk", DT), row(NDT, "non_junk", NDT), row(IE, "non_junk", IE)]
    pw = per_work_errors({"W": rows}, "end_to_end")
    assert pw["W"]["tvd"] == 0.0
    assert all(v == 0.0 for v in pw["W"]["signed_error"].values())


def test_gate_dropping_a_category_skews_the_mix():
    # IE is truly a third of the text but the gate discards every IE row
    rows = [row(DT, "non_junk", DT), row(NDT, "non_junk", NDT), row(IE, "junk", IE)]
    pw = per_work_errors({"W": rows}, "end_to_end")
    assert pw["W"]["true_mix"][IE] == 1 / 3
    assert pw["W"]["pred_mix"][IE] == 0.0
    assert pw["W"]["signed_error"][IE] == -1 / 3


def test_variants_isolate_the_two_stages():
    # gate wrongly drops a DT row; stage 2 wrongly calls an NDT row IE
    rows = [row(DT, "junk", DT), row(NDT, "non_junk", IE)]
    assert variant_labels(rows, "end_to_end") == [None, IE]
    assert variant_labels(rows, "perfect_gate") == [DT, IE]      # gate error removed
    assert variant_labels(rows, "perfect_stage2") == [None, NDT]  # stage-2 error removed


def test_leaked_junk_is_counted_in_the_predicted_mix():
    # a true-junk row the gate passed becomes a real error in the output
    rows = [row(DT, "non_junk", DT), row(JUNK, "non_junk", NDT)]
    pw = per_work_errors({"W": rows}, "end_to_end")
    assert pw["W"]["true_mix"][DT] == 1.0
    assert pw["W"]["pred_mix"][NDT] == 0.5


def test_teleology_is_the_mirror_of_essence():
    # DT+NDT = 1 - IE among non-junk, so it is not independent evidence
    rows = [row(DT, "non_junk", DT), row(NDT, "non_junk", IE), row(IE, "non_junk", IE)]
    v = per_work_errors({"W": rows}, "end_to_end")["W"]
    assert abs(v["teleology_signed_error"] + v["signed_error"][IE]) < 1e-12


def test_bias_table_reports_spread_across_works():
    works = {f"W{i}": [row(DT, "non_junk", DT if i % 2 == 0 else NDT, work=f"W{i}")]
             for i in range(4)}
    b = bias_table(per_work_errors(works, "end_to_end"))
    assert b[DT]["n_works"] == 4
    assert b[DT]["sd"] > 0                      # the works genuinely disagree
    assert b[DT]["predict_90pct"] == 1.645 * b[DT]["sd"]


def test_within_work_ranking_detects_a_flipped_order():
    # true order DT > NDT; predicted order NDT > DT
    rows = [row(DT, "non_junk", NDT), row(DT, "non_junk", NDT), row(NDT, "non_junk", DT)]
    r = within_work_ranking(per_work_errors({"W": rows}, "end_to_end"))
    assert r["full_order_correct"] == 0


def test_wilson_brackets_the_point_estimate():
    lo, hi = wilson(9, 10)
    assert lo < 0.9 < hi and 0 <= lo and hi <= 1
    assert wilson(0, 0) == (None, None)


def test_wilson_handles_the_boundary():
    lo, hi = wilson(10, 10)          # normal approximation would give (1, 1)
    assert lo < 1.0 and hi == 1.0


# --- the across-book ranking claim -------------------------------------------
# "ordering two texts is correct 90% of the time, and ~99% once the true gap
# exceeds 20pp" is the claim the decade chart rests on, so it gets its own
# coverage rather than riding on the within-book tests.

def _works(spec):
    """spec: {work: (true_mix, pred_mix)} -> rows that produce exactly those mixes."""
    out = {}
    for w, (t, p) in spec.items():
        rows = []
        for cat, n in zip((DT, NDT, IE), t):
            rows += [row(cat, "junk", cat, work=w)] * n      # true side, gate drops them
        for cat, n in zip((DT, NDT, IE), p):
            rows += [row(JUNK, "non_junk", cat, work=w)] * n  # predicted side
        out[w] = rows
    return out


def test_across_work_ranking_scores_every_ordered_pair():
    # A has more DT than B, less NDT, less IE -- and the model agrees on all three.
    # No category may be tied between the books, or that pair is (correctly) skipped.
    pw = per_work_errors(_works({"A": ((8, 1, 1), (8, 1, 1)),
                                 "B": ((2, 6, 2), (2, 6, 2))}), "end_to_end")
    r = across_work_ranking(pw)
    assert r["_pooled_3cat"]["n"] == 3          # 1 pair x 3 untied categories
    assert r["_pooled_3cat"]["correct"] == 3    # all orderings recovered


def test_across_work_ranking_catches_an_inverted_ordering():
    # A truly has more DT than B, but the model predicts the reverse
    pw = per_work_errors(_works({"A": ((8, 1, 1), (2, 7, 1)),
                                 "B": ((2, 7, 1), (8, 1, 1))}), "end_to_end")
    assert across_work_ranking(pw)["divine_teleology"]["correct"] == 0


def test_across_work_ranking_skips_exact_ties():
    # identical true mixes -> no ordering exists, so nothing should be scored
    pw = per_work_errors(_works({"A": ((5, 3, 2), (5, 3, 2)),
                                 "B": ((5, 3, 2), (3, 5, 2))}), "end_to_end")
    assert across_work_ranking(pw)["_pooled_3cat"]["n"] == 0


def test_cluster_bootstrap_is_wider_than_wilson():
    """The load-bearing correction: pairwise comparisons come from a handful of
    books and are dependent, so resampling books must give a wider interval
    than a Wilson interval that assumes 354 independent observations."""
    spec = {}
    for i in range(8):                       # 8 books, one of them mis-ordered
        t = (8 - i, i, 2)
        p = (i, 8 - i, 2) if i == 3 else t
        spec[f"W{i}"] = (t, p)
    pw = per_work_errors(_works(spec), "end_to_end")
    r = across_work_ranking(pw)
    w_lo, w_hi = r["_pooled_3cat"]["ci"]
    c_lo, c_hi = r["_pooled_3cat"]["ci_cluster_bootstrap"]
    assert c_lo is not None
    assert (c_hi - c_lo) > (w_hi - w_lo), "cluster bootstrap must not be narrower than Wilson"


def test_binned_puts_each_gap_in_the_right_bucket():
    obs = [(0.01, True), (0.30, True), (0.30, False)]
    b = {(x["gap_lo"], x["gap_hi"]): x for x in binned(obs)}
    assert b[(0.0, 0.025)]["n"] == 1 and b[(0.0, 0.025)]["rate"] == 1.0
    assert b[(0.20, 1.01)]["n"] == 2 and b[(0.20, 1.01)]["rate"] == 0.5
