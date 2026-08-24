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
