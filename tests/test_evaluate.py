"""Unit tests for eval/evaluate.py logic. No live API calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ---------------------------------------------------------------------------
# results_path sanitises model names and avoids overwrites
# ---------------------------------------------------------------------------

def test_results_path_sanitises_slashes():
    from eval.evaluate import results_path
    p = results_path("org/model-name", "v1")
    assert "/" not in p.name
    assert "org-model-name" in p.name


def test_results_path_contains_model_and_version():
    from eval.evaluate import results_path
    p = results_path("gpt-4o-mini", "v1")
    assert "gpt-4o-mini" in p.name
    assert "v1" in p.name


def test_results_path_increments_if_exists(tmp_path, monkeypatch):
    from eval import evaluate
    monkeypatch.setattr(evaluate, "__file__", str(tmp_path / "evaluate.py"))
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "eval_m_v1.csv").touch()
    p = evaluate.results_path("m", "v1")
    assert p.name == "eval_m_v1_2.csv"
    p.touch()
    p2 = evaluate.results_path("m", "v1")
    assert p2.name == "eval_m_v1_3.csv"


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def test_correct_when_tags_match():
    row = {"correct_tag": "junk"}
    pred = {"tag": "junk", "reasoning": ""}
    correct = pred["tag"] == row["correct_tag"]
    assert correct is True


def test_incorrect_when_tags_differ():
    row = {"correct_tag": "divine_teleology"}
    pred = {"tag": "junk", "reasoning": ""}
    correct = pred["tag"] == row["correct_tag"]
    assert correct is False


def test_mixed_label_always_incorrect():
    """Model can't return 'mixed'; mixed ground-truth rows always score False."""
    row = {"correct_tag": "mixed"}
    for tag in ("junk", "divine_teleology", "non_divine_teleology", "internal_essence"):
        assert (tag == row["correct_tag"]) is False


# ---------------------------------------------------------------------------
# Eval set loads with expected columns
# ---------------------------------------------------------------------------

def test_eval_set_has_required_columns():
    import csv
    from config.config import PATHS
    with open(PATHS["evaluation_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 49
    for col in ("text", "correct_tag", "xy_rationale"):
        assert col in rows[0], f"Missing column: {col}"


def test_eval_set_no_blank_correct_tag():
    import csv
    from config.config import PATHS
    with open(PATHS["evaluation_csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    blank = [r for r in rows if not r["correct_tag"].strip()]
    assert blank == [], f"Rows with blank correct_tag: {blank}"
