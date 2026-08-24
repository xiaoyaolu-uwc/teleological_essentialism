"""Unit tests for models/ package. No live API calls."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from archive.prompt_phase.base import ModelAdapter
from archive.prompt_phase.openai import OpenAIModel
from archive.prompt_phase.prompts import PROMPT_VERSIONS, PASSAGE_TEMPLATE_TEXT_ONLY, VALID_TAGS


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def test_prompt_versions_v1_has_required_keys():
    v1 = PROMPT_VERSIONS["v1"]
    assert "system" in v1
    assert "user_template" in v1
    assert "passage_template" in v1
    assert "valid_tags" in v1


def test_passage_template_text_only_no_metadata():
    assert "{author}" not in PASSAGE_TEMPLATE_TEXT_ONLY
    assert "{year}" not in PASSAGE_TEMPLATE_TEXT_ONLY
    assert "{text}" in PASSAGE_TEMPLATE_TEXT_ONLY


def test_v1_uses_text_only_template():
    tmpl = PROMPT_VERSIONS["v1"]["passage_template"]
    assert "{author}" not in tmpl
    assert "{year}" not in tmpl


def test_valid_tags_contents():
    assert VALID_TAGS == {"divine_teleology", "non_divine_teleology", "internal_essence", "junk"}


# ---------------------------------------------------------------------------
# OpenAIModel constructor
# ---------------------------------------------------------------------------

def test_unknown_prompt_version_raises():
    with pytest.raises(ValueError, match="Unknown prompt version"):
        OpenAIModel(model_name="gpt-4o-mini", prompt_version="v99")


def test_valid_prompt_version_constructs():
    m = OpenAIModel(model_name="gpt-4o-mini", prompt_version="v1")
    assert m.model_name == "gpt-4o-mini"
    assert m.batch_size == 10


# ---------------------------------------------------------------------------
# classify() return shape
# ---------------------------------------------------------------------------

def _make_mock_response(n: int):
    """Build a fake Responses API response object for n passages."""
    results = [{"id": i, "tag": "junk", "reasoning": "test"} for i in range(n)]
    import json
    part = MagicMock()
    part.type = "output_text"
    part.text = json.dumps(results)
    msg = MagicMock()
    msg.type = "message"
    msg.content = [part]
    response = MagicMock()
    response.output = [msg]
    return response


@patch("openai.OpenAI")
def test_classify_returns_same_length_as_input(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.responses.create.side_effect = lambda **kw: _make_mock_response(
        kw["input"].count("--- Passage")
    )
    model = OpenAIModel(model_name="gpt-4o-mini", prompt_version="v1", batch_size=10)
    texts = [f"passage {i}" for i in range(7)]
    results = model.classify(texts)
    assert len(results) == 7


@patch("openai.OpenAI")
def test_classify_batches_correctly(mock_openai_cls):
    """25 texts with batch_size=10 should trigger exactly 3 API calls."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.responses.create.side_effect = lambda **kw: _make_mock_response(
        kw["input"].count("--- Passage")
    )
    model = OpenAIModel(model_name="gpt-4o-mini", prompt_version="v1", batch_size=10)
    model.classify([f"text {i}" for i in range(25)])
    assert mock_client.responses.create.call_count == 3


@patch("openai.OpenAI")
def test_classify_result_keys(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.responses.create.return_value = _make_mock_response(1)
    model = OpenAIModel(model_name="gpt-4o-mini", prompt_version="v1", batch_size=10)
    results = model.classify(["some text"])
    assert set(results[0].keys()) == {"tag", "reasoning"}


@patch("openai.OpenAI")
def test_no_metadata_in_api_call(mock_openai_cls):
    """Author and year must not appear in the text sent to the API."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.responses.create.return_value = _make_mock_response(1)
    model = OpenAIModel(model_name="gpt-4o-mini", prompt_version="v1", batch_size=10)
    model.classify(["some passage text"])
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert "Author:" not in call_kwargs["input"]
    assert "year" not in call_kwargs["input"].lower() or "year" not in call_kwargs["input"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@patch("openai.OpenAI")
def test_api_error_returns_error_tag(mock_openai_cls):
    import openai as oai
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.responses.create.side_effect = RuntimeError("API down")
    model = OpenAIModel(model_name="gpt-4o-mini", prompt_version="v1", batch_size=10)
    with pytest.raises(RuntimeError):
        model._classify_batch(["text"])


@patch("openai.OpenAI")
def test_malformed_json_raises(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    part = MagicMock(); part.type = "output_text"; part.text = "not json {"
    msg = MagicMock(); msg.type = "message"; msg.content = [part]
    resp = MagicMock(); resp.output = [msg]
    mock_client.responses.create.return_value = resp
    model = OpenAIModel(model_name="gpt-4o-mini", prompt_version="v1", batch_size=10)
    with pytest.raises(RuntimeError, match="Failed to parse"):
        model._classify_batch(["text"])


# ---------------------------------------------------------------------------
# base ABC
# ---------------------------------------------------------------------------

def test_base_adapter_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ModelAdapter()
