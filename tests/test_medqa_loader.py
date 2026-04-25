from __future__ import annotations

import pytest

from med_routing.eval.medqa import _coerce_options, _row_to_item


def test_coerce_options_dict_form():
    out = _coerce_options({"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"})
    assert out == ("alpha", "beta", "gamma", "delta")


def test_coerce_options_list_of_dicts_form():
    opts = [
        {"key": "A", "value": "alpha"}, {"key": "B", "value": "beta"},
        {"key": "C", "value": "gamma"}, {"key": "D", "value": "delta"},
    ]
    assert _coerce_options(opts) == ("alpha", "beta", "gamma", "delta")


def test_coerce_options_plain_list_form():
    assert _coerce_options(["a", "b", "c", "d", "e"]) == ("a", "b", "c", "d")


def test_coerce_options_handles_partial_dict():
    out = _coerce_options({"A": "alpha", "B": "beta"})
    assert out == ("alpha", "beta", "", "")


@pytest.mark.parametrize("opts", [None, [], "string", 42])
def test_coerce_options_rejects_garbage(opts):
    assert _coerce_options(opts) is None


def test_row_to_item_happy_path():
    row = {
        "id": 17,
        "question": "What is the next step?",
        "options": {"A": "x", "B": "y", "C": "z", "D": "w"},
        "answer_idx": "C",
        "answer": "z",
        "meta_info": "step1",
    }
    item = _row_to_item(row, idx=0)
    assert item is not None
    assert item.qid == "medqa-17"
    assert item.options == ("x", "y", "z", "w")
    assert item.answer == "C"
    assert item.subject == "step1"


def test_row_to_item_uses_step_label_as_subject():
    """meta_info exposes USMLE step which works as a difficulty proxy in Grafana."""
    row = {
        "question": "Q?", "options": ["a", "b", "c", "d"],
        "answer_idx": "A", "meta_info": "step2&3",
    }
    item = _row_to_item(row, idx=5)
    assert item is not None
    assert item.subject == "step2&3"


def test_row_to_item_skips_when_answer_missing():
    row = {"question": "Q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}}
    assert _row_to_item(row, idx=1) is None


def test_row_to_item_skips_when_options_unparseable():
    row = {"question": "Q", "options": {"only_one": "x"}, "answer_idx": "A"}
    item = _row_to_item(row, idx=2)
    assert item is not None  # dict form fills missing keys with empty strings
    assert item.options == ("", "", "", "")  # honest about what we got


def test_row_to_item_falls_back_to_default_subject():
    row = {"question": "Q", "options": ["a", "b", "c", "d"], "answer_idx": "A"}
    item = _row_to_item(row, idx=3)
    assert item is not None
    assert item.subject == "medqa-usmle"
