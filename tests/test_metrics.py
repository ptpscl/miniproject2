"""Unit tests for pure metric and input-construction functions."""

import numpy as np
import pytest

from src.metrics import (
    make_input_text,
    compute_metrics,
    make_run_name,
    format_seconds,
)


def test_make_input_text_utterance_only():
    row = {"context": "A said hi", "utterance": "Oh great"}
    assert make_input_text(row, "utterance_only") == "Oh great"


def test_make_input_text_context_utterance():
    row = {"context": "A said hi", "utterance": "Oh great"}
    out = make_input_text(row, "context_utterance")
    assert "[CONTEXT]" in out and "[UTTERANCE]" in out
    assert "A said hi" in out and "Oh great" in out


def test_make_input_text_handles_nan():
    assert make_input_text({"context": np.nan, "utterance": "hey"}, "utterance_only") == "hey"


def test_make_input_text_bad_variant_raises():
    with pytest.raises(ValueError):
        make_input_text({"context": "a", "utterance": "b"}, "nope")


def test_compute_metrics_known_case():
    # y_true=[1,1,0,0], y_pred=[1,0,0,0]: sarcasm P=1.0, R=0.5, acc=0.75
    m = compute_metrics([1, 1, 0, 0], [1, 0, 0, 0])
    assert abs(m["accuracy"] - 0.75) < 1e-9
    assert abs(m["sarcasm_precision"] - 1.0) < 1e-9
    assert abs(m["sarcasm_recall"] - 0.5) < 1e-9
    assert abs(m["sarcasm_f1"] - (2 * 1.0 * 0.5 / 1.5)) < 1e-9


def test_compute_metrics_perfect():
    m = compute_metrics([1, 0, 1, 0], [1, 0, 1, 0])
    assert abs(m["macro_f1"] - 1.0) < 1e-9


def test_make_run_name_format():
    assert make_run_name("EN", "TL", "m", "context_utterance", 42) == \
        "EN->TL__m__context_utterance__seed42"


def test_format_seconds():
    assert format_seconds(67) == "1m07s"