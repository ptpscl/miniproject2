"""Unit tests for lightweight EDA helpers."""

import pandas as pd

from src import eda


def _eda_frame():
    return pd.DataFrame([
        {
            "id": "1",
            "lang": "EN",
            "context": "hello there",
            "utterance": "great",
            "label": 1,
            "original_en_label": 1,
        },
        {
            "id": "2",
            "lang": "EN",
            "context": "plain text",
            "utterance": "fine",
            "label": 0,
            "original_en_label": 0,
        },
        {
            "id": "1",
            "lang": "TL",
            "context": "kumusta",
            "utterance": "ayos",
            "label": 0,
            "original_en_label": 1,
        },
        {
            "id": "2",
            "lang": "TL",
            "context": "teksto",
            "utterance": "sige",
            "label": 0,
            "original_en_label": 0,
        },
    ])


def test_run_eda_returns_label_shift_summary(monkeypatch):
    monkeypatch.setattr(eda, "display", lambda *_args, **_kwargs: None)

    shift = eda.run_eda(_eda_frame())

    assert list(shift["lang"]) == ["TL"]
    assert shift["sum"].iloc[0] == 1
    assert shift["count"].iloc[0] == 2
    assert shift["percent_changed"].iloc[0] == 50.0


def test_compute_fleiss_kappa_skips_when_columns_unavailable():
    out = eda.compute_fleiss_kappa(_eda_frame(), ["missing_a", "missing_b"])

    assert out is None
