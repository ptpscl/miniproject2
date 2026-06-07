"""Lightweight tests for runner helper logic that does not train models."""

import pandas as pd

from src.results import ResultStore
from src.runners import load_precomputed_llm_predictions, majority_class_floor


def test_load_precomputed_llm_predictions_filters_languages(tmp_path):
    path = tmp_path / "llm.csv"
    pd.DataFrame([
        {
            "id": "1",
            "lang": "EN",
            "input_variant": "context_utterance",
            "y_true": 1,
            "y_pred": 1,
            "model_name": "gpt",
            "provider": "openai",
            "seed": 42,
            "original_en_label": 1,
        },
        {
            "id": "2",
            "lang": "DE",
            "input_variant": "context_utterance",
            "y_true": 0,
            "y_pred": 1,
            "model_name": "gpt",
            "provider": "openai",
            "seed": 42,
            "original_en_label": 0,
        },
    ]).to_csv(path, index=False)

    out = load_precomputed_llm_predictions(path)

    assert set(out["lang"]) == {"EN"}
    assert out["y_true"].dtype.kind in {"i", "u"}


def test_majority_class_floor_logs_baseline_result():
    rows = []
    for i in range(20):
        label = 1 if i < 14 else 0
        rows.append({
            "id": f"id_{i}",
            "lang": "EN",
            "context": "context",
            "utterance": "utterance",
            "label": label,
            "original_en_label": label,
        })
    df = pd.DataFrame(rows)
    store = ResultStore()

    majority_class_floor(store, df, test_lang="EN", seed=42, input_variant="utterance_only")

    assert len(store.results) == 1
    assert store.results[0]["model_name"] == "majority_floor"
    assert store.results[0]["model_family"] == "baseline"
    assert len(store.predictions_frame()) == store.results[0]["n_test"]
