"""Unit tests for the precomputed-LLM filter and relabel logic."""

import pandas as pd

from src.results import ResultStore
from src.runners import add_precomputed_llm_results


def _toy_llm_df():
    rows = []
    for lang in ("EN", "TL"):
        for i in range(6):
            rows.append({
                "id": f"id_{i}", "lang": lang, "input_variant": "context_utterance",
                "y_true": i % 2, "y_pred": i % 2, "model_name": "gpt-4o-mini",
                "provider": "openai", "seed": 42, "original_en_label": i % 2,
            })
    return pd.DataFrame(rows)


def test_llm_rows_relabelled_to_in_language_settings():
    store = ResultStore()
    add_precomputed_llm_results(store, _toy_llm_df())
    settings = {r["setting"] for r in store.results}
    # EN rows -> EN->EN, TL rows -> TL->TL (never LLM->EN)
    assert settings == {"EN->EN", "TL->TL"}
    assert all(r["model_family"] == "llm_reference" for r in store.results)