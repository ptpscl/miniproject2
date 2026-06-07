"""Unit tests for splitting (the leakage guard is the important one)."""

import pandas as pd

from src.data import get_train_test_ids, build_train_test_data


def _toy_df(n=40):
    rows = []
    for i in range(n):
        label = i % 2
        for lang in ("EN", "TL"):
            rows.append({
                "id": f"id_{i}", "lang": lang,
                "context": f"ctx {i}", "utterance": f"utt {i}",
                "label": label, "original_en_label": label,
            })
    return pd.DataFrame(rows)


def test_no_id_leakage_between_train_and_test():
    df = _toy_df()
    train_ids, test_ids = get_train_test_ids(df, seed=42)
    assert len(train_ids & test_ids) == 0


def test_split_is_deterministic_per_seed():
    df = _toy_df()
    a = get_train_test_ids(df, seed=42)
    b = get_train_test_ids(df, seed=42)
    assert a == b


def test_build_train_test_has_text_column():
    df = _toy_df()
    tr, te = build_train_test_data(df, "EN", "TL", seed=42, input_variant="context_utterance")
    assert "text" in tr.columns and "text" in te.columns
    assert len(tr) > 0 and len(te) > 0