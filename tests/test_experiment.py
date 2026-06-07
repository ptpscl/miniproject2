"""Unit tests for experiment aggregation and paired-comparison helpers."""

import pandas as pd

from src.experiment import build_stats_table, paired_compare, summarise


def _results_frame():
    rows = []
    for seed, en_score, tl_score, target_score in [
        (1, 0.70, 0.50, 0.62),
        (2, 0.60, 0.55, 0.66),
        (3, 0.65, 0.52, 0.64),
    ]:
        for setting, score in [
            ("EN->EN", en_score),
            ("EN->TL", tl_score),
            ("TL->TL", target_score),
        ]:
            rows.append({
                "model_family": "scratch_transformer",
                "model_name": "toy_model",
                "setting": setting,
                "input_variant": "context_utterance",
                "seed": seed,
                "macro_f1": score,
                "sarcasm_f1": score - 0.05,
                "accuracy": score + 0.02,
                "n_params": 123,
                "train_time_s": 1.5,
            })
    return pd.DataFrame(rows)


def test_summarise_groups_and_counts_runs():
    out = summarise(_results_frame())
    en_row = out[(out["model_name"] == "toy_model") & (out["setting"] == "EN->EN")]

    assert len(en_row) == 1
    assert en_row["n_runs"].iloc[0] == 3
    assert abs(en_row["mean_macro_f1"].iloc[0] - 0.65) < 1e-12
    assert "mean_train_time_s" in out.columns


def test_paired_compare_uses_matched_seeds():
    out = paired_compare(_results_frame(), "toy_model", "EN->EN", "EN->TL")

    assert out["n_pairs"] == 3
    assert abs(out["mean_a"] - 0.65) < 1e-12
    assert abs(out["mean_b"] - 0.5233333333333333) < 1e-12
    assert out["delta"] < 0


def test_build_stats_table_excludes_llm_reference():
    df = _results_frame()
    llm = df.iloc[[0]].copy()
    llm["model_family"] = "llm_reference"
    llm["model_name"] = "openai_reference"
    combined = pd.concat([df, llm], ignore_index=True)

    out = build_stats_table(combined)

    assert set(out["model_name"]) == {"toy_model"}
    assert len(out) == 2
