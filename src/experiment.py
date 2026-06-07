"""Experiment registries, the main loop, and results aggregation.

The loop follows the two-act design: Act 1 optimises every rung in the
EN->EN setting (including ablations); Act 2 stress-tests the main ladder
across all settings (EN->EN, EN->TL, TL->TL) for cross-lingual transfer
and target-language fine-tuning. Pretrained models run only on the
expensive-seed subset.
"""

import numpy as np
import pandas as pd

from . import config
from .results import ResultStore
from .runners import (
    CLASSICAL_MODELS,
    TRANSFORMER_MODELS,
    TAGALOG_SPECIFIC_MODELS,
    run_classical_experiment,
    majority_class_floor,
    run_torch_experiment,
    run_transformer_experiment,
    load_precomputed_llm_predictions,
    add_precomputed_llm_results,
)

# ------------------------------------------------------------------
# Architecture registries (display name -> (arch key, model_kwargs))
# ------------------------------------------------------------------
CNN_RNN_MODELS = {
    "text_cnn": ("text_cnn", {}),
    "bigru": ("bigru", {}),
}

SCRATCH_ATTENTION_MODELS = {
    "single_head_attn": ("single_head_attn", {}),
    "multi_head_attn": ("multi_head_attn", {"num_heads": 4}),
    "tr_enc_L2_sin": ("transformer_encoder",
                      {"num_layers": 2, "num_heads": 4, "pos_encoding": "sinusoidal"}),
}

# Ablations run in EN->EN only (Act 1).
ABLATION_MODELS = {
    "abl_heads1": ("transformer_encoder", {"num_layers": 2, "num_heads": 1, "pos_encoding": "sinusoidal"}),
    "abl_heads2": ("transformer_encoder", {"num_layers": 2, "num_heads": 2, "pos_encoding": "sinusoidal"}),
    "abl_layers1": ("transformer_encoder", {"num_layers": 1, "num_heads": 4, "pos_encoding": "sinusoidal"}),
    "abl_layers4": ("transformer_encoder", {"num_layers": 4, "num_heads": 4, "pos_encoding": "sinusoidal"}),
    "abl_pos_learn": ("transformer_encoder", {"num_layers": 2, "num_heads": 4, "pos_encoding": "learnable"}),
    "abl_pos_none": ("transformer_encoder", {"num_layers": 2, "num_heads": 4, "pos_encoding": "none"}),
}


def run_all_experiments(df_all, store=None):
    """Run the full experiment matrix and return the populated store.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Combined EN+TL data.
    store : ResultStore, optional
        Store to populate. A fresh one is created if not given.

    Returns
    -------
    ResultStore
        Populated with every run's results and predictions.
    """
    store = store or ResultStore()

    for seed in config.SEEDS:
        for train_lang, test_lang in config.SETTINGS:
            is_en_en = (train_lang == "EN" and test_lang == "EN")
            expensive_ok = seed in config.EXPENSIVE_SEEDS

            if config.RUN_CLASSICAL:
                majority_class_floor(store, df_all, test_lang, seed, config.INPUT_VARIANT)
                for name, model in CLASSICAL_MODELS.items():
                    run_classical_experiment(store, df_all, train_lang, test_lang,
                                             name, model, config.INPUT_VARIANT, seed)

            if config.RUN_CNN_BIGRU:
                for name, (arch, kw) in CNN_RNN_MODELS.items():
                    run_torch_experiment(store, df_all, train_lang, test_lang,
                                         arch, name, config.INPUT_VARIANT, seed, kw)

            if config.RUN_SCRATCH_ATTENTION:
                for name, (arch, kw) in SCRATCH_ATTENTION_MODELS.items():
                    run_torch_experiment(store, df_all, train_lang, test_lang,
                                         arch, name, config.INPUT_VARIANT, seed, kw)

            if config.RUN_ABLATIONS and is_en_en:
                for name, (arch, kw) in ABLATION_MODELS.items():
                    run_torch_experiment(store, df_all, train_lang, test_lang,
                                         arch, name, config.INPUT_VARIANT, seed, kw)

            if config.RUN_TRANSFORMERS and expensive_ok:
                for name, hf_id in TRANSFORMER_MODELS.items():
                    run_transformer_experiment(store, df_all, train_lang, test_lang,
                                               name, hf_id, config.INPUT_VARIANT, seed)
                if config.RUN_TAGALOG_ROBERTA and train_lang == "TL" and test_lang == "TL":
                    for name, hf_id in TAGALOG_SPECIFIC_MODELS.items():
                        run_transformer_experiment(store, df_all, train_lang, test_lang,
                                                   name, hf_id, config.INPUT_VARIANT, seed)

    # Reference tier: precomputed LLM predictions (no API calls).
    llm_df = load_precomputed_llm_predictions(config.LLM_PREDICTIONS_PATH)
    add_precomputed_llm_results(store, llm_df)
    return store


def summarise(results_df):
    """Aggregate raw results to mean +/- std per model x setting.

    Returns
    -------
    pandas.DataFrame
        One row per (family, model, setting), sorted by mean macro-F1.
    """
    agg = {
        "mean_macro_f1": ("macro_f1", "mean"),
        "std_macro_f1": ("macro_f1", "std"),
        "mean_sarcasm_f1": ("sarcasm_f1", "mean"),
        "std_sarcasm_f1": ("sarcasm_f1", "std"),
        "mean_accuracy": ("accuracy", "mean"),
        "std_accuracy": ("accuracy", "std"),
        "n_runs": ("seed", "nunique"),
        "mean_params": ("n_params", "mean"),
    }
    if "train_time_s" in results_df.columns:
        agg["mean_train_time_s"] = ("train_time_s", "mean")
    return (
        results_df
        .groupby(["model_family", "model_name", "setting", "input_variant"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values("mean_macro_f1", ascending=False)
    )


def paired_compare(results_df, model_name, setting_a, setting_b, metric="macro_f1"):
    """Paired Wilcoxon across seeds comparing one model in two settings.

    Returns
    -------
    dict
        Means, delta (b - a), Wilcoxon p-value, pair count, and a note.
    """
    from scipy.stats import wilcoxon
    a = results_df[(results_df.model_name == model_name) & (results_df.setting == setting_a)][["seed", metric]]
    b = results_df[(results_df.model_name == model_name) & (results_df.setting == setting_b)][["seed", metric]]
    paired = pd.merge(a, b, on="seed", suffixes=("_a", "_b"))
    if len(paired) < 2:
        return {"model_name": model_name, "a": setting_a, "b": setting_b,
                "n_pairs": len(paired), "mean_a": np.nan, "mean_b": np.nan,
                "delta": np.nan, "p_value": np.nan, "note": "need >=2 seeds"}
    try:
        _, p = wilcoxon(paired[f"{metric}_b"], paired[f"{metric}_a"])
    except ValueError:
        p = np.nan
    return {"model_name": model_name, "a": setting_a, "b": setting_b,
            "n_pairs": len(paired),
            "mean_a": paired[f"{metric}_a"].mean(), "mean_b": paired[f"{metric}_b"].mean(),
            "delta": paired[f"{metric}_b"].mean() - paired[f"{metric}_a"].mean(),
            "p_value": p, "note": ""}


def build_stats_table(results_df):
    """Build paired transfer and fine-tuning comparisons for trained models.

    Returns
    -------
    pandas.DataFrame
        Two rows per trained model: EN->EN vs EN->TL (transfer cost) and
        EN->TL vs TL->TL (fine-tuning benefit).
    """
    trained = results_df[results_df["model_family"] != "llm_reference"]
    rows = []
    for name in sorted(trained["model_name"].unique()):
        rows.append(paired_compare(trained, name, "EN->EN", "EN->TL"))
        rows.append(paired_compare(trained, name, "EN->TL", "TL->TL"))
    return pd.DataFrame(rows)
