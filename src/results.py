"""Shared results store and the common result-row scaffold.

A single :class:`ResultStore` instance collects every run's summary row
and per-example predictions, so all runners write to the same place and
the experiment driver can serialise them at the end.
"""

import pandas as pd

from .metrics import make_run_name

# Maps an architecture key to its reporting family (used in tables/figures).
MODEL_FAMILY_BY_ARCH = {
    "text_cnn": "cnn_rnn",
    "bigru": "cnn_rnn",
    "single_head_attn": "scratch_transformer",
    "multi_head_attn": "scratch_transformer",
    "transformer_encoder": "scratch_transformer",
}


def base_result(train_lang, test_lang, model_name, model_family,
                input_variant, seed, n_train, n_test, **extra):
    """Build the common result-dict scaffold shared by all runners.

    The ``setting`` field is derived as ``f"{train_lang}->{test_lang}"``,
    which is what aligns every model (including the LLM reference) into
    the same setting buckets used by the RQ tables.

    Returns
    -------
    dict
        The base result row, updated with any ``extra`` metric fields.
    """
    return {
        "run_name": make_run_name(train_lang, test_lang, model_name, input_variant, seed),
        "model_family": model_family,
        "model_name": model_name,
        "train_lang": train_lang,
        "test_lang": test_lang,
        "setting": f"{train_lang}->{test_lang}",
        "input_variant": input_variant,
        "seed": seed,
        "n_train": n_train,
        "n_test": n_test,
        **extra,
    }


class ResultStore:
    """Collects result rows and per-example predictions across runs."""

    def __init__(self):
        self.results = []
        self.predictions = []

    def add(self, result, y_true, y_pred, test_df):
        """Append one result row plus its per-example prediction frame.

        Parameters
        ----------
        result : dict
            A row built by :func:`base_result` (plus metrics).
        y_true, y_pred : array-like of int
            Ground-truth and predicted labels for the test set.
        test_df : pandas.DataFrame
            The test frame (provides ``id`` and ``original_en_label``).

        Returns
        -------
        None
        """
        self.results.append(result)
        self.predictions.append(pd.DataFrame({
            "run_name": result["run_name"],
            "model_family": result["model_family"],
            "model_name": result["model_name"],
            "train_lang": result["train_lang"],
            "test_lang": result["test_lang"],
            "setting": result["setting"],
            "input_variant": result["input_variant"],
            "seed": result["seed"],
            "id": test_df["id"].values,
            "original_en_label": test_df["original_en_label"].values,
            "y_true": y_true,
            "y_pred": y_pred,
        }))

    def clear(self):
        """Reset both stores (used before a fresh experiment run)."""
        self.results.clear()
        self.predictions.clear()

    def results_frame(self):
        """Return all result rows as a single DataFrame."""
        return pd.DataFrame(self.results)

    def predictions_frame(self):
        """Return all per-example predictions concatenated, or empty."""
        if not self.predictions:
            return pd.DataFrame()
        return pd.concat(self.predictions, ignore_index=True)
