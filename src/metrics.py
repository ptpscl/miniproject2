"""Metrics, input-text construction, and small logging helpers.

These are pure functions with no heavy dependencies, which makes them
fast to unit-test and safe to import anywhere (including CI).
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def make_input_text(row, variant):
    """Build the model input string from a data row.

    Parameters
    ----------
    row : pandas.Series or dict
        Row containing ``'context'`` and ``'utterance'`` fields.
    variant : str
        Either ``'utterance_only'`` or ``'context_utterance'``.

    Returns
    -------
    text : str
        The constructed input string.

    Raises
    ------
    ValueError
        If ``variant`` is not a recognised option.
    """
    context = "" if pd.isna(row.get("context", "")) else str(row.get("context", ""))
    utterance = "" if pd.isna(row.get("utterance", "")) else str(row.get("utterance", ""))

    if variant == "utterance_only":
        return utterance
    if variant == "context_utterance":
        return f"[CONTEXT]\n{context}\n\n[UTTERANCE]\n{utterance}"
    raise ValueError(f"Unknown input variant: {variant}")


def add_input_column(df, variant):
    """Return a copy of ``df`` with a ``'text'`` column built per ``variant``.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``'context'`` and ``'utterance'`` columns.
    variant : str
        Input variant passed to :func:`make_input_text`.

    Returns
    -------
    pandas.DataFrame
        Copy with an added ``'text'`` column.
    """
    df = df.copy()
    df["text"] = df.apply(lambda r: make_input_text(r, variant), axis=1)
    return df


def compute_metrics(y_true, y_pred):
    """Compute accuracy plus macro and sarcasm-class precision/recall/F1.

    Assumes the positive (sarcastic) class is encoded as ``1``.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth labels.
    y_pred : array-like of int
        Predicted labels.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``macro_precision``, ``macro_recall``,
        ``macro_f1``, ``sarcasm_precision``, ``sarcasm_recall``,
        ``sarcasm_f1``.
    """
    acc = accuracy_score(y_true, y_pred)
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_bin, r_bin, f_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": p_mac,
        "macro_recall": r_mac,
        "macro_f1": f_mac,
        "sarcasm_precision": p_bin,
        "sarcasm_recall": r_bin,
        "sarcasm_f1": f_bin,
    }


def make_run_name(train_lang, test_lang, model_name, input_variant, seed):
    """Construct a unique, parseable run identifier.

    Returns
    -------
    str
        Formatted as ``'TRAIN->TEST__model__variant__seedN'``.
    """
    return f"{train_lang}->{test_lang}__{model_name}__{input_variant}__seed{seed}"


def format_seconds(seconds):
    """Format a duration in seconds as ``'MmSSs'`` for compact logging.

    Returns
    -------
    str
        e.g. ``'1m07s'`` or ``'0m42s'``.
    """
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def print_epoch_progress(epoch, n_epochs, train_loss, val_loss, val_macro_f1, elapsed):
    """Print a single-line training-progress update for one epoch.

    Lets you confirm a slow run is actually learning (validation macro-F1
    moving) rather than hung. Intended to be called once per epoch.

    Parameters
    ----------
    epoch : int
        Current epoch (1-indexed).
    n_epochs : int
        Maximum number of epochs.
    train_loss, val_loss : float
        Mean training and validation loss for the epoch.
    val_macro_f1 : float
        Validation macro-F1 for the epoch.
    elapsed : float
        Seconds elapsed since training start.

    Returns
    -------
    None
    """
    print(
        f"  epoch {epoch:>2}/{n_epochs} | "
        f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
        f"val_macroF1 {val_macro_f1:.4f} | {format_seconds(elapsed)}"
    )
