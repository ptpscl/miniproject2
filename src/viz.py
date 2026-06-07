"""Plotting helpers, styled to match the course's notebook conventions.

Each function returns ``(fig, ax)`` so figures can be displayed in the
report, saved to disk, or checked in a test without rendering to screen.
"""

import matplotlib.pyplot as plt


def plot_history(history):
    """Plot training/validation loss and validation macro-F1 over epochs.

    Parameters
    ----------
    history : dict
        Keys ``'train_loss'``, ``'val_loss'``, ``'val_macro_f1'``, each a
        list with one entry per epoch.

    Returns
    -------
    fig, axes : matplotlib Figure and ndarray of Axes
        The loss panel and the macro-F1 panel.
    """
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_f1 = history["val_macro_f1"]
    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].plot(epochs, train_loss, "o", color="tab:orange", label="Train Loss")
    axes[0].plot(epochs, val_loss, "--", color="tab:orange", label="Validation Loss")
    axes[1].plot(epochs, val_f1, "o", color="tab:blue", label="Validation Macro-F1")
    for ax in axes:
        ax.set_xlabel("Epochs")
        ax.legend()
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Macro-F1")
    fig.suptitle("Training History Plots", fontsize=16, weight="bold")
    return fig, axes


def plot_complexity_ladder(rq3_table):
    """Bar plot of EN->EN macro-F1 per model, ordered by parameter count.

    Parameters
    ----------
    rq3_table : pandas.DataFrame
        Must contain ``model_name``, ``mean_macro_f1``, ``std_macro_f1``,
        and ``mean_params``.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    t = rq3_table.sort_values("mean_params")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(t["model_name"], t["mean_macro_f1"], color="tab:blue")
    ax.errorbar(t["model_name"], t["mean_macro_f1"], yerr=t["std_macro_f1"].fillna(0),
                fmt="none", ecolor="black", capsize=3)
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Model (ascending parameters)")
    ax.set_title("Complexity vs. Payoff (EN->EN)", weight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig, ax


def plot_transfer_collapse(trained_summary):
    """Grouped bars of macro-F1 in EN->EN vs EN->TL per trained model.

    Parameters
    ----------
    trained_summary : pandas.DataFrame
        Summary table restricted to trained models (no LLM reference).

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    piv = (trained_summary[trained_summary["setting"].isin(["EN->EN", "EN->TL"])]
           .pivot_table(index="model_name", columns="setting", values="mean_macro_f1"))
    fig, ax = plt.subplots(figsize=(12, 5))
    piv.plot(kind="bar", ax=ax, color=["tab:blue", "tab:orange"])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Cross-Lingual Transfer Collapse", weight="bold")
    ax.axhline(0.5, ls="--", color="gray", label="chance")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    return fig, ax
