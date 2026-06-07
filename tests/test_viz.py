"""Unit tests for visualization helper return types."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.viz import plot_complexity_ladder, plot_transfer_collapse


def test_plot_complexity_ladder_returns_figure_and_axis():
    df = pd.DataFrame({
        "model_name": ["small", "large"],
        "mean_macro_f1": [0.55, 0.60],
        "std_macro_f1": [0.01, 0.02],
        "mean_params": [10, 100],
    })

    fig, ax = plot_complexity_ladder(df)

    assert fig is ax.figure
    assert ax.get_ylabel() == "Macro-F1"
    assert len(ax.patches) == 2
    plt.close(fig)


def test_plot_transfer_collapse_returns_grouped_bar_plot():
    df = pd.DataFrame({
        "model_name": ["toy", "toy", "other", "other"],
        "setting": ["EN->EN", "EN->TL", "EN->EN", "EN->TL"],
        "mean_macro_f1": [0.70, 0.50, 0.60, 0.55],
    })

    fig, ax = plot_transfer_collapse(df)

    assert fig is ax.figure
    assert ax.get_ylabel() == "Macro-F1"
    assert "Cross-Lingual Transfer" in ax.get_title()
    plt.close(fig)
