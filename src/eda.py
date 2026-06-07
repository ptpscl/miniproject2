"""Exploratory data analysis and inter-rater agreement.

Functions here return their key tables so they can be both displayed in
the notebook and asserted on in tests. ``display`` is used when running
inside Jupyter; it is imported defensively so the module also imports
cleanly in a plain Python / CI context.
"""

import pandas as pd

try:  # display exists in IPython/Jupyter; fall back to print elsewhere.
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print


def run_eda(df):
    """Print core EDA tables and return the label-shift summary.

    Tables shown: rows per language, class balance, utterance/context
    word-length statistics, and the label-shift table (how often the TL
    label differs from the original English label).

    Parameters
    ----------
    df : pandas.DataFrame
        Combined data frame with ``lang``, ``label`` and
        ``original_en_label`` columns.

    Returns
    -------
    pandas.DataFrame
        Per-language count and fraction of TL labels that differ from
        their original English label.
    """
    display(df.groupby("lang").size().reset_index(name="n_rows"))

    lab = df.groupby(["lang", "label"]).size().reset_index(name="count")
    lab["percent"] = lab.groupby("lang")["count"].transform(lambda x: 100 * x / x.sum())
    display(lab)

    df = df.copy()
    df["utt_wc"] = df["utterance"].apply(lambda x: len(str(x).split()))
    df["ctx_wc"] = df["context"].apply(lambda x: len(str(x).split()))
    display(df.groupby("lang")[["utt_wc", "ctx_wc"]].agg(["mean", "median", "min", "max"]))

    shift = (
        df[df["lang"] == "TL"]
        .assign(changed=lambda d: d["label"] != d["original_en_label"])
        .groupby("lang")["changed"].agg(["sum", "count", "mean"]).reset_index()
    )
    shift["percent_changed"] = 100 * shift["mean"]
    display(shift)
    return shift


def compute_fleiss_kappa(df, rater_cols):
    """Compute Fleiss' kappa across rater columns, if available.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame containing the per-rater label columns.
    rater_cols : list of str
        Column names holding each rater's categorical label.

    Returns
    -------
    float or None
        Fleiss' kappa, or None if the rater columns are not present.
    """
    if not rater_cols or not all(c in df.columns for c in rater_cols):
        print("Fleiss' kappa skipped: set RATER_COLS to your rater columns.")
        return None
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
    sub = df[rater_cols].dropna().astype(int).to_numpy()
    table, _ = aggregate_raters(sub)
    kappa = fleiss_kappa(table)
    print(f"Fleiss' kappa across {len(rater_cols)} raters: {kappa:.4f}")
    return kappa
