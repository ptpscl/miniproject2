"""Data loading, leakage-safe splitting, and dataset assembly.

The split is performed on the original conversation ``id`` rather than on
rows, which prevents the same item leaking across train and test in
different languages (the EN and TL copies share an id).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .metrics import add_input_column


def load_one_language(lang, path):
    """Load and normalise one language's MUStARD file.

    Renames the language-specific columns to a common schema
    (``id``, ``context``, ``utterance``, ``label``) and tags the language.

    Parameters
    ----------
    lang : str
        ``'EN'`` or ``'TL'``.
    path : str
        Path to the Excel file.

    Returns
    -------
    pandas.DataFrame
        Columns: ``id``, ``lang``, ``context``, ``utterance``, ``label``.

    Raises
    ------
    ValueError
        If the file is missing any expected column.
    """
    cfg = config.LANG_CONFIG[lang]
    df = pd.read_excel(path, sheet_name=cfg["sheet_name"])

    expected = [cfg["id_col"], cfg["context_col"], cfg["utterance_col"], cfg["label_col"]]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"{lang} file missing columns: {missing}\nFound: {df.columns.tolist()}"
        )

    df = df.rename(columns={
        cfg["id_col"]: "id",
        cfg["context_col"]: "context",
        cfg["utterance_col"]: "utterance",
        cfg["label_col"]: "label",
    })
    df = df[["id", "context", "utterance", "label"]].copy()
    df["id"] = df["id"].astype(str)
    df["context"] = df["context"].fillna("").astype(str)
    df["utterance"] = df["utterance"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    df["lang"] = lang
    return df[["id", "lang", "context", "utterance", "label"]]


def load_all_data(paths=None, smoke_test=None, smoke_n_ids=None):
    """Load EN and TL files and assemble the combined frame.

    Aligns each TL row to its original English label so that label shift
    (sarcasm changing under translation) is measurable downstream.

    Parameters
    ----------
    paths : dict, optional
        Mapping ``{'EN': path, 'TL': path}``. Defaults to ``config.PATHS``.
    smoke_test : bool, optional
        If True, keep only the first ``smoke_n_ids`` ids. Defaults to
        ``config.SMOKE_TEST``.
    smoke_n_ids : int, optional
        Number of ids to keep in smoke mode. Defaults to
        ``config.SMOKE_N_IDS``.

    Returns
    -------
    pandas.DataFrame
        Combined frame with an added ``original_en_label`` column.
    """
    paths = paths or config.PATHS
    smoke_test = config.SMOKE_TEST if smoke_test is None else smoke_test
    smoke_n_ids = config.SMOKE_N_IDS if smoke_n_ids is None else smoke_n_ids

    df_en = load_one_language("EN", paths["EN"])
    english_label_map = dict(zip(df_en["id"], df_en["label"]))

    df_tl = load_one_language("TL", paths["TL"])
    df_tl["original_en_label"] = df_tl["id"].map(english_label_map).astype("Int64")

    df_en["original_en_label"] = df_en["label"]
    df_all = pd.concat([df_en, df_tl], ignore_index=True)
    df_all["original_en_label"] = df_all["original_en_label"].astype(int)

    if smoke_test:
        keep_ids = sorted(df_all["id"].unique())[:smoke_n_ids]
        df_all = df_all[df_all["id"].isin(keep_ids)].copy()

    return df_all


def get_train_test_ids(df, seed, test_size=0.30):
    """Split original ids (not rows) into train/test, stratified on label.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined frame containing an ``'EN'`` subset to define the split.
    seed : int
        Random seed for the split.
    test_size : float, optional
        Fraction held out for test. Default 0.30.

    Returns
    -------
    train_ids, test_ids : set of str
        Disjoint sets of conversation ids.
    """
    base = (
        df[df["lang"] == "EN"][["id", "label"]]
        .drop_duplicates("id")
        .sort_values("id")
    )
    stratify = base["label"] if base["label"].nunique() > 1 and len(base) >= 10 else None
    train_ids, test_ids = train_test_split(
        base["id"], test_size=test_size, random_state=seed, stratify=stratify
    )
    return set(train_ids), set(test_ids)


def subset_lang_ids(df, lang, ids):
    """Return rows for a given language whose id is in ``ids``."""
    return df[(df["lang"] == lang) & (df["id"].isin(ids))].copy()


def build_train_test_data(df, train_lang, test_lang, seed, input_variant):
    """Build train/test frames with a ``'text'`` column for one setting.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined data frame.
    train_lang, test_lang : str
        Language codes for the train and test sides of the setting.
    seed : int
        Seed controlling the id split.
    input_variant : str
        Passed through to :func:`make_input_text`.

    Returns
    -------
    train_df, test_df : pandas.DataFrame
        Reset-index frames, each carrying the constructed ``'text'`` column.
    """
    train_ids, test_ids = get_train_test_ids(df, seed)
    train_df = subset_lang_ids(df, train_lang, train_ids)
    test_df = subset_lang_ids(df, test_lang, test_ids)
    train_df = add_input_column(train_df, input_variant)
    test_df = add_input_column(test_df, input_variant)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
