"""Experiment runners for each model family.

All runners write into a shared :class:`~src.results.ResultStore`. Heavy
libraries (torch, transformers) are imported lazily inside the functions
that need them so this module stays importable on a CPU-only machine.
"""

import os
import time
import tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from . import config
from .runtime import DEVICE, set_all_seeds, clear_memory
from .metrics import compute_metrics, print_epoch_progress
from .data import build_train_test_data
from .results import base_result, MODEL_FAMILY_BY_ARCH


# ------------------------------------------------------------------
# Rung 1: classical + chance floor
# ------------------------------------------------------------------
CLASSICAL_MODELS = {
    "tfidf_logreg": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    "tfidf_svm": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LinearSVC(class_weight="balanced")),
    ]),
}


def run_classical_experiment(store, df, train_lang, test_lang, model_name, model,
                             input_variant, seed):
    """Train and evaluate a classical sklearn pipeline; log the result.

    The classical "complexity" proxy is the fitted TF-IDF vocabulary size
    (number of features), recorded as ``n_params`` so the EN->EN ladder
    figure has a meaningful x-axis value for these models.

    Returns
    -------
    None
    """
    set_all_seeds(seed)
    train_df, test_df = build_train_test_data(df, train_lang, test_lang, seed, input_variant)

    t0 = time.time()
    model.fit(train_df["text"].tolist(), train_df["label"].values)
    train_time = round(time.time() - t0, 4)

    y_pred = model.predict(test_df["text"].tolist())
    y_true = test_df["label"].values
    metrics = compute_metrics(y_true, y_pred)

    # Feature count as a meaningful complexity proxy for classical models.
    try:
        n_features = len(model.named_steps["tfidf"].vocabulary_)
    except Exception:
        n_features = 0

    result = base_result(
        train_lang, test_lang, model_name, "classical",
        input_variant, seed, len(train_df), len(test_df),
        n_params=n_features, train_time_s=train_time, **metrics,
    )
    store.add(result, y_true, y_pred, test_df)


def majority_class_floor(store, df, test_lang, seed, input_variant):
    """Log a majority-class chance baseline (predicts the training mode)."""
    set_all_seeds(seed)
    train_df, test_df = build_train_test_data(df, test_lang, test_lang, seed, input_variant)
    maj = int(train_df["label"].mode().iloc[0])
    y_true = test_df["label"].values
    y_pred = np.full_like(y_true, maj)
    metrics = compute_metrics(y_true, y_pred)
    result = base_result(
        test_lang, test_lang, "majority_floor", "baseline",
        input_variant, seed, len(train_df), len(test_df),
        n_params=0, **metrics,
    )
    store.add(result, y_true, y_pred, test_df)


# ------------------------------------------------------------------
# Rungs 2-6: torch models (CNN, BiGRU, attention ladder)
# ------------------------------------------------------------------
def _torch_predict(model, loader):
    """Run inference over a loader; return ``(y_true, y_pred)`` arrays."""
    import torch
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(DEVICE))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(y.numpy())
    return np.array(trues), np.array(preds)


def run_torch_experiment(store, df, train_lang, test_lang, arch, model_name,
                         input_variant, seed, model_kwargs=None, verbose=True):
    """Train a torch model with validation-based early stopping; log result.

    Uses a stratified validation split carved from train, Adam with weight
    decay, dropout (inside the model), and early stopping on validation
    macro-F1. Returns the per-epoch history for plotting.

    Returns
    -------
    dict
        History with keys ``train_loss``, ``val_loss``, ``val_macro_f1``.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from .models import SimpleVocab, TorchTextDataset, build_torch_model

    set_all_seeds(seed)
    clear_memory()
    model_kwargs = dict(model_kwargs or {})

    train_df, test_df = build_train_test_data(df, train_lang, test_lang, seed, input_variant)

    strat = train_df["label"] if train_df["label"].nunique() > 1 else None
    tr_df, val_df = train_test_split(
        train_df, test_size=config.VAL_FRACTION, random_state=seed, stratify=strat
    )

    vocab = SimpleVocab(min_freq=1)
    vocab.fit(tr_df["text"].tolist())

    max_len = config.MAX_LENGTH
    bs = config.TORCH_BATCH_SIZE
    train_loader = DataLoader(
        TorchTextDataset(tr_df["text"].tolist(), tr_df["label"].tolist(), vocab, max_len),
        batch_size=bs, shuffle=True)
    val_loader = DataLoader(
        TorchTextDataset(val_df["text"].tolist(), val_df["label"].tolist(), vocab, max_len),
        batch_size=bs)
    test_loader = DataLoader(
        TorchTextDataset(test_df["text"].tolist(), test_df["label"].tolist(), vocab, max_len),
        batch_size=bs)

    model = build_torch_model(arch, len(vocab.itos), **model_kwargs).to(DEVICE)
    opt = torch.optim.Adam(
        model.parameters(), lr=config.TORCH_LR, weight_decay=config.TORCH_WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1, best_state, patience, start = -1.0, None, 0, time.time()

    for epoch in range(1, config.NUM_EPOCHS_TORCH + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        train_loss = ep_loss / max(nb, 1)

        model.eval()
        vl, vnb, vp, vt = 0.0, 0, [], []
        with torch.no_grad():
            for X, y in val_loader:
                logits = model(X.to(DEVICE))
                vl += loss_fn(logits, y.to(DEVICE)).item()
                vnb += 1
                vp.extend(torch.argmax(logits, 1).cpu().numpy())
                vt.extend(y.numpy())
        val_loss = vl / max(vnb, 1)
        val_f1 = compute_metrics(np.array(vt), np.array(vp))["macro_f1"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        if verbose:
            print_epoch_progress(
                epoch, config.NUM_EPOCHS_TORCH, train_loss, val_loss, val_f1,
                time.time() - start,
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred = _torch_predict(model, test_loader)
    metrics = compute_metrics(y_true, y_pred)
    n_params = sum(p.numel() for p in model.parameters())

    result = base_result(
        train_lang, test_lang, model_name, MODEL_FAMILY_BY_ARCH[arch],
        input_variant, seed, len(tr_df), len(test_df),
        n_params=n_params, best_val_macro_f1=best_f1,
        train_time_s=round(time.time() - start, 2),
        num_heads=model_kwargs.get("num_heads"),
        num_layers=model_kwargs.get("num_layers"),
        pos_encoding=model_kwargs.get("pos_encoding"),
        **metrics,
    )
    store.add(result, y_true, y_pred, test_df)

    del model, train_loader, val_loader, test_loader
    clear_memory()
    return history


# ------------------------------------------------------------------
# Rung 7: pretrained HuggingFace transformers
# ------------------------------------------------------------------
TRANSFORMER_MODELS = {
    "distilbert_multi": "distilbert-base-multilingual-cased",
    "mbert": "bert-base-multilingual-cased",
    "xlmr_base": "xlm-roberta-base",
}
TAGALOG_SPECIFIC_MODELS = {
    "tagalog_roberta": "jcblaise/roberta-tagalog-base",
}


def run_transformer_experiment(store, df, train_lang, test_lang, model_name,
                               hf_model_id, input_variant, seed):
    """Fine-tune and evaluate a pretrained HF classifier; log the result.

    The HuggingFace access token is read from the ``HF_TOKEN`` environment
    variable (``None`` if unset, which is fine for public models).
    Tagalog-specific models are skipped unless the setting is TL->TL.

    Returns
    -------
    None
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    if model_name == "tagalog_roberta" and not (train_lang == "TL" and test_lang == "TL"):
        print(f"Skipping {model_name} for {train_lang}->{test_lang} (TL->TL only).")
        return

    hf_token = os.environ.get("HF_TOKEN")

    set_all_seeds(seed)
    clear_memory()
    train_df, test_df = build_train_test_data(df, train_lang, test_lang, seed, input_variant)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id, num_labels=2, token=hf_token
    )

    class HFTextDataset(Dataset):
        """Tokenised dataset for HuggingFace sequence classification."""

        def __init__(self, texts, labels, tok, max_length):
            self.enc = tok(texts, truncation=True, padding=True, max_length=max_length)
            self.labels = list(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(int(self.labels[idx]))
            return item

    train_ds = HFTextDataset(train_df["text"].tolist(), train_df["label"].tolist(),
                             tokenizer, config.MAX_LENGTH)
    test_ds = HFTextDataset(test_df["text"].tolist(), test_df["label"].tolist(),
                            tokenizer, config.MAX_LENGTH)

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=config.NUM_EPOCHS_TRANSFORMER,
            per_device_train_batch_size=config.BATCH_SIZE,
            per_device_eval_batch_size=config.BATCH_SIZE,
            learning_rate=config.LEARNING_RATE,
            weight_decay=0.01,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            seed=seed,
            fp16=torch.cuda.is_available(),
        )
        trainer = Trainer(model=model, args=args, train_dataset=train_ds)
        trainer.train()
        raw = trainer.predict(test_ds)

    y_pred = np.argmax(raw.predictions, axis=1)
    y_true = test_df["label"].values
    metrics = compute_metrics(y_true, y_pred)
    n_params = sum(p.numel() for p in model.parameters())

    result = base_result(
        train_lang, test_lang, model_name, "transformer",
        input_variant, seed, len(train_df), len(test_df),
        n_params=n_params, train_time_s=round(time.time() - start, 2), **metrics,
    )
    store.add(result, y_true, y_pred, test_df)

    del trainer, model, tokenizer, train_ds, test_ds
    clear_memory()


# ------------------------------------------------------------------
# Reference tier: precomputed LLM-as-judge predictions (no API calls)
# ------------------------------------------------------------------
def load_precomputed_llm_predictions(path, keep_langs=("EN", "TL")):
    """Load cached LLM-as-judge predictions and keep only study languages.

    The cached file may contain extra languages (e.g. DE/ES/VI from a
    larger experiment); these are filtered out so only the study's
    languages remain.

    Parameters
    ----------
    path : str
        Path to the predictions CSV.
    keep_langs : tuple of str
        Languages to retain. Defaults to ``("EN", "TL")``.

    Returns
    -------
    pandas.DataFrame
        Filtered, type-normalised predictions.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or labels are not binary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM predictions file not found: {path}")

    llm_df = pd.read_csv(path)
    required = {
        "id", "lang", "input_variant", "y_true", "y_pred",
        "model_name", "provider", "seed", "original_en_label",
    }
    missing = required - set(llm_df.columns)
    if missing:
        raise ValueError(
            f"LLM predictions missing columns: {sorted(missing)}\n"
            f"Found: {list(llm_df.columns)}"
        )

    llm_df = llm_df.copy()
    llm_df["id"] = llm_df["id"].astype(str)
    llm_df["lang"] = llm_df["lang"].astype(str)
    llm_df["input_variant"] = llm_df["input_variant"].astype(str)
    llm_df["y_true"] = llm_df["y_true"].astype(int)
    llm_df["y_pred"] = llm_df["y_pred"].astype(int)
    llm_df["seed"] = llm_df["seed"].astype(int)
    llm_df["original_en_label"] = llm_df["original_en_label"].astype(int)

    bad_true = sorted(set(llm_df["y_true"]) - {0, 1})
    bad_pred = sorted(set(llm_df["y_pred"]) - {0, 1})
    if bad_true or bad_pred:
        raise ValueError(f"Labels must be 0/1. Bad y_true={bad_true}, y_pred={bad_pred}")

    # Keep only the study's languages (drops DE/ES/VI etc.).
    llm_df = llm_df[llm_df["lang"].isin(list(keep_langs))].copy()
    return llm_df


def add_precomputed_llm_results(store, llm_df):
    """Append precomputed LLM results to the store as a reference tier.

    Each LLM row is labelled with ``train_lang == test_lang == lang`` so
    its ``setting`` becomes ``EN->EN`` / ``TL->TL`` and aligns with the
    in-language buckets used by the RQ tables (the LLM judges in-language;
    it performs no cross-lingual transfer).

    Returns
    -------
    None
    """
    group_cols = ["lang", "input_variant", "seed", "provider", "model_name"]
    for (lang, input_variant, seed, provider, model_name), sub in llm_df.groupby(group_cols):
        sub = sub.copy().reset_index(drop=True)
        y_true = sub["y_true"].values
        y_pred = sub["y_pred"].values
        metrics = compute_metrics(y_true, y_pred)

        clean_name = f"{provider}_{model_name}".replace("/", "-").replace(" ", "_")
        result = base_result(
            train_lang=lang, test_lang=lang, model_name=clean_name,
            model_family="llm_reference", input_variant=input_variant,
            seed=int(seed), n_train=0, n_test=len(sub),
            n_params=np.nan, llm_model=model_name, provider=provider,
            prompt_version=(sub["prompt_version"].iloc[0]
                            if "prompt_version" in sub.columns else np.nan),
            latency_s=np.nan, latency_per_item_s=np.nan, **metrics,
        )
        store.add(result, y_true, y_pred, sub)
