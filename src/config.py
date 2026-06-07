"""Central configuration for the MP2 sarcasm-detection project.

All tunable constants, file paths, and experiment switches live here so
that every other module and the report notebook import a single source
of truth. Paths are resolved relative to the repository root, assuming
the committed ``data/`` folder layout::

    <repo>/
        data/
            English_MS.xlsx
            Tagalog.xlsx
            llm_predictions_all_languages.csv
        results/
        src/
        tests/
"""

import os

# ------------------------------------------------------------------
# Repository-relative paths
# ------------------------------------------------------------------
# This file lives at <repo>/src/config.py, so the repo root is one
# directory up from here. Resolving paths this way means the code works
# identically on Colab, CI, and a teammate's laptop without edits.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# ------------------------------------------------------------------
# Languages
# ------------------------------------------------------------------
SOURCE_LANG = "EN"
TARGET_LANG = "TL"

LANG_NAMES = {
    "EN": "English",
    "TL": "Tagalog/Taglish",
}

# ------------------------------------------------------------------
# Data files (committed under data/)
# ------------------------------------------------------------------
PATHS = {
    "EN": os.path.join(DATA_DIR, "English_MS.xlsx"),
    "TL": os.path.join(DATA_DIR, "Tagalog.xlsx"),
}

# Precomputed LLM-as-judge predictions (cached so no API calls are made).
LLM_PREDICTIONS_PATH = os.path.join(DATA_DIR, "llm_predictions_all_languages.csv")

# ------------------------------------------------------------------
# Column mapping per language file
# ------------------------------------------------------------------
LANG_CONFIG = {
    "EN": {
        "sheet_name": "English MUStARD",
        "id_col": "original_id",
        "context_col": "context",
        "utterance_col": "utterance",
        "label_col": "original_en_label",
    },
    "TL": {
        "sheet_name": "Sheet1",
        "id_col": "original_id",
        "context_col": "context_tl",
        "utterance_col": "utterance_tl",
        "label_col": "sarcasm_tl",
    },
}

# Per-rater label columns in the TL file (for Fleiss' kappa). Leave empty
# if the rater columns are not present in the committed data.
RATER_COLS = []

# ------------------------------------------------------------------
# Smoke test: keep True for quick end-to-end checks; False for the
# full multi-seed run that produces the reported results.
# ------------------------------------------------------------------
SMOKE_TEST = False
SMOKE_N_IDS = 10

SEEDS = [13, 21, 42, 87, 101, 123, 202, 303, 404, 505]
if SMOKE_TEST:
    SEEDS = [42]

# Pretrained transformers are the expensive rung; in the final run they
# use the full seed list for statistical power.
EXPENSIVE_SEEDS = SEEDS if not SMOKE_TEST else [42]

# ------------------------------------------------------------------
# Model-family switches
# ------------------------------------------------------------------
RUN_CLASSICAL = True
RUN_CNN_BIGRU = True
RUN_SCRATCH_ATTENTION = True   # rungs 4-6
RUN_TRANSFORMERS = True        # rung 7 (pretrained HF)
RUN_TAGALOG_ROBERTA = True     # TL->TL only
RUN_ABLATIONS = True           # EN->EN only
# Live LLM API calls are disabled; precomputed predictions are loaded.
RUN_LLM_JUDGE = False

# ------------------------------------------------------------------
# Training settings
# ------------------------------------------------------------------
MAX_LENGTH = 192

BATCH_SIZE = 4 if SMOKE_TEST else 8
NUM_EPOCHS_TRANSFORMER = 1 if SMOKE_TEST else 3
LEARNING_RATE = 2e-5

TORCH_BATCH_SIZE = 4 if SMOKE_TEST else 16
NUM_EPOCHS_TORCH = 1 if SMOKE_TEST else 15
TORCH_LR = 1e-3
TORCH_WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3
VAL_FRACTION = 0.15

# ------------------------------------------------------------------
# Input construction
# ------------------------------------------------------------------
INPUT_VARIANT = "context_utterance"   # or "utterance_only"

# ------------------------------------------------------------------
# LLM-as-judge (documentation only; predictions are precomputed)
# ------------------------------------------------------------------
LLM_MODEL = "gpt-4o-mini"
LLM_JUDGE_SEED = 42

# ------------------------------------------------------------------
# Cross-lingual settings for trained models
# ------------------------------------------------------------------
SETTINGS = [
    ("EN", "EN"),
    ("EN", "TL"),
    ("TL", "TL"),
]
