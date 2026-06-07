# Understanding Transformers Through Complexity and Cost
### A Ladder of Architectures for English and Tagalog/Taglish Sarcasm Detection

![CI](https://github.com/{{ORG_OR_USER}}/{{REPO}}/actions/workflows/ci.yml/badge.svg)

Machine Learning 3 — Mini Project 2 (MSDS 2026)

## Overview
This project frames sarcasm detection as binary text classification and
asks not just *whether* sarcasm can be detected, but *what it costs* to
detect it. We build an ordered ladder of architectures — from a TF-IDF
baseline, through from-scratch attention, to a fine-tuned multilingual
Transformer, capped by a zero-shot LLM reference — and measure macro-F1
against parameter count and training time at each rung. We then test
cross-lingual transfer between English and a human-validated
Tagalog/Taglish translation of MUStARD.

## Research Questions
- **RQ1** — How well do English-trained models transfer to Tagalog/Taglish?
- **RQ2** — Does target-language fine-tuning (TL→TL) beat transfer (EN→TL)?
- **RQ3** — Does added architectural complexity improve macro-F1, and at what cost?
- **RQ4** — How does the trained stack compare to a zero-shot LLM reference and to published MUStARD results?

## Repository Structure
.
├── .github/workflows/ci.yml   # lint + tests on every push
├── data/                      # input data (EN, TL, precomputed LLM preds)
├── results/                   # CSVs produced by the pipeline
├── src/                       # all functions (importable package)
├── tests/                     # pytest unit tests
├── run_experiments.py         # training pipeline -> writes results/
├── report.ipynb               # narrative report (loads results/)
├── environment.yml            # pinned conda environment
└── requirements.txt

## Setup
```bash
conda env create -f environment.yml
conda activate mp2-sarcasm
```

Place the three data files in `data/`:
- `English_MS.xlsx`
- `Tagalog.xlsx`
- `llm_predictions_all_languages.csv`

For the pretrained models, set a HuggingFace token if needed:
```bash
export HF_TOKEN=...    # optional; only for gated models
```

## How to Run
Run the full experiment pipeline (produces the CSVs in `results/`):
```bash
python run_experiments.py
```
Then open `report.ipynb`, which loads those CSVs and presents the
analysis. The report does not retrain models.

## Tests
```bash
pytest -q
```
Tests cover metric correctness, train/test split leakage, the LLM
loader's filter/relabel, and model forward-pass shapes. Model tests
auto-skip if PyTorch is not installed.

## Data
The English data is the text-only MUStARD sarcasm dataset. The
Tagalog/Taglish version was produced by machine-translating MUStARD and
then validating it with three human raters, judging for naturalness and
re-translating where needed. Split is by original conversation id (not
by row) to prevent cross-lingual leakage.

## Authors
{{MEMBER NAMES}}