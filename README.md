# Understanding Transformers Through Complexity and Cost
### A Ladder of Architectures for English and Tagalog/Taglish Sarcasm Detection

![CI](https://github.com/ptpscl/miniproject2/actions/workflows/ci.yml/badge.svg)

Machine Learning 3, Mini Project 2 (MSDS 2026)

## Overview
This project frames sarcasm detection as binary text classification and
asks not just *whether* sarcasm can be detected, but *what it costs* to
detect it. We build an ordered ladder of architectures, from a TF-IDF
baseline, through from-scratch attention, to a fine-tuned multilingual
Transformer, capped by a zero-shot LLM reference, and measure macro-F1
against parameter count and training time at each rung. We then test
cross-lingual transfer between English and a human-validated
Tagalog/Taglish translation of MUStARD.

## Research Questions
- **RQ1:** How well do English-trained models transfer to Tagalog/Taglish?
- **RQ2:** Does target-language fine-tuning (TL to TL) beat transfer (EN to TL)?
- **RQ3:** Does added architectural complexity improve macro-F1, and at what cost?
- **RQ4:** How does the trained stack compare to a zero-shot LLM reference and to published MUStARD results?

## Repository Structure
```text
.
|-- .github/workflows/ci.yml   # lint and tests on every push
|-- data/                      # input data (EN, TL, precomputed LLM predictions)
|-- figures/                   # report figures generated from committed results
|-- results/                   # CSVs produced by the pipeline
|-- src/                       # all functions (importable package)
|-- tests/                     # pytest unit tests
|-- run_experiments.py         # training pipeline, writes results/
|-- report.ipynb               # narrative report, loads results/
|-- environment.yml            # pinned conda environment
`-- requirements.txt
```

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
export HF_TOKEN=...    # optional, only for gated models
```

## Reproducibility
The committed report reads existing CSV outputs from `results/`; opening
`report.ipynb` does not retrain the models. To regenerate the result
CSVs from the raw data, run:
```bash
python run_experiments.py
```

The full experiment can be slow because it includes multi-seed neural
and pretrained Transformer runs. Lightweight validation can be run with:
```bash
pytest -q
ruff check src tests
```

`environment.yml` is the full project environment for notebooks,
experiments, and report reproduction. GitHub Actions uses a lightweight
CPU test environment for linting and unit tests only; CI does not rerun
full training or recreate all experiment outputs.

The complexity ladder figure used in the report is saved in `figures/`.

## Tests
```bash
pytest -q
```
Tests cover metric correctness, train/test split leakage, experiment
aggregation helpers, EDA helper behavior, visualization return types, the
LLM loader's filter/relabel logic, and model forward-pass shapes. Model
tests auto-skip if PyTorch is not installed.

## Data
The English data is the text-only MUStARD sarcasm dataset. The
Tagalog/Taglish version was produced by machine-translating MUStARD and
then validating it with three human raters, judging for naturalness and
re-translating where needed. Split is by original conversation id, not
by row, to prevent cross-lingual leakage.

## Authors
Doria, Michelle Joanna
Inciso, Leonard Ray
Moran, Maria Patricia
Pascual, Ronald Patrick
