## Why
We need a lightweight, reproducible spam classification pipeline to demonstrate end-to-end ML workflows in the project. Phase-1 will provide a baseline offline training pipeline (TF-IDF + SVM), model artifact storage, and a simple CLI predictor so other tasks (integration, deployment, evaluation) can build on a working example.

## What Changes
- Add dataset download and storage under `datasets/raw/`.
- Add preprocessing script `scripts/preprocess.py` that produces `datasets/processed/cleaned.csv`.
- Add training script `scripts/train.py` to build a scikit-learn pipeline (TF-IDF + SVM), evaluate it, and save model artifacts to `models/`.
- Add prediction CLI `scripts/predict.py` that loads saved artifacts and classifies input text.
- Add `requirements.txt` and `README.md` describing usage and reproducibility steps.

## Impact
- Affected specs: `ingestion` (data), `ml` (model training/prediction)
- Affected files (new):
  - `datasets/raw/sms_spam_no_header.csv`
  - `datasets/processed/cleaned.csv`
  - `scripts/preprocess.py`
  - `scripts/train.py`
  - `scripts/predict.py`
  - `models/` (artifact output)
  - `requirements.txt`
  - `README.md`

## Migration
- No migration of production data required for Phase-1; this is an additive demo feature. If a production DB exists later, adapt `scripts/train.py` to read from canonical data stores.

## Acceptance criteria
- `datasets/raw/sms_spam_no_header.csv` exists in `datasets/raw/` (download step succeeded).
- `datasets/processed/cleaned.csv` is produced by `scripts/preprocess.py` and contains normalized text and labels.
- `scripts/train.py` trains a TF-IDF + SVM pipeline and writes model artifacts (e.g., `models/model.joblib`, `models/vectorizer.joblib`).
- `scripts/predict.py` loads artifacts and returns `spam`/`ham` predictions for CLI input.
- `requirements.txt` lists the Python dependencies to run the scripts.
- `README.md` describes how to run preprocess, train and predict locally.

## Risks
- Dataset licensing: the dataset used is derived from a public repo; confirm acceptable use for this demo.
- Model quality: SVM baseline may not reach production-grade metrics; this is accepted for Phase-1 as a demo baseline.

## Timeline (Phase-1)
- 1 day: download dataset and implement preprocessing
- 1 day: implement training pipeline and save artifacts
- 0.5 day: implement predict CLI and write README + requirements
