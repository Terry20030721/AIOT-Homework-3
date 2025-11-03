## ADDED Requirements

### Requirement: Spam classification baseline (TF-IDF + SVM)
The system SHALL provide a reproducible training pipeline to build a spam/ham classifier using TF-IDF feature extraction and an SVM classifier. The pipeline SHALL output serialized artifacts into `models/` for later use by a prediction CLI.

#### Scenario: Dataset downloaded and preprocessed
- **WHEN** the developer runs the download and preprocessing steps described in `openspec/changes/add-spam-classification-svm/tasks.md`
- **THEN** the raw CSV file is saved to `datasets/raw/sms_spam_no_header.csv` and `datasets/processed/cleaned.csv` is produced containing columns `label` and `text` with normalized text values.

#### Scenario: Model training produces artifacts
- **WHEN** the developer runs `scripts/train.py` on `datasets/processed/cleaned.csv`
- **THEN** a scikit-learn pipeline (TF-IDF + SVM) is trained, evaluation metrics are printed (e.g., cross-validation accuracy, precision/recall), and `models/model.joblib` and `models/vectorizer.joblib` (or a single `models/pipeline.joblib`) are written.

#### Scenario: Prediction CLI returns expected labels
- **WHEN** the developer runs `scripts/predict.py "<text>"` using the saved artifacts
- **THEN** the CLI SHALL output a deterministic predicted label of either `spam` or `ham` and exit with code 0. If model files are missing, the CLI SHALL return an error message and exit non-zero.

#### Scenario: Malformed input handled
- **WHEN** the CLI receives an empty string or binary input
- **THEN** the CLI SHALL print a clear error message and exit with a non-zero code; it SHALL NOT crash with an unhandled exception.
