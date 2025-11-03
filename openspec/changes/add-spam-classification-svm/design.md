## Context

Phase-1 is a minimal, reproducible ML baseline to demonstrate end-to-end training and inference. The training will use a local CSV as dataset and produce joblib-serialized artifacts. This is not intended as a production-grade service.

## Goals
- Reproducible training with deterministic results (set random seed)
- Minimal dependencies and clear instructions for local runs

## Decisions
- Pipeline: `TfidfVectorizer` -> `LinearSVC` (scikit-learn). Use `Pipeline` so both vectorizer and model can be saved as a single artifact (recommended: `joblib.dump(pipeline, 'models/pipeline.joblib')`).
- Data format: Input CSV (no header) will be downloaded to `datasets/raw/` and preprocessed to `datasets/processed/cleaned.csv` with columns `label` (`spam`/`ham`) and `text` (string).
- Persistence: Use `joblib` for serializing the fitted pipeline.
- CLI: `scripts/predict.py` will use `argparse` or `click` to support single-string predictions and a small `--stdin` mode to read lines.

## Reproducibility
- Set a fixed random seed in `scripts/train.py` (e.g., `random_state=42`) and pin scikit-learn version in `requirements.txt`.

## Open Questions
- Do you want a single combined `pipeline.joblib` (vectorizer + model) or separate files for vectorizer and model? (Default: combined pipeline.)
