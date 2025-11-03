## 1. Implementation
- [x] 1.1 Create `datasets/raw/` and download `sms_spam_no_header.csv`
- [x] 1.2 Implement `scripts/preprocess.py` to read raw CSV, clean text (lowercase, remove punctuation, normalize whitespace), and write `datasets/processed/cleaned.csv` with columns `label,text`.
- [x] 1.3 Implement `scripts/train.py` to build a scikit-learn Pipeline: `TfidfVectorizer` -> `LinearSVC` (or `SVC(kernel='linear')`), evaluate with cross-validation, and persist artifacts to `models/` using `joblib`.
- [x] 1.4 Implement `scripts/predict.py` CLI that loads `models/` artifacts and predicts label for supplied text.
- [x] 1.5 Create `requirements.txt` pinning necessary packages (pandas, scikit-learn, joblib, click or argparse)
- [x] 1.6 Create `README.md` with clear run instructions for preprocess / train / predict.
- [ ] 1.7 Add unit tests for preprocessing and a smoke test for prediction (optional: pytest fixture using a tiny sample CSV).
- [ ] 1.8 (Optional) Add a simple GitHub Actions workflow to run lints and tests on PRs.

## 2. Validation
- [x] 2.1 Run `scripts/preprocess.py` and verify `datasets/processed/cleaned.csv` exists and has expected columns
- [x] 2.2 Run `scripts/train.py` and confirm `models/pipeline.joblib` is created (pipeline contains vectorizer+model)
- [x] 2.3 Run `scripts/predict.py "some text"` and confirm output is `spam` or `ham`

## 3. Notes
- Keep all code minimal and well-documented; favor clarity for homework/demo use.
