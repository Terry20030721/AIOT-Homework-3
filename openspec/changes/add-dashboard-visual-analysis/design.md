## Context

This dashboard is intended for local, interactive analysis and debugging of the Phase‑1 spam classifier. It is not a production inference service. The dataset is small so retraining on the fly is acceptable.

## Goals
- Provide immediate visual insights into label distribution and token-level signals
- Allow quick model re-runs to observe sensitivity to split/seed choices

## Decisions
- UI: Use Streamlit for rapid development and easy local serving. Use `matplotlib` (and optionally `seaborn`) for plotting. Keep visual elements simple and informative.
- Training: Use a scikit-learn Pipeline: `TfidfVectorizer` -> `LinearSVC`. Because `LinearSVC` lacks `predict_proba`, wrap with `CalibratedClassifierCV` to obtain calibrated probabilities for ROC/PR curves. Use `random_state=42` by default and allow user override.
- Token ranking: Use `CountVectorizer` or `TfidfVectorizer` to extract tokens; aggregate counts per class (sum of term counts over documents of the class) and show top N tokens per class. Offer option to switch to TF‑IDF weighting.
- Performance metrics: Use `sklearn.metrics.roc_curve`, `precision_recall_curve`, and `auc` / `average_precision_score` for summaries. Plot both ROC and PR curves on the dashboard.

## Implementation notes
- Use `train_test_split(df['text'], df['label'], test_size=0.2, random_state=seed, stratify=df['label'])` for deterministic splits and to preserve class balance.
- Use Streamlit caching for dataset load and optionally pipeline training to speed UI interaction (`st.cache_data` / `st.cache_resource` depending on Streamlit version).
- Do not overwrite `models/` artifacts; save internal retrained pipeline only in-memory or to a temp file if needed.

## Open Questions
- Default top‑N value? (Propose N=20)
- Use CalibratedClassifierCV with `cv=3` for calibration (costlier) or use `decision_function` as score for ROC/PR (faster). Default: use `CalibratedClassifierCV(cv=3)` for nicer probabilities.
