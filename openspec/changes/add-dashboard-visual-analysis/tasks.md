## 1. Implementation
- [ ] 1.1 Add `scripts/dashboard.py` (Streamlit app) that can be started via `streamlit run scripts/dashboard.py`.
- [ ] 1.2 Update `requirements.txt` to include `streamlit` and `matplotlib` (optional: `seaborn`).
- [ ] 1.3 Dashboard reads `datasets/processed/cleaned.csv` automatically; show friendly error if missing.
- [ ] 1.4 Dashboard performs internal train/test split (default 80/20) with fixed `random_state`.
- [ ] 1.5 Dashboard trains a TF‑IDF + SVM pipeline internally (use `LinearSVC` + `CalibratedClassifierCV` to obtain probabilities) for analysis only.
- [ ] 1.6 Implement Data Overview chart (spam vs ham count bar chart).
- [ ] 1.7 Implement Top Tokens by Class (show top N tokens per class, default N=20).
- [ ] 1.8 Implement Model Performance (Test): compute ROC curve and Precision‑Recall curve on test set and plot curves with AUC / average precision values.
- [ ] 1.9 Add minimal UI controls: top‑N selector, test split ratio slider, random seed input, and a refresh / retrain button.
- [ ] 1.10 Add small text descriptions and usage instructions in the Streamlit UI and README.

Note: Items 1.3–1.8 MUST run automatically on app startup — the dashboard shall load the dataset and display the three charts immediately without requiring the user to click a "run" or "train" button. UI controls (1.9) are optional for tuning but shall not be required to trigger the initial display.

## 2. Validation
- [ ] 2.1 Run `streamlit run scripts/dashboard.py` and confirm app starts without errors.
- [ ] 2.2 Verify the three charts render and reflect dataset values.
- [ ] 2.3 Verify retrain with different seeds / split ratios updates model curves.

## 3. Notes
- Keep dashboard self-contained for local analysis; it retrains for visualization and does not overwrite `models/` artifacts produced in Phase‑1.
- Use caching (e.g., `@st.cache` or `st.cache_data` / `st.cache_resource`) for expensive operations if needed.
