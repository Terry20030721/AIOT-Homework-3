## ADDED Requirements

### Requirement: Local dashboard for visual analysis of spam classifier
The system SHALL provide a local Streamlit-based dashboard that reads `datasets/processed/cleaned.csv`, trains an internal analysis model (TF‑IDF + SVM), and displays interactive visualizations for data overview, token importance, and model performance.

#### Scenario: Dashboard reads dataset and shows data overview
- **WHEN** the developer runs `streamlit run scripts/dashboard.py`
- **THEN** the dashboard SHALL read `datasets/processed/cleaned.csv` and display a bar chart showing counts of `spam` vs `ham`.

#### Scenario: App auto-loads and auto-runs analysis on startup
- **WHEN** the developer starts the app with `streamlit run scripts/dashboard.py`
- **THEN** the dashboard SHALL automatically load `datasets/processed/cleaned.csv`, perform the train/test split, retrain the internal SVM analysis pipeline, and render all required charts immediately — without requiring any user interactions (no button presses) to trigger the initial analysis.

#### Scenario: Dashboard displays top tokens by class
- **WHEN** the dataset is loaded
- **THEN** the dashboard SHALL compute token frequencies (or TF-IDF scores) per class and display top N tokens for each class.

#### Scenario: Dashboard retrains internal model and shows ROC/PR
- **WHEN** the user requests a retrain (default split 80/20)
- **THEN** the dashboard SHALL split the data deterministically using a fixed seed, train a TF‑IDF + SVM pipeline (with probability calibration), compute ROC and Precision‑Recall curves on the test set, and display the curves with summary metrics (AUC, average precision).

#### Scenario: Dashboard displays confusion matrix and classification summary
- **WHEN** the internal analysis model is trained and evaluated on the test set
- **THEN** the dashboard SHALL display a confusion matrix (counts and optionally normalized) for the test predictions, and SHALL present a small classification summary (accuracy, precision, recall, F1) for the `spam`/`ham` labels.

#### Scenario: Missing dataset handled gracefully
- **WHEN** `datasets/processed/cleaned.csv` is missing
- **THEN** the dashboard SHALL show a clear message describing how to run preprocessing (link to README) and SHALL NOT crash.

#### Scenario: Automated test generation and probability prediction
- **WHEN** the dashboard exposes or includes a test utility (e.g., a hidden developer panel or script)
- **THEN** the utility SHALL be able to randomly generate synthetic example texts labeled `spam` or `ham` (or sample real examples from the dataset), feed them to the internal analysis pipeline, and return the predicted label and the spam probability (a numeric value between 0 and 1).
- **AND** the dashboard SHALL display the generated example, its ground-truth label (if applicable), the predicted label, and the predicted spam probability in a concise test output area.
