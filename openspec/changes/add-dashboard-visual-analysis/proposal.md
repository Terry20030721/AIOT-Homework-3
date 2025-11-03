## Why
我們需要一個本地視覺化儀表板來快速檢視 Phase‑1（Spam classification）資料與模型表現，方便教學與分析。Phase‑2 將提供一個使用 Streamlit 的互動式 dashboard，讓開發者在本機就能看到資料分布、重要字詞與模型在測試集上的 ROC / Precision‑Recall 曲線。

## What Changes
- 新增 `scripts/dashboard.py`（Streamlit app）並在 README 加入啟動指引。
- App 啟動時「自動」讀取 `datasets/processed/cleaned.csv` 並「自動」執行分析（啟動即顯示圖表，無需按鈕觸發）。
- dashboard 在內部做 train/test 切分並重新訓練一個 SVM pipeline（供分析使用，不取代 Phase‑1 的已保存模型）。訓練過程必須能產生機率（例如使用 `CalibratedClassifierCV`）。
- 畫出三個圖表：Data Overview、Top Tokens by Class、Model Performance (Test)。
- 更新 `requirements.txt` 加入 `streamlit`、`matplotlib`（與可選的 `seaborn`、`scikit-learn` 已存在於 Phase‑1 中）。

## Impact
- 受影響規格：`visualization`（新增 capability）；影響檔案：`scripts/dashboard.py`、`requirements.txt`、`README.md`。

## Acceptance criteria
- 可啟動 dashboard：`streamlit run scripts/dashboard.py` 並在 local browser 顯示介面。
- dashboard 自動讀取 `datasets/processed/cleaned.csv`（錯誤情況會顯示友善提示）。
- dashboard 在後端自行完成 train/test 切分（例如 80/20）並以固定 random seed 以利重現。
- dashboard 會重新訓練一個 TF‑IDF + SVM pipeline（或 LinearSVC + calibration）以取得分數/機率，用於畫 ROC 與 Precision‑Recall 曲線。
- 三張圖表正確呈現：spam vs ham 長條圖、每類別 top tokens（可互換 top N）、測試集上的 ROC 與 PR 曲線並顯示 AUC/average‑precision 數值。

## Risks
- 在本機執行訓練可能需要較多 CPU / memory（不過資料集很小，應可接受）。
- 若使用 LinearSVC 需注意沒有 predict_proba；將使用 CalibratedClassifierCV 或 decision_function 作為替代以產生機率分數。

## Timeline
- 0.5 day: scaffold `scripts/dashboard.py` 與更新 requirements/README
- 0.5 day: 實作資料統計、token 聚合、並畫圖
- 0.5 day: 實作 model retrain 與 ROC/PR 繪製，測試與微調
