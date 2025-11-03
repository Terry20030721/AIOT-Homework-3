Streamlit 本機介面（啟動後在瀏覽器開啟）

http://localhost:8501

---

# Spam classification — Phase 1 & Phase 2

簡介
本專案示範一個簡單的 Spam/Ham 分類工作流程（Phase‑1）：下載公開 SMS 資料集、前處理、訓練 TF‑IDF + SVM 的 baseline，並提供一個 CLI 預測工具。Phase‑2 增加一個本地的 Streamlit 視覺化儀表板（`scripts/dashboard.py`），可自動載入資料、在啟動時重新切分/訓練分析用模型，並顯示資料概覽、關鍵字排行與模型效能（ROC / PR）以及 Live Inference 區塊。

檔案重點
- `datasets/raw/` — 原始下載的 CSV 檔
- `datasets/processed/cleaned.csv` — 前處理後的資料（含 `label,text` 欄位）
- `models/` — Phase‑1 的訓練 artifact（`pipeline.joblib`）
- `scripts/preprocess.py` — 下載並清理原始 CSV
- `scripts/train.py` — 訓練 TF‑IDF + LinearSVC pipeline 並儲存 artifact
- `scripts/predict.py` — CLI 方式載入 artifact 並做單筆預測
- `scripts/dashboard.py` — Streamlit 儀表板（啟動時會自動載入並執行分析）
- `requirements.txt` — Python 相依套件清單
- `openspec/changes/` — OpenSpec 變更提案與規格（包含 `add-spam-classification-svm` 與 `add-dashboard-visual-analysis`）

快速開始（conda env 範例）
1. 啟動你的 conda 環境（例如 `myenv`）：

```bash
conda activate myenv
```

2. 安裝依賴：

```bash
pip install -r requirements.txt
```

（備註：若偏好使用 conda 二進位套件以避免編譯，可先 `conda install -y numpy pandas scikit-learn matplotlib seaborn`，再 `pip install streamlit joblib`。）

Phase‑1：資料處理與訓練
1. 前處理（若已下載會使用本地檔案）：

```bash
python scripts/preprocess.py --raw datasets/raw/sms_spam_no_header.csv --out datasets/processed/cleaned.csv
```

2. 訓練並建立 artifact：

```bash
python scripts/train.py --data datasets/processed/cleaned.csv --out-dir models
```

3. CLI 預測（單句）：

```bash
python scripts/predict.py "Free entry in 2 a wkly comp to win FA Cup final tkts"
# => spam
```

Phase‑2：啟動 Streamlit 儀表板（自動載入與分析）

```bash
streamlit run scripts/dashboard.py
```

- 啟動後，瀏覽器開啟本機 URL（預設 http://localhost:8501）即可看到：
	1) Data Overview：spam vs ham 長條圖
	2) Top Tokens by Class：每類別最常出現的 token
	3) Model Performance (Test)：ROC 與 Precision‑Recall 曲線（含 AUC / AP）
	4) Live Inference：可貼入文字或生成隨機範例（dataset/sample template），按 Predict 顯示預測標籤與 spam 機率

錯誤處理
- 如果 `datasets/processed/cleaned.csv` 不存在或欄位格式錯誤，dashboard 會顯示友善錯誤訊息並提示如何執行 preprocessing，不會直接崩潰。

開發與測試
- `openspec/changes/add-spam-classification-svm/` 與 `openspec/changes/add-dashboard-visual-analysis/` 包含 proposal、tasks、design 與 spec，用來追蹤變更與驗收準則。
- 建議加入簡單單元測試（preprocess 的輸入／輸出、predict CLI 的 smoke test）與 CI（例如 GitHub Actions）以在 PR 時驗證基礎功能。

常見問題
- 若 Streamlit 無法啟動或出現套件缺少，請確認 `requirements.txt` 已安裝；macOS 若提示 Watchdog 建議安裝 Xcode command line tools：

```bash
xcode-select --install
pip install watchdog
```



