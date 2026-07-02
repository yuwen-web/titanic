# Titanic 乘客管理 + 機器學習

一個 Flask + SQLite + AJAX 的專案：既能對 Titanic 乘客資料做 CRUD，
也能一鍵訓練機器學習模型、並預測乘客是否生還。

## 資料夾結構

```
專案資料夾/
├── app.py              # Flask 主程式（頁面路由 + CRUD API + ML API）
├── ml_model.py         # 機器學習核心（前處理、訓練、存/載模型、預測）
├── init_db.py          # 建立 my_db.db 並從 titanic.csv 匯入資料
├── titanic.csv         # 原始資料
├── requirements.txt    # 套件清單
└── templates/          # ← 所有 HTML 一定要放在這個資料夾
    ├── index.html      # 首頁：乘客列表 + CRUD
    ├── new.html        # 新增乘客
    ├── edit.html       # 編輯乘客
    ├── train.html      # 模型訓練頁
    └── predict.html    # 預測頁（單筆輸入 / 上傳 CSV）
```

> 執行後會自動產生：`my_db.db`（資料庫）、`titanic_model.pkl`（訓練好的模型）。

## 啟動步驟

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 建立資料庫（只需第一次，或想重置資料時）
python init_db.py

# 3. 啟動網站
python app.py
```

然後開瀏覽器到 http://127.0.0.1:5000

## 操作流程

1. 首頁可瀏覽 / 搜尋 / 新增 / 編輯 / 刪除乘客（CRUD）。
2. 進到「模型訓練」頁，按「開始訓練」→ 頁面會輪詢狀態，訓練完成後顯示最佳超參數與評估指標，模型自動存成 `titanic_model.pkl`。
3. 進到「預測」頁，輸入一筆乘客資料，或上傳 CSV，得到「是否生還」與「生還機率」。

## 對應作業需求

| 需求 | 實作位置 |
|---|---|
| 1. 新增 ML 頁面 | `train.html`、`predict.html`，路由在 `app.py` 的 `/train`、`/predict` |
| 2. 一鍵訓練 + 調超參數 + 顯示最佳超參數 | `POST /api/train` → `ml_model.start_training()`；用 `GridSearchCV` 調 `n_estimators`、`max_depth`、`min_samples_split` |
| 3. 觀察訓練完成 + 儲存模型 | `GET /api/train/status` 輪詢；`joblib.dump` 存模型 |
| 4. 輸入/上傳 CSV 預測存活與機率 | `POST /api/predict`（單筆）、`POST /api/predict/batch`（CSV）；機率用 `predict_proba()[:, 1]` |

## 技術重點

- 模型：RandomForest；用 sklearn `Pipeline` 把「前處理 + 分類器」綁在一起存檔，
  預測時自動套用跟訓練時一模一樣的補值與編碼，避免 train/predict 前處理不一致。
- 特徵：Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
  （不用 PassengerId / Name / Ticket / Cabin）。
- 訓練用背景執行緒進行，前端才能一邊輪詢狀態、一邊不被卡住。
- `GridSearchCV` 特意設 `n_jobs=1`：資料量小、跑很快，且在 Flask 背景執行緒 /
  Windows 下用多進程容易讓伺服器崩潰。
