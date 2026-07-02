"""
ml_model.py
============================================================
這個檔案負責「機器學習」的所有邏輯，跟 web（app.py）分開。
包含：讀資料、前處理 Pipeline、背景訓練、狀態追蹤、存/載模型、單筆與批次預測。

設計重點：用 sklearn Pipeline 把「前處理 + 模型」綁成一個物件再存檔，
          這樣預測時會自動套用跟訓練時「一模一樣」的補值與編碼，不會對不上。
"""

import os
import sqlite3
import threading

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================
# 0. 常數設定
# ============================================================

DATABASE = "my_db.db"
MODEL_PATH = "titanic_model.pkl"   # 訓練好的模型會存成這個檔

# 我們拿來預測的特徵（features）。刻意不用 PassengerId / Name / Ticket / Cabin：
#   - PassengerId 只是流水號，跟生還無關
#   - Name / Ticket 是高變異度的文字，直接丟進去會過擬合
#   - Cabin 缺失太多（約 77%）
NUMERIC = ["Age", "SibSp", "Parch", "Fare"]        # 數值型特徵
CATEGORICAL = ["Pclass", "Sex", "Embarked"]        # 類別型特徵（Pclass 雖是數字，但意義上是類別）
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Survived"                                # 預測目標：0=未生還, 1=生還


# ============================================================
# 1. 訓練狀態（給前端輪詢用）
# ============================================================
# 這是一個全域字典，記錄目前訓練到哪個階段。
# 前端每隔一兩秒打 /api/train/status，就能「觀察」訓練是否完成（需求 3）。
training_status = {
    "state": "idle",        # idle=尚未訓練 / training=訓練中 / done=完成 / error=失敗
    "message": "尚未開始訓練",
    "best_params": None,    # 最佳超參數
    "cv_score": None,       # 交叉驗證的最佳分數（roc_auc）
    "test_accuracy": None,  # 在測試集上的準確率
    "test_auc": None,       # 在測試集上的 AUC
    "n_samples": None,      # 用了幾筆資料訓練
}


# ============================================================
# 2. 讀資料
# ============================================================

def load_data():
    """從 SQLite 讀出整張 titanic 表，回傳 DataFrame。

    注意：這裡「另開一個連線」，而不是共用 app.py 的全域 db。
    因為訓練是在背景執行緒跑，用自己的連線最安全。
    """
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql("SELECT * FROM titanic", conn)
    conn.close()
    return df


# ============================================================
# 3. 建立前處理 + 模型的 Pipeline
# ============================================================

def build_pipeline():
    """組出一個 Pipeline：ColumnTransformer(前處理) -> RandomForest(模型)。"""

    # 數值特徵：用「中位數」補缺失值（Age 有不少 NaN）
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    # 類別特徵：先用「眾數」補缺失值（Embarked 有少數 NaN），再做 one-hot 編碼
    #   handle_unknown="ignore"：預測時若出現訓練沒看過的類別，不會報錯
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # ColumnTransformer：對不同欄位套用不同前處理
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])

    # 完整 Pipeline：資料進來 -> 前處理 -> 隨機森林分類器
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(random_state=42)),
    ])
    return pipe


# ============================================================
# 4. 實際訓練（會在背景執行緒中被呼叫）
# ============================================================

def _train_worker():
    """真正做訓練的函式。過程中會一直更新 training_status。"""
    global training_status
    try:
        # --- 4-1 讀資料 ---
        training_status.update({"state": "training", "message": "讀取資料中..."})
        df = load_data()
        X = df[FEATURES]
        y = df[TARGET]

        # --- 4-2 切訓練集 / 測試集 ---
        # stratify=y：讓生還/未生還的比例在兩邊保持一致
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- 4-3 用 GridSearchCV 搜尋超參數 ---
        training_status["message"] = "搜尋最佳超參數中（GridSearchCV）..."
        pipe = build_pipeline()

        # 要調的超參數。注意 key 的寫法是「步驟名__參數名」，
        # 因為分類器在 Pipeline 裡叫做 "clf"，所以是 clf__n_estimators。
        param_grid = {
            "clf__n_estimators": [100, 200, 300],       # 森林裡幾棵樹
            "clf__max_depth": [None, 5, 10],            # 每棵樹最深幾層（None=不限制）
            "clf__min_samples_split": [2, 5, 10],       # 一個節點至少幾筆才能再分裂
        }

        # cv=5：5 折交叉驗證；scoring="roc_auc"：用 AUC 當評分標準
        # n_jobs=1：單執行緒。資料量小、很快就跑完；且在 Flask 背景執行緒 / Windows
        #           上用多進程(n_jobs=-1)容易讓伺服器崩潰或卡住，所以刻意設 1 換取穩定。
        grid = GridSearchCV(
            pipe, param_grid, cv=5, scoring="roc_auc", n_jobs=1
        )
        grid.fit(X_train, y_train)

        # --- 4-4 用最佳模型在「測試集」上評估 ---
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        # 這裡是入門者最容易寫錯的地方：算 AUC 要用「機率」而不是 0/1 預測
        # predict_proba(X)[:, 1] 才是「生還(=1)的機率」
        y_proba = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # --- 4-5 存模型 ---
        # 整個 Pipeline（含前處理）一起存進 .pkl，預測時載回來就能直接用
        joblib.dump(best_model, MODEL_PATH)

        # --- 4-6 更新狀態為完成，並附上結果 ---
        training_status.update({
            "state": "done",
            "message": "訓練完成，模型已儲存",
            "best_params": grid.best_params_,
            "cv_score": round(float(grid.best_score_), 4),
            "test_accuracy": round(float(acc), 4),
            "test_auc": round(float(auc), 4),
            "n_samples": int(len(df)),
        })
    except Exception as e:
        training_status.update({"state": "error", "message": f"訓練失敗：{e}"})


def start_training():
    """對外的觸發函式：開一條背景執行緒去訓練，馬上回傳（不卡住 web）。

    回傳 False 代表「已經在訓練中」，避免重複觸發。
    """
    if training_status["state"] == "training":
        return False

    # 重置狀態
    training_status.update({
        "state": "training",
        "message": "準備開始訓練...",
        "best_params": None,
        "cv_score": None,
        "test_accuracy": None,
        "test_auc": None,
        "n_samples": None,
    })

    # daemon=True：主程式結束時這條執行緒也會跟著結束
    t = threading.Thread(target=_train_worker, daemon=True)
    t.start()
    return True


# ============================================================
# 5. 預測
# ============================================================

def model_exists():
    """檢查模型檔是否存在。"""
    return os.path.exists(MODEL_PATH)


def load_model():
    """把存好的模型載回來。"""
    return joblib.load(MODEL_PATH)


def _coerce_types(df):
    """把欄位轉成正確的型別。

    表單/CSV 讀進來常常都是字串，數值欄位要轉成數字，
    否則補值器（SimpleImputer median）會出錯。
    """
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # 轉不動的變 NaN，交給補值器處理
    # Pclass 雖歸在類別，但用字串 "1"/"2"/"3" 也可以 one-hot，這裡統一轉成字串比較穩
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype(str)
    return df


def predict_one(data: dict):
    """單筆預測。data 是 {特徵名: 值} 的字典。

    回傳 (預測結果 0/1, 生還機率 float)。
    """
    model = load_model()
    # 只取需要的特徵，組成一列 DataFrame
    row = {k: data.get(k) for k in FEATURES}
    X = pd.DataFrame([row])
    X = _coerce_types(X)

    proba = float(model.predict_proba(X)[:, 1][0])  # 生還機率
    pred = int(proba >= 0.5)                          # 機率 >= 0.5 判為生還
    return pred, proba


def predict_batch(df: pd.DataFrame):
    """批次預測。傳入一個 DataFrame（例如從上傳的 CSV 讀出來的）。

    回傳一個「原資料 + 兩個新欄位」的 DataFrame：
      - Predicted_Survived：預測 0/1
      - Survival_Probability：生還機率
    """
    model = load_model()

    # 檢查必要特徵是否齊全
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要欄位：{missing}")

    X = df[FEATURES].copy()
    X = _coerce_types(X)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    result = df.copy()
    result["Predicted_Survived"] = pred
    result["Survival_Probability"] = np.round(proba, 4)
    return result
