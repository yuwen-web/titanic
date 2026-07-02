import io
import sqlite3

import pandas as pd
from flask import Flask, jsonify, request, render_template

import ml_model   # ← 我們寫好的機器學習模組

app = Flask(__name__)

# ============================================================
# 1. 全域讀取資料庫
# ============================================================

DATABASE = "my_db.db"

# 在全域開一個連線，每個 route 都能直接用 db 存取資料庫。
db = sqlite3.connect(DATABASE, check_same_thread=False)

# 讓查詢結果可以用 row["欄位名稱"] 存取，而不是 row[0]、row[1]。
db.row_factory = sqlite3.Row


# ============================================================
# 2. 小工具：把 SQLite Row 轉成 dict
# ============================================================

def row_to_dict(row):
    return dict(row)


# ============================================================
# 3. 前端頁面 Routes
# ============================================================

# 首頁（乘客列表 + CRUD）
@app.route("/")
def index_page():
    return render_template("index.html")

# 新增乘客頁面
@app.route("/passengers/new")
def new_passenger_page():
    return render_template("new.html")

# 編輯乘客頁面
@app.route("/passengers/<int:passenger_id>/edit")
def edit_passenger_page(passenger_id):
    return render_template("edit.html", passenger_id=passenger_id)

# 模型訓練頁面（機器學習）
@app.route("/train")
def train_page():
    return render_template("train.html")

# 預測頁面（機器學習）
@app.route("/predict")
def predict_page():
    return render_template("predict.html")


# ============================================================
# 4. API：取得全部乘客資料，包含簡單分頁與姓名搜尋
# GET /api/passengers?page=1&per_page=20&search=xxx
# ============================================================

@app.route("/api/passengers", methods=["GET"])
def get_passengers():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    search = request.args.get("search", "")
    offset = (page - 1) * per_page

    if search != "":
        total_row = db.execute(
            "SELECT COUNT(*) AS total FROM titanic WHERE Name LIKE ?",
            (f"%{search}%",)
        ).fetchone()
        rows = db.execute(
            """
            SELECT * FROM titanic
            WHERE Name LIKE ?
            ORDER BY PassengerId
            LIMIT ? OFFSET ?
            """,
            (f"%{search}%", per_page, offset)
        ).fetchall()
    else:
        total_row = db.execute(
            "SELECT COUNT(*) AS total FROM titanic"
        ).fetchone()
        rows = db.execute(
            """
            SELECT * FROM titanic
            ORDER BY PassengerId
            LIMIT ? OFFSET ?
            """,
            (per_page, offset)
        ).fetchall()

    total = total_row["total"]

    return jsonify({
        "message": "ok",
        "items": [row_to_dict(row) for row in rows],
        "page": page,
        "per_page": per_page,
        "total": total
    }), 200


# ============================================================
# 5. API：取得單一乘客
# GET /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["GET"])
def get_passenger(passenger_id):
    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    if row is None:
        return jsonify({"error": "找不到資料"}), 404

    return jsonify({"message": "ok", "item": row_to_dict(row)}), 200


# ============================================================
# 6. API：新增乘客
# POST /api/passengers
# ============================================================

@app.route("/api/passengers", methods=["POST"])
def create_passenger():
    data = request.get_json()

    cursor = db.execute(
        """
        INSERT INTO titanic (
            Survived, Pclass, Name, Sex, Age,
            SibSp, Parch, Ticket, Fare, Cabin, Embarked
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["Survived"], data["Pclass"], data["Name"], data["Sex"], data["Age"],
            data["SibSp"], data["Parch"], data["Ticket"], data["Fare"],
            data["Cabin"], data["Embarked"]
        )
    )
    db.commit()
    new_id = cursor.lastrowid

    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (new_id,)
    ).fetchone()

    return jsonify({"message": "created", "item": row_to_dict(row)}), 201


# ============================================================
# 7. API：修改乘客
# PUT /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["PUT"])
def update_passenger(passenger_id):
    data = request.get_json()

    cursor = db.execute(
        """
        UPDATE titanic SET
            Survived = ?, Pclass = ?, Name = ?, Sex = ?, Age = ?,
            SibSp = ?, Parch = ?, Ticket = ?, Fare = ?, Cabin = ?, Embarked = ?
        WHERE PassengerId = ?
        """,
        (
            data["Survived"], data["Pclass"], data["Name"], data["Sex"], data["Age"],
            data["SibSp"], data["Parch"], data["Ticket"], data["Fare"],
            data["Cabin"], data["Embarked"], passenger_id
        )
    )
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "找不到資料"}), 404

    row = db.execute(
        "SELECT * FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    ).fetchone()

    if row is None:
        return jsonify({"error": "找不到資料"}), 404

    return jsonify({"message": "updated", "item": row_to_dict(row)}), 200


# ============================================================
# 8. API：刪除乘客
# DELETE /api/passengers/1
# ============================================================

@app.route("/api/passengers/<int:passenger_id>", methods=["DELETE"])
def delete_passenger(passenger_id):
    cursor = db.execute(
        "DELETE FROM titanic WHERE PassengerId = ?",
        (passenger_id,)
    )
    db.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "找不到資料"}), 404

    return jsonify({"message": "deleted"}), 200


# ============================================================
# 9. 機器學習 API
# ============================================================

# 9-1 觸發訓練（需求 2）
# POST /api/train
@app.route("/api/train", methods=["POST"])
def api_train():
    started = ml_model.start_training()
    if not started:
        # 已經在訓練中，避免重複觸發
        return jsonify({"message": "已在訓練中", "status": ml_model.training_status}), 409
    return jsonify({"message": "訓練已開始", "status": ml_model.training_status}), 202


# 9-2 查詢訓練狀態（需求 3：讓頁面觀察是否訓練完成）
# GET /api/train/status
@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    return jsonify({
        "model_exists": ml_model.model_exists(),
        "status": ml_model.training_status
    }), 200


# 9-3 單筆預測（需求 4）
# POST /api/predict   body: {Pclass, Sex, Age, SibSp, Parch, Fare, Embarked}
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not ml_model.model_exists():
        return jsonify({"error": "尚未訓練模型，請先到訓練頁按下開始訓練"}), 400

    data = request.get_json()
    try:
        pred, proba = ml_model.predict_one(data)
    except Exception as e:
        return jsonify({"error": f"預測失敗：{e}"}), 400

    return jsonify({
        "message": "ok",
        "predicted_survived": pred,               # 0 或 1
        "survival_probability": round(proba, 4)   # 生還機率
    }), 200


# 9-4 批次預測（需求 4：上傳 CSV）
# POST /api/predict/batch   form-data: file=<csv>
@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    if not ml_model.model_exists():
        return jsonify({"error": "尚未訓練模型，請先到訓練頁按下開始訓練"}), 400

    if "file" not in request.files:
        return jsonify({"error": "沒有收到檔案"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "沒有選擇檔案"}), 400

    try:
        # 讀取上傳的 CSV
        content = file.read()
        df = pd.read_csv(io.BytesIO(content))
        result = ml_model.predict_batch(df)
    except Exception as e:
        return jsonify({"error": f"批次預測失敗：{e}"}), 400

    # pandas 的 NaN 沒辦法直接轉成合法 JSON，先換成 None（會變成 null）
    result = result.where(pd.notnull(result), None)

    return jsonify({
        "message": "ok",
        "count": int(len(result)),
        "items": result.to_dict(orient="records")
    }), 200


# ============================================================
# 10. 啟動 Flask
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
