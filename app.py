import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Chargement du modèle
MODEL_PATH = os.path.join("app", "best_model.pkl")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
    else:
        print(f"⚠️  Modèle introuvable à {MODEL_PATH}. Lance d'abord export_model.py")

load_model()

FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500

    try:
        data = request.get_json()
        features = [float(data[f]) for f in FEATURES]
        X = np.array(features).reshape(1, -1)

        prediction = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1])

        risk_level = "Élevé" if proba > 0.6 else "Modéré" if proba > 0.35 else "Faible"
        risk_color = "red" if proba > 0.6 else "orange" if proba > 0.35 else "green"

        return jsonify({
            "prediction": prediction,
            "probability": round(proba * 100, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
