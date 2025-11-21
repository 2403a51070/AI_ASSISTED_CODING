# app.py
# Minimal Flask app: loads model.joblib and serves a single page + /predict

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

APP_PORT = 5000
MODEL_PATH = "model.joblib"
CURRENT_YEAR = 2025

app = Flask(__name__)

# Try to load model; if missing, app will still run but /predict will return an error
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded model:", MODEL_PATH)
    except Exception as e:
        print("Error loading model:", e)
        model = None
else:
    print("Model not found. Run: python train.py to create model.joblib")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not available. Please run train.py first."}), 500

    data = request.get_json() or request.form.to_dict()
    try:
        year = int(data.get("year"))
        mileage = float(data.get("mileage_km"))
        make = data.get("make", "missing")
        model_name = data.get("model", "missing")
        transmission = data.get("transmission", "missing")
        fuel = data.get("fuel", "missing")
        city = data.get("city", "missing")
    except Exception as e:
        return jsonify({"error": "Invalid input", "details": str(e)}), 400

    age = CURRENT_YEAR - year
    X = pd.DataFrame([{
        "age": age,
        "mileage_km": mileage,
        "make": make,
        "model": model_name,
        "transmission": transmission,
        "fuel": fuel,
        "city": city
    }])
    pred = model.predict(X)[0]
    lower = int(pred * 0.9)
    upper = int(pred * 1.1)
    explanation = "Estimate based mainly on age and mileage (newer and lower mileage → higher price)."

    return jsonify({
        "predicted_price": int(pred),
        "lower": lower,
        "upper": upper,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True, port=APP_PORT)
