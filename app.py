from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load model dan fitur
model = joblib.load("delivery_model.pkl")
feature_names = joblib.load("model_features.pkl")

app = Flask(__name__)

# =========================
# Home → Form Input
# =========================
@app.route("/")
def index():
    return render_template("index.html")

# =========================
# Prediksi
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {}

        # Ambil seluruh input form
        for key in request.form:
            value = request.form[key]
            # Numeric → convert to float
            try:
                data[key] = float(value)
            except:
                data[key] = value  # categorical

        df = pd.DataFrame([data])

        # Prediksi
        proba = model.predict_proba(df)[0][1]
        pred = "Delayed" if proba >= 0.5 else "On-time"

        confidence = round(proba * 100, 2)

        return render_template(
            "result.html",
            result=pred,
            confidence=confidence,
            row=data
        )

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# Halaman Grafik
# =========================
@app.route("/charts")
def charts():
    return render_template("charts.html")

# Run
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
