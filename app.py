"""
CASEFILE — Flask Web UI for Missing Person Location Prediction
==============================================================
A self-contained Flask app that loads the trained model, displays a
performance dashboard (5x repeated evaluation summary), and accepts
a case record via a form to produce a Top-5 probable location report.

Run:
    python app.py              # starts on http://127.0.0.1:5000
    python app.py --port 8080  # custom port
    python app.py --host 0.0.0.0 --port 5000

Requires the artifacts saved by build_model.py:
    models/location_model.pkl, models/label_encoder.pkl,
    results/evaluation_summary.json
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from casefile_common import DROP_COLS, engineer_features, get_feature_cols

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "location_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
SUMMARY_PATH = os.path.join(BASE_DIR, "results", "evaluation_summary.json")

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

with open(SUMMARY_PATH, encoding="utf-8") as fh:
    summary = json.load(fh)

AREAS = summary["areas"]

PRIORITY_LABELS = ["Very High", "High", "Medium", "Low", "Very Low"]
PRIORITY_COLORS = ["#ef4444", "#f97316", "#eab308", "#3b82f6", "#6b7280"]

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main UI dashboard."""
    return render_template("index.html")


@app.route("/api/model-info")
def api_model_info():
    """Return the evaluation summary and model metadata for the dashboard."""
    return jsonify(summary)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Accept a single case record as JSON and return Top-5 area predictions
    with probabilities, priority labels, and an ethical notice.
    """
    data = request.get_json(force=True, silent=True) or {}

    required = [
        "Age_Group", "Gender", "Last_Latitude", "Last_Longitude",
        "Last_Seen_Time", "Day", "Weather", "Usual_Area",
        "Average_Distance", "Average_Speed", "Previous_Area",
        "Time_Since_Last_Seen",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    # Build a single-row DataFrame in the same schema the model expects
    row = {
        "Case_ID": data.get("Case_ID", "UI-0001"),
        "Person_ID": data.get("Person_ID", "P-UI01"),
        "Age_Group": str(data["Age_Group"]),
        "Gender": str(data["Gender"]),
        "Last_Latitude": float(data["Last_Latitude"]),
        "Last_Longitude": float(data["Last_Longitude"]),
        "Last_Seen_Time": str(data["Last_Seen_Time"]),
        "Day": str(data["Day"]),
        "Weather": str(data["Weather"]),
        "Usual_Area": str(data["Usual_Area"]),
        "Average_Distance": float(data["Average_Distance"]),
        "Average_Speed": float(data["Average_Speed"]),
        "Previous_Area": str(data["Previous_Area"]),
        "Time_Since_Last_Seen": float(data["Time_Since_Last_Seen"]),
    }

    input_df = pd.DataFrame([row])
    input_eng = engineer_features(input_df)

    # Reproduce the exact drop the pipeline was trained with
    feature_cols = get_feature_cols(input_eng)

    probs = model.predict_proba(input_eng[feature_cols])[0]
    top_indices = np.argsort(probs)[::-1][:5]

    predictions = []
    for rank, idx in enumerate(top_indices):
        predictions.append({
            "rank": rank + 1,
            "area": label_encoder.inverse_transform([idx])[0],
            "probability": round(float(probs[idx]) * 100, 1),
            "priority": PRIORITY_LABELS[rank],
            "color": PRIORITY_COLORS[rank],
        })

    return jsonify({
        "case": {k: v for k, v in row.items() if k in [
            "Case_ID", "Person_ID", "Age_Group", "Gender", "Day",
            "Weather", "Usual_Area", "Previous_Area",
        ]},
        "predictions": predictions,
        "ethical_notice": (
            "This output is a statistical probability estimate for search "
            "prioritization only. It does not constitute a definitive prediction "
            "or proof of location. Always cross-reference with field intelligence."
        ),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASEFILE web UI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"CASEFILE UI running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)