from flask import Blueprint, Flask, request, jsonify
from backend.static_analyser.python_analyser import analyze_python_code, load_model, preprocess_input
from backend.static_analyser.js_analyser import analyze_js_code 
from backend.rules_engine import RulesEngine
import json
import numpy as np

# Blueprint exported for backend.app to import (predict_bp)
predict_bp = Blueprint("predict", __name__, url_prefix="")

# Load ML model once at startup
model = load_model()
rules_engine = RulesEngine()

label_map = {0: "High Quality", 1: "Medium Quality", 2: "Low Quality"}

@predict_bp.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data or "source_code" not in data or "language" not in data:
        return jsonify({"error": "source_code and language are required"}), 400

    source_code = data["source_code"]
    language = data["language"].lower()
    software_name = data.get("software_name", "Unknown")

    # Select analyzer based on language
    if language == "python":
        metrics = analyze_python_code(source_code)
    elif language == "javascript":
        metrics = analyze_js_code(source_code)  # You’ll need this function
    else:
        return jsonify({"error": f"Unsupported language: {language}"}), 400

    if "error" in metrics:
        return jsonify({"error": metrics["error"]}), 400

    # Evaluate rules
    rules_result = rules_engine.evaluate(metrics)

    # Predict with ML (you might want separate models for JS/Python)
    input_array = preprocess_input(metrics)
    pred = model.predict(input_array)[0]
    pred_label = label_map.get(pred, f"Unknown ({pred})")
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probabilities = {
            label_map.get(i, f"Class {i}"): float(p)
            for i, p in enumerate(model.predict_proba(input_array)[0])
        }

    output = {
        "software_name": software_name,
        "language": language,
        "metrics": metrics,
        "problems": rules_result.get("problems", []),
        "recommendations": rules_result.get("recommendations", []),
        "prediction": {
            "label": pred_label,
            "class": int(pred),
            "probabilities": probabilities
        }
    }

    return jsonify(output), 200
