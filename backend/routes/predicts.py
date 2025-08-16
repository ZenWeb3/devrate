from flask import Blueprint, request, jsonify
from backend.utils.model_loader import load_model, preprocess_input

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])  # ✅ Only POST is allowed
def predict():
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        features = preprocess_input(data)
        model = load_model()
        prediction = model.predict([features])[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
