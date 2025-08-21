import joblib
import numpy as np

# Fixed feature order (must match training exactly)
FEATURE_ORDER = ['ev(g)', 'v(g)', 'branchCount', 'loc', 'lOCode']

def load_model(model_name='rf.pkl'):
    """Load a trained model by filename from models folder."""
    return joblib.load(f'ml_model/models/rf_tuned.pkl')

def preprocess_input(data: dict):
    """Convert input dict into ordered NumPy array for prediction."""
    try:
        values = [float(data[feat]) for feat in FEATURE_ORDER]
    except KeyError as e:
        raise ValueError(f"Missing feature in input data: {e}")
    return np.array(values).reshape(1, -1)  # reshape for scikit-learn
