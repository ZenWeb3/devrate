import joblib
import numpy as np

def load_model():
    return joblib.load('ml_model/models/rf.pkl')

def preprocess_input(data):
    return np.array([float(v) for v in data.values()])
