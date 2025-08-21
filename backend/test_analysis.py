from backend.static_analyser.python_analyser import analyze_python_code
from backend.utils.model_loader import load_model, preprocess_input
import json

if __name__ == "__main__":
    # Sample Python code
    sample_code = '''
def example(x):
    if x > 0:
        return x
    elif x < 0:
        return -x
    else:
        return 0
    '''

    # Analyze the code to compute metrics
    metrics = analyze_python_code(sample_code)
    print("Metrics:", json.dumps(metrics, indent=2))

    # Load the trained model
    model = load_model()

    # Preprocess the metrics for prediction
    if "error" not in metrics:
        input_data = preprocess_input(metrics)

        # Make a prediction
        prediction = model.predict(input_data)
        print("Prediction:", prediction)
    else:
        print("Error in analysis:", metrics["error"])