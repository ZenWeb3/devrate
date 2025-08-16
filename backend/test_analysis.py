from backend.static_analyser.python_analyser import analyze_python_code
import json

if __name__ == "__main__":
    sample_code = '''
def example(x):
    if x > 0:
        return x
    elif x < 0:
        return -x
    else:
        return 0
'''
    result = analyze_python_code(sample_code)
    print(json.dumps(result, indent=2))
    