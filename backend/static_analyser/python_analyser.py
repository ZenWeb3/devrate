import subprocess
import tempfile
import os
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
import json

def analyze_python_code(code: str):
    with tempfile.NamedTemporaryFile(mode='w', suffix=".py", delete=False) as temp_file:
        temp_file.write(code)
        temp_path = temp_file.name

    try:
        # Radon raw metrics
        with open(temp_path, 'r') as f:
            raw_metrics = analyze(f.read())

        # Cyclomatic complexity
        with open(temp_path, 'r') as f:
            complexity_objs = cc_visit(f.read())
            complexity = []
            for obj in complexity_objs:
                complexity.append({
                    "name": obj.name,
                    "complexity": obj.complexity,
                    "lineno": obj.lineno,
                    "col_offset": obj.col_offset,
                    "endline": getattr(obj, 'endline', None),
                })

        # Maintainability Index
        with open(temp_path, 'r') as f:
            mi_score = mi_visit(f.read(), True)

        # Pylint analysis
        pylint_result = subprocess.run(
            ['pylint', temp_path, '--output-format=json', '--disable=all', '--enable=warning,error,convention'],
            capture_output=True,
            text=True
        )

        try:
            pylint_issues = json.loads(pylint_result.stdout)
        except json.JSONDecodeError:
            pylint_issues = []

        return {
            "loc": raw_metrics.loc,
            "lloc": raw_metrics.lloc,
            "sloc": raw_metrics.sloc,
            "comments": raw_metrics.comments,
            "multi": raw_metrics.multi,
            "blank": raw_metrics.blank,
            "maintainability_index": mi_score,
            "cyclomatic_complexity": complexity,
            "pylint_issues": pylint_issues
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        os.remove(temp_path)
