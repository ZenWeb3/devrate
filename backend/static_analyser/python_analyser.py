import os
import json
import tempfile
import lizard
import joblib
import numpy as np
import logging
import ast
from radon.metrics import h_visit
from backend.rules_engine import RulesEngine
from ml_model.src.preprocess import load_and_preprocess_data   # keep consistent with training

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Load selected features (top-k from training)
_, _, SELECTED_FEATURES = load_and_preprocess_data(k_features=10)


# --- AST-based supplement to compute unique ops/operands ---
class ASTAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.uniq_ops = set()
        self.uniq_opnds = set()
        self.total_ops = 0

    def visit_BinOp(self, node):
        self.uniq_ops.add(type(node.op).__name__)
        self.total_ops += 1
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.uniq_ops.add(type(node.op).__name__)
        self.total_ops += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.uniq_ops.add(type(node.op).__name__)
        self.total_ops += 1
        self.generic_visit(node)

    def visit_Compare(self, node):
        for op in node.ops:
            self.uniq_ops.add(type(op).__name__)
            self.total_ops += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        self.uniq_opnds.add(node.id)
        self.total_ops += 1
        self.generic_visit(node)

    def results(self):
        return {
            "uniq_Op": len(self.uniq_ops),
            "uniq_Opnd": len(self.uniq_opnds),
            "total_Op": self.total_ops,
        }


# --- Analyzer combining Lizard + AST ---
def analyze_python_code(source_code: str, use_halstead=True):
    """Analyze Python code and return only ML-selected metrics."""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(source_code)
            tf_path = tf.name

        try:
            analysis = lizard.analyze_file(tf_path)
            logging.info(f"Lizard analysis completed for {tf_path}")

            v_g_total = ev_g_total = branch_count_total = loc_total = lloc_total = 0
            function_count = len(analysis.function_list)

            if not analysis.function_list:
                v_g_total = analysis.average_cyclomatic_complexity or 1
                ev_g_total = max(1, v_g_total - 1)
                branch_count_total = max(0, v_g_total - 1)
                loc_total = analysis.nloc or len(source_code.splitlines())
                lloc_total = loc_total
            else:
                for func in analysis.function_list:
                    v_g_total += func.cyclomatic_complexity
                    ev_g_total += max(1, func.cyclomatic_complexity - 1)
                    branch_count_total += max(0, func.cyclomatic_complexity - 1)
                    loc_total += func.length
                    lloc_total += func.nloc

            # Halstead difficulty (d) via radon
            d_total = 0
            if use_halstead:
                try:
                    halstead_results = h_visit(source_code)
                    for _, report in halstead_results.functions:
                        d_total += report.difficulty
                except Exception as e:
                    logging.warning(f"Halstead analysis failed: {e}")

            # AST-based unique operators/operands
            ast_analyzer = ASTAnalyzer()
            try:
                tree = ast.parse(source_code)
                ast_analyzer.visit(tree)
            except SyntaxError as e:
                logging.error(f"AST parse error: {e}")
            ast_metrics = ast_analyzer.results()

            # Final metrics dictionary aligned with SELECTED_FEATURES
            metrics = {
                "d": d_total,
                "loc": loc_total,
                "v(g)": v_g_total,
                "branchCount": branch_count_total,
                "lOCode": lloc_total,
                "uniq_Opnd": ast_metrics["uniq_Opnd"],
                "iv(g)": max(1, v_g_total - 1),   # approximation
                "ev(g)": ev_g_total,
                "uniq_Op": ast_metrics["uniq_Op"],
                "total_Op": ast_metrics["total_Op"],
            }
        finally:
            try:
                os.unlink(tf_path)
            except OSError:
                pass

        return metrics
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return {"error": str(e)}


def load_model():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model_path = os.path.join(project_root, "ml_model", "models", "rf_tuned.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def preprocess_input(metrics: dict):
    values = [float(metrics.get(f, 0)) for f in SELECTED_FEATURES]
    return np.array(values).reshape(1, -1)


def run_analysis(source_code: str, output_path="analysis_output.json"):
    metrics = analyze_python_code(source_code)
    if "error" in metrics:
        output = {"error": metrics["error"]}
    else:
        rules_engine = RulesEngine()
        rules_result = rules_engine.evaluate(metrics)

        model = load_model()
        input_array = preprocess_input(metrics)
        pred = model.predict(input_array)[0]

        # ✅ match label map with training labels
        label_map = {0: "high", 1: "medium", 2: "low"}
        pred_label = label_map.get(pred, f"Unknown ({pred})")
        probabilities = {
            label_map.get(i, f"Class {i}"): float(p)
            for i, p in enumerate(model.predict_proba(input_array)[0])
        } if hasattr(model, "predict_proba") else {}

        output = {
            "metrics": metrics,
            "selected_features": list(SELECTED_FEATURES),
            # "feature_ranking": feature_scores.to_dict(orient="records") if hasattr(feature_scores, "to_dict") else feature_scores,
            "problems": rules_result["problems"],
            "recommendations": rules_result["recommendations"],
            "prediction": {
                "label": pred_label,
                "class": int(pred),
                "probabilities": probabilities,
            },
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logging.info(f"✅ Analysis saved to {output_path}")
    return output


if __name__ == "__main__":
    sample_code = '''
def complex_example(a, b):
    total = 0
    for i in range(a):
        if i % 2 == 0:
            total += i * b
        else:
            try:
                total += i / (b - i)
            except ZeroDivisionError:
                total += 1
    while total < 100:
        total = total * 2 - a
    def helper(x):
        return x ** 2 + x // 2
    total += helper(total)
    return total
'''
    result = run_analysis(sample_code)
    print(json.dumps(result, indent=2))
