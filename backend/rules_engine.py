import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RulesEngine:
    """
    A professional rules engine for Python and JavaScript analyzers.
    Uses all 10 selected metrics to detect code quality issues.
    Returns only 'problems' and 'recommendations'.
    """

    def evaluate(self, metrics: dict):
        self.problems = []        # reset at every call
        self.recommendations = [] 

        # --- LOC and logical lines ---
        if metrics.get("loc", 0) > 50:
            self.problems.append("Function has too many lines of code")
            self.recommendations.append("Refactor into smaller, modular functions")
        elif metrics.get("loc", 0) > 30:
            self.problems.append("Function is moderately long")
            self.recommendations.append("Consider splitting into helper functions")

        if metrics.get("lOCode", 0) > 40:
            self.problems.append("Function has too many logical lines")
            self.recommendations.append("Simplify complex expressions or loops")

        # --- Cyclomatic complexity ---
        if metrics.get("v(g)", 0) > 10:
            self.problems.append("High cyclomatic complexity")
            self.recommendations.append("Reduce branching and simplify control flow")
        elif metrics.get("v(g)", 0) > 6:
            self.problems.append("Moderate cyclomatic complexity")
            self.recommendations.append("Consider simplifying nested conditions")

        # --- Essential and extended cyclomatic complexity ---
        if metrics.get("ev(g)", 0) > 6:
            self.problems.append("High essential complexity")
            self.recommendations.append("Refactor complex control flow")

        # --- Branch count ---
        if metrics.get("branchCount", 0) > 10:
            self.problems.append("Too many conditional branches")
            self.recommendations.append("Refactor branches into simpler logic")

        # --- Halstead difficulty and volume ---
        if metrics.get("d", 0) > 20:
            self.problems.append("High Halstead difficulty")
            self.recommendations.append("Simplify calculations and logic")

        if metrics.get("iv(g)", 0) > 20:
            self.problems.append("High Halstead volume")
            self.recommendations.append("Reduce code size or complexity")

        # --- Operators / Operands ---
        if metrics.get("total_Op", 0) == 0:
            self.problems.append("No operators detected")
            self.recommendations.append("Check if the function lacks meaningful operations")

        if metrics.get("uniq_Op", 0) < 2:
            self.problems.append("Very few unique operators")
            self.recommendations.append("Code may be too trivial or repetitive")

        if metrics.get("uniq_Opnd", 0) < 2:
            self.problems.append("Very few unique operands")
            self.recommendations.append("Code may be too trivial or not meaningful")

        if metrics.get("total_Opnd", 0) == 0:
            self.problems.append("No operands detected")
            self.recommendations.append("Check if the function lacks meaningful operations")

        # --- If nothing is flagged ---
        if not self.problems:
            self.recommendations.append("Code looks good. No major issues detected")

        logging.debug(f"RulesEngine problems: {self.problems}")
        logging.debug(f"RulesEngine recommendations: {self.recommendations}")

        return {"problems": self.problems, "recommendations": self.recommendations}
