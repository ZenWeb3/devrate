# backend/static_analyser/js_analyser.py
import esprima
import math
import logging
from typing import Dict, Any, Tuple, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

COMMENT_SINGLE = "//"
COMMENT_BLOCK_START = "/*"
COMMENT_BLOCK_END = "*/"

# --------------------------
# Helpers: LOC / logical LOC
# --------------------------
def _strip_comments_and_count_locs(source: str) -> Tuple[int, int]:
    """
    Returns (loc, lOCode)
    - loc: total non-empty lines
    - lOCode: lines that are not comments and not blank
    """
    lines = source.splitlines()
    in_block = False
    loc = 0
    locode = 0

    for raw in lines:
        stripped = raw.strip()
        if stripped:
            loc += 1
        if not stripped or stripped.startswith(COMMENT_SINGLE):
            continue
        if in_block:
            if COMMENT_BLOCK_END in stripped:
                in_block = False
            continue
        if COMMENT_BLOCK_START in stripped:
            in_block = True
            continue
        locode += 1
    return loc, locode

# --------------------------
# Halstead metrics
# --------------------------
OPERATOR_TOKENS = {
    "=", "+", "-", "*", "/", "%", "++", "--",
    "==", "!=", "===", "!==",
    "<", ">", "<=", ">=",
    "&&", "||", "!", "~",
    "<<", ">>", ">>>",
    "&", "|", "^",
    "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", ">>>=", "&=", "|=", "^=",
    "?", ":", ".", ",", ";", "=>"
}

def _halstead_metrics(tokens) -> Dict[str, float]:
    distinct_ops: Set[str] = set()
    distinct_opnds: Set[str] = set()
    N1 = 0  # total operators
    N2 = 0  # total operands

    for tok in tokens:
        ttype = tok.type
        tval = str(tok.value) if hasattr(tok, "value") else ""
        if ttype in ("Punctuator", "Keyword", "Boolean", "Null"):
            distinct_ops.add(tval or ttype)
            N1 += 1
        elif ttype in ("Identifier", "Numeric", "String", "Template"):
            distinct_opnds.add(tval)
            N2 += 1
        else:
            distinct_opnds.add(tval)
            N2 += 1

    n1 = max(1, len(distinct_ops))
    n2 = max(1, len(distinct_opnds))
    total_ops = N1 + N2

    V = float(total_ops) * math.log2(n1 + n2) if (n1 + n2) > 1 else 0.0
    D = (n1 / 2.0) * (N2 / n2) if n2 > 0 else 0.0

    return {
        "d": D,
        "iv(g)": V,
        "uniq_Op": n1,
        "uniq_Opnd": n2,
        "total_Op": total_ops
    }

# --------------------------
# Cyclomatic Complexity
# --------------------------
DECISION_NODES = {
    "IfStatement", "ForStatement", "ForInStatement", "ForOfStatement",
    "WhileStatement", "DoWhileStatement", "SwitchCase", "ConditionalExpression"
}

LOGICAL_BINOPS = {"&&", "||"}

def _count_complexity_and_branches(node) -> Tuple[int, int]:
    v_g = 1
    branch = 0
    stack = [node]

    while stack:
        cur = stack.pop()
        t = getattr(cur, "type", "")

        if t in DECISION_NODES:
            v_g += 1
            branch += 1

        if t == "LogicalExpression":
            op = getattr(cur, "operator", "")
            if op in LOGICAL_BINOPS:
                v_g += 1
                branch += 1

        for key, value in cur.__dict__.items():
            if key in ("range", "loc"):
                continue
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, "type"):
                        stack.append(item)
            elif hasattr(value, "type"):
                stack.append(value)

    return max(1, v_g), max(0, branch)

# --------------------------
# Public API
# --------------------------
def analyze_js_code(source_code: str) -> Dict[str, Any]:
    """
    Analyze JavaScript/TypeScript code and return exactly the 10 selected metrics:
    loc, v(g), ev(g), iv(g), d, lOCode, uniq_Op, uniq_Opnd, total_Op, branchCount
    """
    try:
        loc, locode = _strip_comments_and_count_locs(source_code)

        try:
            ast_root = esprima.parseScript(source_code, loc=False, tolerant=True, comment=True)
        except Exception:
            ast_root = esprima.parseModule(source_code, loc=False, tolerant=True, comment=True)

        v_g, branch_count = _count_complexity_and_branches(ast_root)
        ev_g = max(1, v_g - 1)

        tokens = esprima.tokenize(source_code, comment=True, tolerant=True)
        hal = _halstead_metrics(tokens)

        metrics = {
            "loc": loc,
            "v(g)": v_g,
            "ev(g)": ev_g,
            "iv(g)": hal["iv(g)"],
            "d": hal["d"],
            "lOCode": locode,
            "uniq_Op": hal["uniq_Op"],
            "uniq_Opnd": hal["uniq_Opnd"],
            "total_Op": hal["total_Op"],
            "branchCount": branch_count
        }
        return metrics

    except Exception as e:
        logging.exception("JS analysis failed")
        return {"error": str(e)}
