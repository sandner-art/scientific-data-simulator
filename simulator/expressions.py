# simulator/expressions.py
"""Safe parsing, evaluation and comparison of mathematical expressions.

Used to score law-discovery answers such as ``"2.3*x**1.5"`` or
``"y = 4.1 exp(-0.7 x)"``. Expressions are parsed with :mod:`ast` against a strict
whitelist - numbers, the declared variables, ``pi``/``e``, ``+ - * / **`` and a
few numpy functions - and evaluated by walking the tree. Nothing is passed to
``eval``; attribute access, indexing, lambdas and other names are rejected.

Two expressions *match* on a domain when their relative RMS difference on
sample points of that domain is below a tolerance (default 1 %), i.e. they
describe the same function there, whatever their written form.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

FUNCTIONS: Dict[str, Callable] = {
    "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arctan": np.arctan, "atan": np.arctan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "abs": np.abs,
}
CONSTANTS = {"pi": math.pi, "e": math.e}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_MAX_LENGTH = 500
_MAX_NODES = 200


class ExpressionError(ValueError):
    """The text is not an allowed expression."""


_TOKEN = re.compile(r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)"
                    r"|(?P<op>\*\*|[-+*/(),])|(?P<bad>\S))")


def _frac(t: str) -> str:
    """``\\frac{a}{b}`` -> ``((a)/(b))``, innermost first."""
    pattern = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(50):
        t, n = pattern.subn(r"((\1)/(\2))", t)
        if not n:
            break
    return t


def _implicit_products(t: str) -> str:
    """Insert ``*`` for implicit products (``2x``, ``3x(1-x)``, ``(a)(b)``, ``2 exp(x)``) on a
    token stream, so number literals such as ``1e-3`` are never split."""
    out, prev = [], None  # prev: "value" (number, variable, constant, ')'), "func", or None
    pos = 0
    while pos < len(t):
        m = _TOKEN.match(t, pos)
        if not m or m.end() == pos:
            break
        pos = m.end()
        kind = m.lastgroup
        tok = m.group(kind)
        starts_value = kind in ("num", "name") or tok == "("
        if prev == "value" and starts_value:
            out.append("*")
        out.append(tok)
        if kind == "num" or (kind == "op" and tok == ")"):
            prev = "value"
        elif kind == "name":
            prev = "func" if tok in FUNCTIONS else "value"
        else:
            prev = None
    return " ".join(out)


def normalise(text: str) -> str:
    """Accept common written forms: ``y = ...``, ``^`` for powers, LaTeX (``\\exp``,
    ``\\cdot``, ``\\frac{}{}``), implicit products like ``2x``, ``3x(1-x)`` and
    ``2 exp(...)``, and trailing punctuation."""
    if not isinstance(text, str):
        raise ExpressionError("expression must be a string")
    t = text.strip().strip("$").strip()
    boxed = re.findall(r"\\boxed\{(.*)\}", t)
    if boxed:
        t = boxed[-1]
    if "=" in t:
        t = t.split("=")[-1]
    t = t.strip().rstrip(".,;:").strip().strip("$").strip()
    t = t.replace("\\cdot", "*").replace("\\times", "*").replace("\\left", "").replace("\\right", "")
    t = _frac(t)
    t = re.sub(r"\\([A-Za-z]+)", lambda m: m.group(1) if m.group(1) in FUNCTIONS or m.group(1) in CONSTANTS
               else m.group(0), t)
    t = t.replace("{", "(").replace("}", ")")
    t = t.replace("^", "**").replace("\u2212", "-").replace("\u00b7", "*").replace("\u00d7", "*")
    if len(t) > 4 * _MAX_LENGTH:
        raise ExpressionError("expression too long")
    return _implicit_products(t).strip()


def parse(text: str, variables: Sequence[str] = ("x",)) -> ast.Expression:
    """Parse and validate an expression; raise :class:`ExpressionError` if not allowed."""
    source = normalise(text)
    if not source:
        raise ExpressionError("empty expression")
    if len(source) > _MAX_LENGTH:
        raise ExpressionError("expression too long")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"not a valid expression: {source!r} ({e.msg})") from None
    except (RecursionError, MemoryError, ValueError):
        raise ExpressionError("expression too deeply nested") from None
    allowed_names = set(variables) | set(CONSTANTS)
    callees = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > _MAX_NODES:
            raise ExpressionError("expression too complex")
        if isinstance(node, (ast.Expression, ast.Load)):
            continue
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            continue
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            continue
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd)):
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            try:
                float(node.value)
            except OverflowError:
                raise ExpressionError("number too large") from None
            continue
        if isinstance(node, ast.Name):
            if id(node) in callees:
                if node.id in FUNCTIONS:
                    continue
            elif node.id in allowed_names:
                continue
            raise ExpressionError(f"unknown name '{node.id}' (allowed: {sorted(allowed_names)} and "
                                  "one-argument calls of the whitelisted functions)")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and len(node.args) == 1 \
                    and not node.keywords:
                continue
            raise ExpressionError("only one-argument calls of whitelisted functions are allowed")
        raise ExpressionError(f"construct not allowed: {type(node).__name__}")
    return tree


def _evaluate(node, env):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, env)
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env[node.id] if node.id in env else CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp):
        value = _evaluate(node.operand, env)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_evaluate(node.left, env), _evaluate(node.right, env))
    if isinstance(node, ast.Call):
        return FUNCTIONS[node.func.id](_evaluate(node.args[0], env))
    raise ExpressionError(f"cannot evaluate {type(node).__name__}")  # pragma: no cover


def evaluate(text: str, **values) -> np.ndarray:
    """Evaluate an expression for the given variable values (arrays broadcast)."""
    tree = parse(text, tuple(values))
    env = {k: np.asarray(v, dtype=float) for k, v in values.items()}
    with np.errstate(all="ignore"):
        return np.asarray(_evaluate(tree, env), dtype=float)


def compare(candidate: str, reference: str, domain: Tuple[float, float], variable: str = "x",
            n: int = 200) -> Dict[str, float]:
    """Relative RMS and max errors of ``candidate`` against ``reference`` on ``domain``."""
    lo, hi = float(domain[0]), float(domain[1])
    x = np.linspace(lo, hi, n)
    truth = evaluate(reference, **{variable: x})
    guess = evaluate(candidate, **{variable: x})
    if guess.shape != truth.shape:
        guess = np.broadcast_to(guess, truth.shape)
    ok = np.isfinite(truth)
    if not np.all(np.isfinite(guess[ok])):
        return {"relative_rms": float("inf"), "relative_max": float("inf")}
    scale = float(np.sqrt(np.mean(truth[ok] ** 2))) or 1.0
    diff = guess[ok] - truth[ok]
    return {"relative_rms": float(np.sqrt(np.mean(diff ** 2)) / scale),
            "relative_max": float(np.max(np.abs(diff)) / scale)}


def matches(candidate: str, reference: str, domain: Tuple[float, float], variable: str = "x",
            rtol: float = 0.01) -> bool:
    """True if ``candidate`` reproduces ``reference`` on ``domain`` (relative RMS <= rtol).
    Invalid candidates (unparseable, disallowed constructs) do not match."""
    try:
        return compare(candidate, reference, domain, variable)["relative_rms"] <= rtol
    except (ExpressionError, RecursionError, OverflowError):
        return False


def format_number(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"
