# scidatasim/experiments/law_discovery/logic.py
"""Hidden laws y = f(x) for law-discovery and symbolic-regression benchmarks.

A run draws (or is given) a law family and its parameters, evaluates the law on
a grid of x and returns y. With ``counterfactual: true`` a *structural* constant
of the law (the 2 of an inverse-square law, the exponent of a Hill curve, ...)
is moved away from its textbook value, so the law cannot be recalled from
memory - it has to be discovered from the data (the idea behind counterfactual
law benchmarks such as NewtonBench).

Config::

    law: power_law            # a family from LAWS, or "random"
    counterfactual: false     # alter the structural constants
    parameters: {a: 2.0}      # optional: fix some parameter values
    parameter_questions: false  # bundles: also ask for the parameters (reveals the family!)
    formula_tolerance: auto   # or a number: the formula question's relative RMS tolerance
    law_seed: 17              # optional: draw family/parameters from this seed instead of the run seed
    n_points: 50
    x_range: [0.5, 8.0]       # default: DEFAULT_DOMAIN for every family (or x_min / x_max);
                              # "natural": the family's own domain (reveals the family!)

Outputs: x (coordinate), y (latent), and labels law_family and law_expression
(the exact law with numbers, e.g. "2.31*x**1.47"). Bundles ask for the family,
the parameters, an extrapolated value and the formula itself.

The formula is scored by :func:`scidatasim.expressions.matches` on the data range *and* half of
the range beyond it. Scored on the data range alone, a polynomial fitted to the data passes for
five of the nine families: it reproduces a smooth curve without finding the law. Power laws stay
partly within its reach: each formula question carries a private ``polynomial_reachable`` label,
and reports give accuracy with and without those items. The tolerance is "auto" by default: the larger of 1 % and the error that a
least-squares fit of the true family reaches in 90 % of replicates of the run's own noise
(residuals resampled). A fixed 1 % cannot be met at 20 dB even with the right law.

The extrapolation question asks for y half the data range beyond the data
(``extrapolation_distance``), within the larger of 5 % of the true value and the error a fit of
the true family reaches at that x under the run's noise (``extrapolation_tolerance: auto``). The
0.7.0rc2 rule (a quarter of the range; at least 1 % of the largest observed |y|; ``legacy``) let
the answer 0 pass a third of the questions and the last observed value 57 %
(``tools/study_extrapolation_scoring.py``). Private labels mark the items that still pass without
the law: ``zero_reachable`` and ``naive_reachable`` (plateaus change little beyond the data).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scidatasim.base import ExperimentLogic
from scidatasim.expressions import compile_expression, evaluate, format_number
from scidatasim.utils import DataDescriptor, DataType

#: Default x range, shared by all families: a family-specific range would identify the family.
DEFAULT_DOMAIN = (0.5, 8.0)
#: The extrapolation question asks for y this fraction of the x range beyond the data (config
#: ``extrapolation_distance``). 0.5 is where the formula's scored range ends; 0.7.0rc2 used 0.25,
#: where the last observed value passed 57 % of the questions (tools/study_extrapolation_scoring.py).
EXTRAPOLATION = 0.5
#: The extrapolation tolerance never goes below this share of |y*|.
EXTRAPOLATION_FLOOR = 0.05
#: The formula question is scored on the data range and this fraction of it beyond. Inside the
#: data range alone a polynomial fitted to the data passes for five of the nine families; at 0.5
#: only power laws remain within reach (their exponents are often close to 1 or 2).
FORMULA_EXTENSION = 0.5
#: The formula tolerance never goes below this relative RMS error.
FORMULA_FLOOR = 0.01
#: Noise replicates and the quantile of the true family's fit error that sets the tolerance.
TOLERANCE_REPLICATES = 30
TOLERANCE_QUANTILE = 0.9

# family: (template, parameter priors, structural constants {name: (textbook, counterfactual range)}, domain)
LAWS: Dict[str, Dict[str, Any]] = {
    "power_law": {"template": "{a}*x**{b}", "priors": {"a": (0.5, 5.0)},
                  "structural": {"b": (1.5, (0.6, 2.6))}, "domain": (0.5, 5.0)},
    "inverse_square": {"template": "{a}/x**{p}", "priors": {"a": (1.0, 20.0)},
                       "structural": {"p": (2.0, (1.4, 2.8))}, "domain": (0.5, 5.0)},
    "exponential_decay": {"template": "{a}*exp(-{k}*x)", "priors": {"a": (1.0, 10.0), "k": (0.2, 1.5)},
                          "structural": {}, "domain": (0.0, 6.0)},
    "logistic": {"template": "{K}/(1 + exp(-{r}*(x - {x0})))",
                 "priors": {"K": (2.0, 10.0), "r": (0.8, 3.0), "x0": (2.0, 4.0)}, "structural": {},
                 "domain": (0.0, 6.0)},
    "michaelis_menten": {"template": "{vmax}*x/({km} + x)", "priors": {"vmax": (1.0, 10.0), "km": (0.3, 3.0)},
                         "structural": {}, "domain": (0.0, 10.0)},
    "hill": {"template": "{vmax}*x**{n}/({k}**{n} + x**{n})", "priors": {"vmax": (1.0, 10.0), "k": (1.0, 4.0)},
             "structural": {"n": (2.0, (1.3, 3.5))}, "domain": (0.0, 8.0)},
    "damped_oscillation": {"template": "{a}*exp(-{g}*x)*cos({w}*x)",
                           "priors": {"a": (1.0, 5.0), "g": (0.1, 0.6), "w": (1.0, 4.0)}, "structural": {},
                           "domain": (0.0, 8.0)},
    "gaussian_peak": {"template": "{a}*exp(-(x - {mu})**2/(2*{s}**2))",
                      "priors": {"a": (1.0, 5.0), "mu": (2.0, 4.0), "s": (0.4, 1.2)},
                      "structural": {}, "domain": (0.0, 6.0)},
    "arrhenius": {"template": "{a}*exp(-{ea}/x)", "priors": {"a": (1.0, 10.0), "ea": (0.5, 4.0)},
                  "structural": {}, "domain": (0.5, 8.0)},
}


def law_expression(family: str, values: Dict[str, float], digits: int = 6) -> str:
    """The law with numbers substituted (``digits`` significant digits; use 17 for
    full precision, e.g. inside a fitting loop)."""
    spec = LAWS[family]

    def fmt(v):
        text = format_number(v, digits)
        return f"({text})" if text.startswith("-") else text  # -2**2 would be -4

    return spec["template"].format(**{k: fmt(v) for k, v in values.items()})


def law_function(family: str):
    """``(names, f)`` for a family: its constants (free ones, then structural) and ``f(x, *values)``,
    compiled once (fast enough for least-squares loops)."""
    spec = LAWS[family]
    names = list(spec["priors"]) + list(spec["structural"])
    compiled = compile_expression(spec["template"].replace("{", "").replace("}", ""), ["x", *names])

    def f(x, *values):
        return compiled(x=x, **dict(zip(names, values)))

    return names, f


def fit_law(family: str, x, y, start: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]:
    """Least-squares constants of ``family`` for the points (x, y), all constants positive (as in
    the catalogue). Starts from ``start`` (default: the prior means and textbook constants).
    ``None`` if the fit fails."""
    import warnings

    from scipy.optimize import curve_fit
    spec = LAWS[family]
    names, f = law_function(family)
    if start is None:
        start = {**{k: float(np.mean(v)) for k, v in spec["priors"].items()},
                 **{k: float(v[0]) for k, v in spec["structural"].items()}}
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() <= len(names):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values, _ = curve_fit(f, x[ok], y[ok], p0=[max(float(start[n]), 1e-6) for n in names],
                                  bounds=(1e-9, np.inf), max_nfev=2000)
    except (RuntimeError, ValueError, TypeError):
        return None
    return dict(zip(names, (float(v) for v in values)))


def achievable_tolerance(family: str, values: Dict[str, float], x, y_observed, domain: Tuple[float, float],
                         rng: np.random.Generator, replicates: int = TOLERANCE_REPLICATES,
                         quantile: float = TOLERANCE_QUANTILE, floor: float = FORMULA_FLOOR) -> float:
    """The relative RMS error (on ``domain``) that a least-squares fit of the *true* family reaches
    in ``quantile`` of noise replicates, and at least ``floor``.

    Replicates resample the run's own residuals (observed minus true values) onto the true curve, so
    any noise model counts. A formula tolerance below this would fail a solver that found the law."""
    names, f = law_function(family)
    truth = [values[n] for n in names]
    x, y_observed = np.asarray(x, float), np.asarray(y_observed, float)
    ok = np.isfinite(x) & np.isfinite(y_observed)
    x, y_observed = x[ok], y_observed[ok]
    clean = f(x, *truth)
    residuals = y_observed - clean
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0 or not np.any(residuals):
        return floor
    grid = np.linspace(float(domain[0]), float(domain[1]), 200)
    reference = f(grid, *truth)
    keep = np.isfinite(reference)
    scale = float(np.sqrt(np.mean(reference[keep] ** 2))) or 1.0
    start = dict(zip(names, truth))
    errors = []
    for _ in range(int(replicates)):
        fitted = fit_law(family, x, clean + rng.choice(residuals, size=x.size, replace=True), start)
        if fitted is None:
            errors.append(np.inf)
            continue
        guess = f(grid, *[fitted[n] for n in names])
        diff = guess[keep] - reference[keep]
        errors.append(float(np.sqrt(np.mean(diff ** 2)) / scale) if np.all(np.isfinite(diff)) else np.inf)
    achieved = float(np.quantile(errors, quantile))
    return max(floor, achieved) if np.isfinite(achieved) else floor


def achievable_point_error(family: str, values: Dict[str, float], x, y_observed, x_star: float,
                           rng: np.random.Generator, replicates: int = TOLERANCE_REPLICATES,
                           quantile: float = TOLERANCE_QUANTILE) -> float:
    """|y(x_star)| error that a least-squares fit of the *true* family reaches at ``x_star`` in
    ``quantile`` of noise replicates (the run's residuals resampled onto the true curve, as in
    :func:`achievable_tolerance`). 0 without noise. An extrapolation tolerance below it would fail a
    solver that found the law; one tied to the data's scale instead lets 0 pass laws that have
    decayed (a third of the questions in 0.7.0rc2)."""
    names, f = law_function(family)
    truth = [values[n] for n in names]
    x, y_observed = np.asarray(x, float), np.asarray(y_observed, float)
    ok = np.isfinite(x) & np.isfinite(y_observed)
    x, y_observed = x[ok], y_observed[ok]
    clean = f(x, *truth)
    residuals = y_observed - clean
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0 or not np.any(residuals):
        return 0.0
    target = float(f(np.array([float(x_star)]), *truth)[0])
    start = dict(zip(names, truth))
    errors = []
    for _ in range(int(replicates)):
        fitted = fit_law(family, x, clean + rng.choice(residuals, size=x.size, replace=True), start)
        guess = float(f(np.array([float(x_star)]), *[fitted[n] for n in names])[0]) if fitted else np.inf
        errors.append(abs(guess - target) if np.isfinite(guess) else np.inf)
    achieved = float(np.quantile(errors, quantile))
    return achieved if np.isfinite(achieved) else 0.0


def naive_reachable(item: Dict[str, Any], x, y) -> bool:
    """Whether an extrapolation question ``item`` passes with an answer that ignores the law: the
    last observed value (mean of the last three points) or a straight line through the last
    quarter of the data. Private (it hints at the family: plateaus and near-linear laws), so
    reports give accuracy with and without these items."""
    from scidatasim.bundle import check_answer
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return False
    x_star = float(re.findall(r"x = ([-+0-9.e]+)", item["question"])[0])
    order = np.argsort(x)
    last = float(np.mean(y[order[-3:]]))
    tail = x >= x.min() + 0.75 * (x.max() - x.min())
    line = (float(np.polyval(np.polyfit(x[tail], y[tail], 1), x_star)) if tail.sum() >= 2 else last)
    return bool(check_answer(item, last) or check_answer(item, line))


def polynomial_reachable(item: Dict[str, Any], results) -> bool:
    """Whether a polynomial fitted to the run's observations (degree by BIC) passes the formula
    question ``item``. Stored with the question as a *private* label (it hints at the family):
    such items can be met by curve fitting, so reports give accuracy with and without them. Power
    laws with exponents near 1 or 2 are the usual case."""
    from scidatasim.bundle import check_answer
    from scidatasim.diagnostics.shortcuts import polynomial_formula
    observed = _observed_series(results)
    x, y = observed if observed is not None else (results["x"]["data"], results["y"]["data"])
    formula = polynomial_formula(x, y)
    return bool(formula is not None and check_answer(item, formula))


def _observed_series(results) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(x, y) of the observed y series, if the run has an observation step."""
    from scidatasim.utils import Role
    for name, info in results.items():
        d = info["descriptor"]
        if name != "y" and name.startswith("y") and d.role is Role.OBSERVATION and d.x_axis in results:
            return np.asarray(results[d.x_axis]["data"], float), np.asarray(info["data"], float)
    return None


class HiddenLawExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = "Closed-form laws evaluated exactly; the formula label reproduces y (tests)."

    def __init__(self, config):
        self.law = config.get("law", "random")
        if self.law != "random" and self.law not in LAWS:
            raise ValueError(f"law must be 'random' or one of {sorted(LAWS)}")
        self.counterfactual = bool(config.get("counterfactual", False))
        self.fixed = dict(config.get("parameters") or {})
        if self.law == "random" and self.fixed:
            raise ValueError("'parameters' needs a fixed law family (parameter names differ between families)")
        self.n_points = int(config.get("n_points", 50))
        if self.n_points < 3:
            raise ValueError("n_points must be >= 3")
        self.x_range = config.get("x_range")
        self.law_seed = config.get("law_seed")
        # Parameter questions name the law family and template, which would give away
        # the family and formula questions of the same run; off unless asked for.
        self.parameter_questions = bool(config.get("parameter_questions", False))
        self.formula_tolerance = config.get("formula_tolerance", "auto")
        self.extrapolation_distance = float(config.get("extrapolation_distance", EXTRAPOLATION))
        if not self.extrapolation_distance > 0:
            raise ValueError("extrapolation_distance must be > 0 (a share of the x range beyond the data)")
        self.extrapolation_tolerance = config.get("extrapolation_tolerance", "auto")
        if self.extrapolation_tolerance not in ("auto", "legacy"):
            raise ValueError("extrapolation_tolerance must be 'auto' or 'legacy'")
        if self.formula_tolerance != "auto" and not (isinstance(self.formula_tolerance, (int, float))
                                                     and self.formula_tolerance > 0):
            raise ValueError("formula_tolerance must be 'auto' or a positive number")
        if config.get("x_min") is not None or config.get("x_max") is not None:
            if config.get("x_min") is None or config.get("x_max") is None:
                raise ValueError("give both x_min and x_max (or x_range)")
            self.x_range = (float(config["x_min"]), float(config["x_max"]))
        self.family = None
        self.values: Dict[str, float] = {}

    # ------------------------------------------------------------------ drawing the law
    def _draw(self, rng) -> Tuple[str, Dict[str, float]]:
        family = self.law if self.law != "random" else sorted(LAWS)[int(rng.integers(len(LAWS)))]
        spec = LAWS[family]
        values: Dict[str, float] = {}
        for name, (lo, hi) in spec["priors"].items():
            values[name] = float(self.fixed[name]) if name in self.fixed else float(rng.uniform(lo, hi))
        for name, (textbook, (lo, hi)) in spec["structural"].items():
            if name in self.fixed:
                values[name] = float(self.fixed[name])
            elif self.counterfactual:
                # keep clear of the textbook value so memorised laws do not fit
                for _ in range(100):
                    value = float(rng.uniform(lo, hi))
                    if abs(value - textbook) >= 0.15 * abs(textbook):
                        break
                values[name] = round(value, 3)
            else:
                values[name] = float(textbook)
        unknown = set(self.fixed) - set(values)
        if unknown:
            raise ValueError(f"Unknown parameters for {family}: {sorted(unknown)}")
        return family, values

    def initialize(self, config):
        rng = np.random.default_rng(int(self.law_seed)) if self.law_seed is not None else self.get_rng(config)
        self.family, self.values = self._draw(rng)
        if self.x_range is None:
            lo, hi = DEFAULT_DOMAIN
        elif isinstance(self.x_range, str):
            if self.x_range != "natural":
                raise ValueError("x_range must be [low, high] or 'natural'")
            lo, hi = LAWS[self.family]["domain"]
        else:
            lo, hi = self.x_range
        if not float(lo) < float(hi):
            raise ValueError("x_range must satisfy low < high")
        self.domain = (float(lo), float(hi))
        return {}

    def run_step(self, state, step):
        return state

    # ------------------------------------------------------------------ results
    def achievable_error(self, x, y_observed, domain, rng):
        """The error a least-squares fit of the true law reaches from these measurements in
        ``TOLERANCE_QUANTILE`` of replicates of their noise (see :func:`achievable_tolerance`)."""
        return achievable_tolerance(self.family, self.values, x, y_observed, tuple(domain), rng, floor=0.0)

    def expression(self) -> str:
        return law_expression(self.family, self.values)

    def get_results(self):
        if self.family is None:
            self.initialize({})
        x = np.linspace(self.domain[0], self.domain[1], self.n_points)
        y = evaluate(self.expression(), x=x)
        if not np.all(np.isfinite(y)):
            raise ValueError(f"the {self.family} law {self.expression()} is not finite on x in "
                             f"[{self.domain[0]:g}, {self.domain[1]:g}]; choose another x_range")
        return {
            "x": {"data": x, "descriptor": DataDescriptor("x", DataType.NDARRAY, units="dimensionless",
                                                          group="time_series", role="coordinate")},
            "y": {"data": y, "descriptor": DataDescriptor("y", DataType.NDARRAY, units="dimensionless",
                                                          group="time_series", plot_type="line", x_axis="x",
                                                          role="latent")},
            "law_family": {"data": self.family, "descriptor": DataDescriptor("law_family", DataType.STRING,
                                                                             role="label")},
            "law_expression": {"data": self.expression(), "descriptor": DataDescriptor(
                "law_expression", DataType.STRING, role="label")},
        }

    def ground_truth_parameters(self, config):
        params = super().ground_truth_parameters(config)
        params.update({f"law.{k}": v for k, v in self.values.items()})
        params["law.family"] = self.family
        params["law.expression"] = self.expression()
        params["law.counterfactual"] = self.counterfactual
        return params

    # ------------------------------------------------------------------ bundle hooks
    def questions(self, results, rng) -> List[Dict[str, Any]]:
        lo, hi = self.domain
        items: List[Dict[str, Any]] = []
        families = sorted(LAWS)
        others = [f for f in families if f != self.family]
        picks = sorted([self.family] + list(rng.choice(others, size=3, replace=False)))
        items.append(dict(template="law_family", question="Which family of laws generated y(x)? Choose one of: "
                                                            + ", ".join(picks) + ".",
                          answer=self.family, answer_type="choice", choices=picks, tolerance=None,
                          tolerance_rule="exact", answerable_from="inference"))
        x_star = hi + self.extrapolation_distance * (hi - lo)
        # Scored beyond the data: inside the data range a flexible curve (a polynomial) can pass
        # without the law. The tolerance is what the true law achieves under this run's noise.
        x_end = hi + FORMULA_EXTENSION * (hi - lo)
        score_domain = (lo, x_end)
        if self.formula_tolerance == "auto":
            observed = _observed_series(results)
            tolerance = (FORMULA_FLOOR if observed is None else achievable_tolerance(
                self.family, self.values, observed[0], observed[1], score_domain,
                np.random.default_rng(int(rng.integers(2 ** 63)))))
            rule = (f"relative RMS error on the stated range at most the larger of {FORMULA_FLOOR:.0%} and the error "
                    f"a least-squares fit of the true law reaches in {TOLERANCE_QUANTILE:.0%} of replicates of "
                    "this run's noise")
        else:
            tolerance, rule = float(self.formula_tolerance), "relative RMS error on the stated range (fixed)"
        formula_item = dict(template="law_formula",
                          question=f"Give a formula for y as a function of x (e.g. 2.5*x**1.3). It is scored on "
                                   f"[{format_number(lo)}, {format_number(x_end, 4)}]: the observed range "
                                   f"[{format_number(lo)}, {format_number(hi)}] and half of it beyond.",
                          answer=self.expression(), answer_type="expression", tolerance=round(tolerance, 6),
                          tolerance_rule=rule, domain=[lo, x_end], variable="x", answerable_from="inference")
        formula_item["polynomial_reachable"] = polynomial_reachable(formula_item, results)
        items.append(formula_item)
        y_star = float(evaluate(self.expression(), x=x_star))
        if np.isfinite(y_star):
            observed = _observed_series(results)
            if self.extrapolation_tolerance == "legacy":  # 0.7.0rc2
                tolerance = max(0.05 * abs(y_star), 0.01 * _finite_max_abs(results["y"]["data"]))
                rule = "5 % of the true value (at least 1 % of the largest observed |y|)"
            else:
                miss = (0.0 if observed is None else achievable_point_error(
                    self.family, self.values, observed[0], observed[1], x_star,
                    np.random.default_rng(int(rng.integers(2 ** 63)))))
                tolerance = max(EXTRAPOLATION_FLOOR * abs(y_star), miss)
                rule = (f"{EXTRAPOLATION_FLOOR:.0%} of the true value, or the error a least-squares fit of the true "
                        f"law reaches at this x in {TOLERANCE_QUANTILE:.0%} of replicates of this run's noise, "
                        "whichever is larger")
            extrapolation = dict(template="extrapolation",
                                 question=f"Predict y at x = {format_number(x_star, 4)} (outside the observed range).",
                                 answer=y_star, answer_type="number", tolerance=tolerance, tolerance_rule=rule,
                                 answerable_from="inference")
            # private labels: answers that ignore the law pass. 0 (the law has decayed below the
            # tolerance), or the last observed value / a line through the last quarter of the data
            # (the law changes little beyond the data, e.g. a plateau)
            extrapolation["zero_reachable"] = bool(abs(y_star) <= tolerance)
            extrapolation["naive_reachable"] = naive_reachable(
                extrapolation, *(observed if observed is not None else (results["x"]["data"], results["y"]["data"])))
            items.append(extrapolation)
        for name, value in (self.values.items() if self.parameter_questions else ()):
            if name in LAWS[self.family]["structural"] and not self.counterfactual:
                continue  # textbook constants are not informative questions
            items.append(dict(template="law_parameter", parameter=name,
                              question=f"In the {self.family.replace('_', ' ')} law "
                                       f"y = {LAWS[self.family]['template'].replace('{', '').replace('}', '')}, "
                                       f"estimate {name}.",
                              answer=value, answer_type="number", tolerance=max(0.05 * abs(value), 1e-6),
                              tolerance_rule="5 % of the true value", answerable_from="inference"))
        for item in items:
            item.setdefault("series", "y")
        return items


def textbook_counterpart(family: str, expression: str, domain: Tuple[float, float], n: int = 200
                         ) -> Optional[str]:
    """The textbook version of a law that best imitates it: structural constants fixed at their
    textbook values, the other constants least-squares fitted to ``expression`` on ``domain``.

    This is the answer of a solver that recalls the textbook form and fits only its free
    constants - the "prior over evidence" error that counterfactual laws are built to expose.
    ``None`` for families without structural constants.
    """
    from scipy.optimize import curve_fit
    spec = LAWS[family]
    if not spec["structural"]:
        return None
    x = np.linspace(float(domain[0]), float(domain[1]), n)
    y = evaluate(expression, x=x)
    names = list(spec["priors"])
    fixed = {k: v[0] for k, v in spec["structural"].items()}

    def model(xv, *p):
        return evaluate(law_expression(family, {**dict(zip(names, p)), **fixed}, digits=17), x=xv)

    start = [float(np.mean(spec["priors"][k])) for k in names]
    with np.errstate(all="ignore"):
        params, _ = curve_fit(model, x, y, p0=start, bounds=(1e-9, np.inf), max_nfev=5000)
    return law_expression(family, {**dict(zip(names, params)), **fixed})


def _finite_max_abs(values) -> float:
    values = np.abs(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else 0.0


def law_suite_design(families=None, counterfactual=(False, True), snr_db=(40, 20), shuffle=True, seed=None):
    """Design table for a law-discovery suite: every family x counterfactual x noise level.

    Rows are shuffled (``shuffle=True``) with ``seed``; the default ``seed=None`` draws a fresh,
    unrecorded order. A known row order would let anyone who sees a run's design-point index
    (or the order of runs) read off its hidden family. The design directory still records the
    table, privately.

    Use with :func:`scidatasim.doe.run_design` (``replicates`` there gives each
    row several independently drawn laws)::

        design = law_suite_design(snr_db=(40, 20, 10))
        run_design(base_config, design, output_dir="laws", replicates=5)

    The base config should define ``observation.operators[0]`` as a gaussian
    operator; its ``snr_db`` is swept.
    """
    import itertools

    import pandas as pd
    families = sorted(families or LAWS)
    unknown = set(families) - set(LAWS)
    if unknown:
        raise ValueError(f"Unknown families {sorted(unknown)}")
    rows = [{"law": f, "counterfactual": bool(c), "observation.operators.0.snr_db": s}
            for f, c, s in itertools.product(families, counterfactual, snr_db)]
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(rows))
        rows = [rows[i] for i in order]
    return pd.DataFrame(rows)
