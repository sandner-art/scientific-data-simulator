# experiments/law_discovery/logic.py
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
    law_seed: 17              # optional: draw family/parameters from this seed instead of the run seed
    n_points: 50
    x_range: [0.5, 8.0]       # default: DEFAULT_DOMAIN for every family (or x_min / x_max);
                              # "natural": the family's own domain (reveals the family!)

Outputs: x (coordinate), y (latent), and labels law_family and law_expression
(the exact law with numbers, e.g. "2.31*x**1.47"). Bundles ask for the family,
the parameters, an extrapolated value and the formula itself (scored by
:func:`simulator.expressions.matches` on the data domain).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator.base import ExperimentLogic
from simulator.expressions import evaluate, format_number
from simulator.utils import DataDescriptor, DataType

#: Default x range, shared by all families: a family-specific range would identify the family.
DEFAULT_DOMAIN = (0.5, 8.0)

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
        items.append(dict(template="law_formula",
                          question=f"Give a formula for y as a function of x that reproduces the data on "
                                   f"[{format_number(lo)}, {format_number(hi)}] (e.g. 2.5*x**1.3).",
                          answer=self.expression(), answer_type="expression", tolerance=0.01,
                          tolerance_rule="relative RMS error <= 1 % on the data domain",
                          domain=[lo, hi], variable="x", answerable_from="inference"))
        x_star = hi + 0.25 * (hi - lo)
        y_star = float(evaluate(self.expression(), x=x_star))
        if np.isfinite(y_star):
            items.append(dict(template="extrapolation",
                              question=f"Predict y at x = {format_number(x_star, 4)} (outside the observed range).",
                              answer=y_star, answer_type="number",
                              tolerance=max(0.05 * abs(y_star), 0.01 * _finite_max_abs(results["y"]["data"])),
                              tolerance_rule="5 % of the true value (at least 1 % of the largest observed |y|)",
                              answerable_from="inference"))
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

    Use with :func:`simulator.doe.run_design` (``replicates`` there gives each
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
