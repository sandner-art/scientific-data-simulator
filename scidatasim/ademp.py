# scidatasim/ademp.py
"""Simulation studies in the ADEMP structure, with Monte Carlo standard errors.

A simulation study compares statistical methods on data whose truth is known. Morris, White and
Crowther (2019, *Statistics in Medicine* 38:2074-2102) structure one as:

* **A**ims - what the study is for (a sentence, recorded);
* **D**ata-generating mechanisms - the rows of a design table, applied to a base configuration
  as overrides (any model, observation operator or law shift of the simulator);
* **E**stimands - quantities with a known true value in every run: a parameter of the run's
  ground truth (``"m"``), a label output, or a function of the run;
* **M**ethods - functions that estimate the estimands from what an analyst would see: the run's
  observations and coordinates only, never its ground truth;
* **P**erformance measures - bias, empirical and model-based standard errors, mean squared error,
  coverage, bias-eliminated coverage and rejection rate, each with its Monte Carlo standard error
  (MCSE), so a reader can tell a real difference between methods from simulation noise.

Every method sees the same simulated data set in a repetition, and every repetition has its own
recorded seed, so any one can be re-run alone. ::

    study = SimulationStudy(
        aims="Is Theil-Sen better than OLS for a slope under outliers?",
        base_config=config, mechanisms=design_table,
        estimands={"slope": "m"},
        methods={"ols": ols, "theil_sen": theil_sen},
        n_sim=n_sim_for(0.01, "coverage"))
    result = study.run("study_output")
    print(result.markdown())

A method returns, per estimand, an :class:`Estimate`, a number, an ``(estimate, se)`` pair or a
mapping with ``estimate`` and optionally ``se``, ``lower``, ``upper``, ``p_value``. A method that
raises, or returns no finite estimate, counts as a failure; failures are reported per method and
mechanism and left out of the measures, as Morris et al. recommend.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from .utils import Role

__all__ = ["Estimate", "SimulationStudy", "StudyResult", "performance", "n_sim_for"]

Estimand = Union[str, Callable[[Any], float]]
Method = Callable[[Dict[str, np.ndarray]], Mapping[str, Any]]


@dataclass
class Estimate:
    estimate: float
    se: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    p_value: Optional[float] = None


def _as_estimate(value: Any) -> Estimate:
    if isinstance(value, Estimate):
        return value
    if isinstance(value, Mapping):
        return Estimate(**{k: value.get(k) for k in ("estimate", "se", "lower", "upper", "p_value")})
    if isinstance(value, (tuple, list)):
        return Estimate(*value)
    return Estimate(float(value))


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


def _public_data(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, np.ndarray]:
    """What an analyst sees: observation and coordinate outputs, nothing of the ground truth."""
    return {name: np.asarray(info["data"]) for name, info in results.items()
            if info["descriptor"].role in (Role.OBSERVATION, Role.COORDINATE)}


def _truth(sim, estimand: Estimand) -> float:
    if callable(estimand):
        return float(estimand(sim))
    params = (sim.ground_truth or {}).get("parameters") or {}
    if estimand in params:
        return float(params[estimand])
    if estimand in sim.results:
        return float(np.asarray(sim.results[estimand]["data"]).reshape(-1)[0])
    raise KeyError(f"estimand {estimand!r} is neither a parameter nor an output of the run")


# ---------------------------------------------------------------------------- performance
def _measures(group: pd.DataFrame, alpha: float) -> Dict[str, Any]:
    ok = group[np.isfinite(group["estimate"])]
    n = len(ok)
    out: Dict[str, Any] = {"n_sim": len(group), "n_failed": int(len(group) - n)}
    if n < 2:
        return out
    est, truth = ok["estimate"].to_numpy(float), ok["truth"].to_numpy(float)
    err = est - truth
    constant = bool(np.allclose(truth, truth[0]))
    spread = est if constant else err  # with a varying truth, the spread of the errors
    empse = float(np.std(spread, ddof=1))
    bias = float(err.mean())
    out.update(truth=float(truth[0]) if constant else math.nan, truth_varies=not constant,
               bias=bias, bias_mcse=empse / math.sqrt(n),
               empse=empse, empse_mcse=empse / math.sqrt(2 * (n - 1)))
    if constant and truth[0] != 0:
        out.update(relative_bias=bias / truth[0], relative_bias_mcse=out["bias_mcse"] / abs(truth[0]))
    sq = err ** 2
    mse = float(sq.mean())
    out.update(mse=mse, mse_mcse=float(math.sqrt(np.sum((sq - mse) ** 2) / (n * (n - 1)))))
    se = ok["se"].to_numpy(float)
    if np.isfinite(se).all() and empse > 0:
        modse = float(math.sqrt(np.mean(se ** 2)))
        ratio = modse / empse
        out.update(modse=modse, relative_error_modse=100.0 * (ratio - 1.0),
                   relative_error_modse_mcse=100.0 * ratio * math.sqrt(
                       np.var(se ** 2, ddof=1) / (4 * n * modse ** 4) + 1.0 / (2 * (n - 1))))
    lower, upper = ok["lower"].to_numpy(float), ok["upper"].to_numpy(float)
    if np.isfinite(lower).all() and np.isfinite(upper).all():
        cover = float(np.mean((lower <= truth) & (truth <= upper)))
        # bias-eliminated: intervals shifted by the estimated bias; is their width right?
        be = float(np.mean((lower - bias <= truth) & (truth <= upper - bias)))
        out.update(coverage=cover, coverage_mcse=math.sqrt(cover * (1 - cover) / n),
                   be_coverage=be, be_coverage_mcse=math.sqrt(be * (1 - be) / n))
    p = ok["p_value"].to_numpy(float)
    if np.isfinite(p).all():
        rej = float(np.mean(p < alpha))
        out.update(rejection=rej, rejection_mcse=math.sqrt(rej * (1 - rej) / n))
    return out


def performance(estimates: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Performance measures with Monte Carlo standard errors, one row per mechanism, method and
    estimand (``estimates`` as written by :meth:`SimulationStudy.run`)."""
    rows = []
    for (point, method, estimand), group in estimates.groupby(["design_point", "method", "estimand"], sort=True):
        rows.append({"design_point": point, "method": method, "estimand": estimand, **_measures(group, alpha)})
    return pd.DataFrame(rows)


def n_sim_for(target_mcse: float, measure: str = "coverage", *, sd: Optional[float] = None,
              coverage: float = 0.95, rejection: float = 0.5) -> int:
    """Repetitions needed for a Monte Carlo SE of ``target_mcse`` (Morris et al. 2019, sec. 5.3).

    * ``"bias"``: needs ``sd``, the expected empirical SE of the estimates: n = (sd / target)^2;
    * ``"coverage"``: n = coverage (1 - coverage) / target^2 (0.95 by default);
    * ``"rejection"``: the same with the expected rejection rate (0.5 is the worst case).
    """
    if target_mcse <= 0:
        raise ValueError("target_mcse must be > 0")
    if measure == "bias":
        if sd is None or sd <= 0:
            raise ValueError("measure 'bias' needs sd > 0 (the expected spread of the estimates)")
        n = (sd / target_mcse) ** 2
    elif measure == "coverage":
        n = coverage * (1 - coverage) / target_mcse ** 2
    elif measure == "rejection":
        n = rejection * (1 - rejection) / target_mcse ** 2
    else:
        raise ValueError("measure must be 'bias', 'coverage' or 'rejection'")
    return int(math.ceil(round(n, 9)))  # 0.95 * 0.05 / 0.005 ** 2 is 1900, not 1900.0000000000002


# ---------------------------------------------------------------------------- study
@dataclass
class StudyResult:
    estimates: pd.DataFrame
    performance: pd.DataFrame
    mechanisms: pd.DataFrame
    aims: str
    output_dir: Optional[str] = None
    base_seed: Optional[int] = None

    def markdown(self, digits: int = 3) -> str:
        """The performance table, per estimand and mechanism: value (MCSE)."""
        def cell(row, key):
            value, mcse = row.get(key), row.get(f"{key}_mcse")
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return ""
            text = f"{value:.{digits}f}"
            return text + (f" ({mcse:.{digits}f})" if mcse is not None and np.isfinite(mcse) else "")
        factors = [c for c in self.mechanisms.columns if c not in ("design_point", "label")]
        lines = [f"Aims: {self.aims}", ""]
        perf = self.performance
        for estimand in sorted(perf["estimand"].unique()):
            for point in sorted(perf["design_point"].unique()):
                mech = self.mechanisms.loc[self.mechanisms["design_point"] == point].iloc[0]
                label = mech["label"] if "label" in mech and isinstance(mech["label"], str) else (
                    ", ".join(f"{f} = {mech[f]}" for f in factors) or f"mechanism {point}")
                lines += [f"### {estimand}: {label}", "",
                          "| method | bias | empirical SE | MSE | coverage | rel. error of model SE (%) | rejection | failed |",
                          "|---|---|---|---|---|---|---|---|"]
                for row in perf[(perf["estimand"] == estimand) & (perf["design_point"] == point)].to_dict("records"):
                    lines.append(f"| {row['method']} | {cell(row, 'bias')} | {cell(row, 'empse')} | {cell(row, 'mse')} | "
                                 f"{cell(row, 'coverage')} | {cell(row, 'relative_error_modse')} | "
                                 f"{cell(row, 'rejection')} | {row['n_failed']} of {row['n_sim']} |")
                lines.append("")
        lines.append("Values with their Monte Carlo standard errors in brackets.")
        return "\n".join(lines)


@dataclass
class SimulationStudy:
    """An ADEMP simulation study on top of the simulator (see the module docstring)."""
    aims: str
    base_config: Dict[str, Any]
    mechanisms: pd.DataFrame
    estimands: Mapping[str, Estimand]
    methods: Mapping[str, Method]
    n_sim: int = 100
    base_seed: Optional[int] = None
    alpha: float = 0.05
    #: estimand -> value under the null hypothesis, for the rejection rate when a method gives
    #: a standard error but no p-value (Wald test)
    null: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.95
    #: a readable name per mechanism (row of ``mechanisms``), for the report
    mechanism_labels: Optional[List[str]] = None

    def run(self, output_dir: Optional[str] = None) -> StudyResult:
        from scipy.stats import norm

        from .doe import _row_overrides, add_replicates_and_seeds, apply_overrides
        from .engine import simulate
        if self.n_sim < 2:
            raise ValueError("n_sim must be at least 2")
        if not self.methods or not self.estimands:
            raise ValueError("a study needs at least one method and one estimand")
        mechanisms = self.mechanisms.reset_index(drop=True).copy()
        if self.mechanism_labels is not None and len(self.mechanism_labels) != len(mechanisms):
            raise ValueError("mechanism_labels needs one label per mechanism")
        table = add_replicates_and_seeds(mechanisms, self.n_sim, self.base_seed, "per_row")
        base_seed = table.attrs["base_seed"]
        mechanisms.insert(0, "design_point", range(len(mechanisms)))
        if self.mechanism_labels is not None:
            mechanisms.insert(1, "label", list(self.mechanism_labels))
        z = norm.ppf(0.5 + self.confidence / 2)
        base = {k: v for k, v in self.base_config.items() if k != "bundle"}
        rows: List[Dict[str, Any]] = []
        for record in table.to_dict("records"):
            config = apply_overrides(base, _row_overrides(record))
            config["seed"] = int(record["seed"])
            config["generate_plots"] = False
            sim = simulate(config)
            truths = {name: _truth(sim, spec) for name, spec in self.estimands.items()}
            data = _public_data(sim.results)
            for method_name, method in self.methods.items():
                try:
                    output = method({k: v.copy() for k, v in data.items()}) or {}
                    error = None
                except Exception as e:  # a failure of the method, recorded and counted
                    output, error = {}, f"{type(e).__name__}: {e}"
                for name, truth in truths.items():
                    est = _as_estimate(output[name]) if name in output else Estimate(math.nan)
                    value, se = _num(est.estimate), _num(est.se)
                    lower, upper, p = _num(est.lower), _num(est.upper), _num(est.p_value)
                    if np.isfinite(se) and not (np.isfinite(lower) and np.isfinite(upper)):
                        lower, upper = value - z * se, value + z * se  # Wald interval
                    if not np.isfinite(p) and np.isfinite(se) and se > 0 and name in self.null:
                        p = float(2 * norm.sf(abs(value - self.null[name]) / se))
                    rows.append({"design_point": record["design_point"], "repetition": record["replicate"],
                                 "seed": int(record["seed"]), "method": method_name, "estimand": name,
                                 "truth": truth, "estimate": value, "se": se, "lower": lower, "upper": upper,
                                 "p_value": p, "error": error if error else (None if np.isfinite(value)
                                                                             else "no finite estimate")})
        estimates = pd.DataFrame(rows)
        perf = performance(estimates, self.alpha)
        result = StudyResult(estimates, perf, mechanisms, self.aims, output_dir, base_seed)
        if output_dir:
            self._write(result, output_dir)
        return result

    def _write(self, result: StudyResult, output_dir: str) -> None:
        from . import __version__
        os.makedirs(output_dir, exist_ok=True)
        result.estimates.to_csv(os.path.join(output_dir, "estimates.csv"), index=False)
        result.performance.to_csv(os.path.join(output_dir, "performance.csv"), index=False)
        result.mechanisms.to_csv(os.path.join(output_dir, "mechanisms.csv"), index=False)

        def describe(item):
            if callable(item):
                doc = (getattr(item, "__doc__", None) or "").strip().splitlines()
                return {"function": f"{getattr(item, '__module__', '?')}.{getattr(item, '__qualname__', repr(item))}",
                        "description": doc[0] if doc else ""}
            return {"ground_truth": item}
        study = {"aims": self.aims,
                 "data_generating_mechanisms": {"base_config": self.base_config,
                                                "design": result.mechanisms.to_dict("records")},
                 "estimands": {k: describe(v) for k, v in self.estimands.items()},
                 "methods": {k: describe(v) for k, v in self.methods.items()},
                 "performance_measures": ["bias", "empirical SE", "MSE", "relative error of model SE", "coverage",
                                          "bias-eliminated coverage", "rejection rate"],
                 "n_sim": self.n_sim, "base_seed": result.base_seed, "alpha": self.alpha,
                 "confidence": self.confidence, "null": dict(self.null), "simulator_version": __version__,
                 "reference": "Morris, White and Crowther (2019), Statistics in Medicine 38:2074-2102"}
        with open(os.path.join(output_dir, "study.json"), "w", encoding="utf-8") as f:
            json.dump(study, f, indent=2, default=str)
        with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write("# Simulation study\n\n" + result.markdown() + "\n")
