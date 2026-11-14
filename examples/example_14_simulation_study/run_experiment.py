"""Example 14: a simulation study in the ADEMP structure.

Aims: how well do three slope estimators recover the slope of a straight line when the
measurements have outliers or a detection limit?

* Data-generating mechanisms: y = 2 x + 1 on [0, 10] with Gaussian noise (sigma 1), crossed
  with n = 20 or 100 points and three measurement problems: none, 10 % outliers (5 SD), and a
  detection limit at y = 6 (values below it reported as 6).
* Estimand: the slope m (true value 2).
* Methods: ordinary least squares (t interval), Theil-Sen (its own 95 % interval) and Huber
  regression (no interval).
* Performance: bias, empirical SE, MSE, coverage and the error of the model SE, each with its
  Monte Carlo SE; the number of repetitions is chosen for an MCSE of coverage near 0.01.

    python -m examples.example_14_simulation_study.run_experiment            # about 10 s
    python -m examples.example_14_simulation_study.run_experiment --quick    # about 3 s
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scidatasim.ademp import Estimate, SimulationStudy, n_sim_for  # noqa: E402
from scidatasim.examples import prepare_output_dir  # noqa: E402

BASE = {
    "experiment_type": "scidatasim.experiments.linear_function.logic.LinearFunctionExperiment",
    "n_points": 20, "m": 2.0, "c": 1.0, "x_min": 0.0, "x_max": 10.0,
    "observation": {"operators": [
        {"type": "gaussian", "sigma": 1.0},
        {"type": "outliers", "rate": 0.0, "scale": 5.0},
        {"type": "censor", "low": -1.0e9, "replace": "limit"},  # off unless a mechanism sets 'low'
    ]},
}


def ols(data):
    """Ordinary least squares, with a t interval."""
    x, y = data["x"], data["y_observed"]
    fit = stats.linregress(x, y)
    t = stats.t.ppf(0.975, len(x) - 2)
    return {"slope": Estimate(fit.slope, fit.stderr, fit.slope - t * fit.stderr, fit.slope + t * fit.stderr,
                              fit.pvalue)}


def theil_sen(data):
    """Theil-Sen: the median of pairwise slopes, with its 95 % interval."""
    slope, _, lower, upper = stats.theilslopes(data["y_observed"], data["x"], alpha=0.95)
    return {"slope": Estimate(slope, lower=lower, upper=upper)}


def huber(data):
    """Huber regression (robust loss, scale from the median absolute deviation); no interval."""
    x, y = data["x"], data["y_observed"]
    start = np.polyfit(x, y, 1)
    scale = 1.4826 * np.median(np.abs(y - np.polyval(start, x))) or 1.0
    fit = optimize.least_squares(lambda p: (y - (p[0] * x + p[1])) / scale, start, loss="huber", f_scale=1.345)
    if not fit.success:
        raise RuntimeError(fit.message)
    return {"slope": fit.x[0]}


def mechanisms():
    problems = {"none": {}, "outliers 10 %": {"observation.operators.1.rate": 0.1},
                "detection limit y < 6": {"observation.operators.2.low": 6.0}}
    rows = []
    for n in (20, 100):
        for name, overrides in problems.items():
            rows.append({"n_points": n, "observation.operators.1.rate": 0.0,
                         "observation.operators.2.low": -1.0e9, **overrides, "problem": name})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_14_simulation_study"))
    parser.add_argument("--quick", action="store_true", help="100 repetitions instead of ~475")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    out = prepare_output_dir(args.output_dir)  # keeps the runner's run.log / run.json
    n_sim = 100 if args.quick else n_sim_for(0.01, "coverage")  # 475: MCSE of 95 % coverage ~ 0.01
    table = mechanisms()
    labels = table.pop("problem")
    study = SimulationStudy(mechanism_labels=[f"n = {n}, {p}" for n, p in zip(table["n_points"], labels)],
        aims="Recovering a straight line's slope when measurements have outliers or a detection limit.",
        base_config=BASE, mechanisms=table, estimands={"slope": "m"},
        methods={"OLS": ols, "Theil-Sen": theil_sen, "Huber": huber}, n_sim=n_sim, base_seed=args.seed)
    result = study.run(out)
    perf = result.performance.merge(pd.DataFrame({"design_point": range(len(labels)), "problem": labels,
                                                  "n": table["n_points"]}), on="design_point")
    print(f"{len(table)} mechanisms x {n_sim} repetitions x 3 methods\n")
    show = perf[["problem", "n", "method", "bias", "bias_mcse", "empse", "coverage", "coverage_mcse"]]
    with pd.option_context("display.width", 120):
        print(show.round(3).to_string(index=False))
    print(f"\nReport: {os.path.join(out, 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
