# tests/test_ademp.py - simulation studies: performance measures and their Monte Carlo SEs
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scidatasim.ademp import Estimate, SimulationStudy, n_sim_for, performance

LINEAR = {"experiment_type": "scidatasim.experiments.linear_function.logic.LinearFunctionExperiment",
          "n_points": 30, "m": 2.0, "c": 1.0, "x_min": 0.0, "x_max": 10.0,
          "observation": {"operators": [{"type": "gaussian", "sigma": 1.0}]}}


def ols(data):
    """Ordinary least squares slope with its t interval."""
    fit = stats.linregress(data["x"], data["y_observed"])
    t = stats.t.ppf(0.975, len(data["x"]) - 2)
    return {"slope": Estimate(fit.slope, fit.stderr, fit.slope - t * fit.stderr, fit.slope + t * fit.stderr)}


def _fake(errors, truth=1.0, se=None, lower=None, upper=None, p=None):
    n = len(errors)
    nan = np.full(n, np.nan)
    return pd.DataFrame({"design_point": 0, "method": "m", "estimand": "e", "truth": truth,
                         "estimate": truth + np.asarray(errors), "se": nan if se is None else se,
                         "lower": nan if lower is None else lower, "upper": nan if upper is None else upper,
                         "p_value": nan if p is None else p})


def test_mcse_formulas_match_the_spread_over_repeated_studies():
    """The MCSE of a measure should equal the standard deviation of that measure across
    independent repetitions of the whole study."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):
        err = rng.normal(0.2, 1.0, size=60)
        se = np.abs(rng.normal(1.0, 0.1, size=60))
        rows.append(performance(_fake(err, se=se, lower=1 + err - 1.96 * se, upper=1 + err + 1.96 * se)).iloc[0])
    table = pd.DataFrame(rows)
    for measure in ("bias", "empse", "mse", "coverage", "relative_error_modse"):
        spread, reported = table[measure].std(), table[f"{measure}_mcse"].mean()
        assert reported == pytest.approx(spread, rel=0.2), measure
    assert table["bias"].mean() == pytest.approx(0.2, abs=0.02)


def test_known_case_has_nominal_coverage_and_size():
    study = SimulationStudy(aims="OLS slope under Gaussian noise", base_config=LINEAR,
                            mechanisms=pd.DataFrame({"n_points": [30]}),
                            estimands={"slope": "m"}, methods={"ols": ols}, n_sim=400, base_seed=1,
                            null={"slope": 2.0})
    result = study.run()
    row = result.performance.iloc[0]
    assert abs(row["bias"]) < 3 * row["bias_mcse"]
    assert abs(row["coverage"] - 0.95) < 3 * row["coverage_mcse"]
    assert abs(row["relative_error_modse"]) < 3 * row["relative_error_modse_mcse"]
    assert abs(row["rejection"] - 0.05) < 3 * row["rejection_mcse"]  # the null is true: the size
    assert row["n_failed"] == 0 and row["truth"] == 2.0


def test_methods_see_only_observations_and_failures_are_counted():
    seen = []

    def peek(data):
        seen.append(set(data))
        return {"slope": 2.0}

    def flaky(data):
        if data["y_observed"][0] > 1.0:
            raise RuntimeError("did not converge")
        return {"slope": np.nan}

    study = SimulationStudy(aims="checks", base_config=LINEAR, mechanisms=pd.DataFrame({"n_points": [20, 40]}),
                            estimands={"slope": "m", "intercept": "c"}, methods={"peek": peek, "flaky": flaky},
                            n_sim=10, base_seed=2)
    result = study.run()
    assert all(keys == {"x", "y_observed"} for keys in seen)  # no latent y, no parameters
    flaky_rows = result.performance[result.performance["method"] == "flaky"]
    assert (flaky_rows["n_failed"] == 10).all()
    errors = result.estimates[result.estimates["method"] == "flaky"]["error"]
    assert errors.str.contains("did not converge").any() and errors.notna().all()
    peek_intercept = result.performance.query("method == 'peek' and estimand == 'intercept'")
    assert (peek_intercept["n_failed"] == 10).all()  # an estimand the method does not return


def test_study_is_reproducible_and_written(tmp_path):
    def make():
        return SimulationStudy(aims="r", base_config=LINEAR, mechanisms=pd.DataFrame({"n_points": [15]}),
                               estimands={"slope": "m"}, methods={"ols": ols}, n_sim=5, base_seed=9)
    a, b = make().run(str(tmp_path)), make().run()
    pd.testing.assert_frame_equal(a.estimates, b.estimates)
    for name in ("estimates.csv", "performance.csv", "mechanisms.csv", "study.json", "report.md"):
        assert (tmp_path / name).exists()
    seed = int(a.estimates["seed"].iloc[2])  # any repetition re-runs alone from its seed
    from scidatasim.engine import simulate
    sim = simulate(dict(LINEAR, n_points=15, seed=seed, generate_plots=False))
    assert ols({"x": sim.results["x"]["data"], "y_observed": sim.results["y_observed"]["data"]})["slope"].estimate \
        == pytest.approx(a.estimates["estimate"].iloc[2])
    assert "| ols |" in (tmp_path / "report.md").read_text()


def test_n_sim_for():
    assert n_sim_for(0.005, "coverage") == 1900
    assert n_sim_for(0.01, "bias", sd=0.5) == 2500
    assert n_sim_for(0.025, "rejection") == 400
    with pytest.raises(ValueError):
        n_sim_for(0.01, "bias")
    with pytest.raises(ValueError):
        SimulationStudy(aims="", base_config=LINEAR, mechanisms=pd.DataFrame({"n_points": [5]}),
                        estimands={"slope": "m"}, methods={"ols": ols}, n_sim=1).run()
    assert math.isfinite(n_sim_for(0.01, "coverage", coverage=0.9))
