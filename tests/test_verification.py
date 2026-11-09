# tests/test_verification.py
"""Numerical verification of the bundled reference models.

A model's output may only be called *reference* data if it agrees with an
analytic solution, converges at its theoretical order, or preserves a known
invariant. These tests check that, and also pin down the documented error of
the deliberately simple legacy configuration (forward Euler, dt = 1).
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from experiments.example_experiment.logic import ExampleExperiment
from experiments.linear_function.logic import LinearFunctionExperiment
from experiments.predator_prey.logic import PredatorPreyExperiment

LV = dict(initial_prey=10.0, initial_predators=5.0, prey_growth_rate=1.1, prey_death_rate=0.4,
          predator_growth_rate=0.1, predator_death_rate=0.4)


def simulate(cls, config):
    model = cls(config)
    state = model.initialize(config)
    for step in range(config.get("n_steps", 1)):
        state = model.run_step(state, step)
    return model.get_results()


def lv(t_end, dt, **kw):
    cfg = dict(LV, n_steps=int(round(t_end / dt)), dt=dt, clamp_negative=False, **kw)
    return simulate(PredatorPreyExperiment, cfg)


def lv_reference(t_end, n_points, **kw):
    """High-accuracy reference from scipy (DOP853, rtol 1e-12)."""
    params = dict(LV, **kw)
    a, b = params["prey_growth_rate"], params["prey_death_rate"]
    c, d = params["predator_growth_rate"], params["predator_death_rate"]
    h = params.get("handling_time", 0.0)
    K = params.get("prey_carrying_capacity")

    def rhs(_t, z):
        x, y = z
        sat = 1.0 + b * h * x
        growth = a * x if K is None else a * x * (1 - x / K)
        return [growth - b * x * y / sat, c * x * y / sat - d * y]

    t = np.linspace(0, t_end, n_points)
    sol = solve_ivp(rhs, (0, t_end), [params["initial_prey"], params["initial_predators"]],
                    t_eval=t, method="DOP853", rtol=1e-12, atol=1e-12)
    return sol.y


def lv_invariant(x, y):
    """V = c x - d ln x + b y - a ln y is conserved by the classical LV system."""
    a, b = LV["prey_growth_rate"], LV["prey_death_rate"]
    c, d = LV["predator_growth_rate"], LV["predator_death_rate"]
    return c * x - d * np.log(x) + b * y - a * np.log(y)


# ------------------------------------------------------------------ closed-form models
def test_linear_function_is_exact():
    cfg = dict(n_points=101, m=-0.75, c=3.0, x_min=-2.0, x_max=8.0)
    r = simulate(LinearFunctionExperiment, cfg)
    np.testing.assert_allclose(r["y"]["data"], -0.75 * r["x"]["data"] + 3.0, rtol=0, atol=1e-12)


def test_sine_is_exact():
    r = simulate(ExampleExperiment, dict(n_steps=50, amplitude=2.5))
    np.testing.assert_allclose(r["value"]["data"], 2.5 * np.sin(r["time"]["data"]), atol=1e-12)


# ------------------------------------------------------------------ Lotka-Volterra
def test_rk4_matches_high_accuracy_reference():
    r = lv(20.0, 0.01, integrator="rk4")
    ref = lv_reference(20.0, len(r["time"]["data"]))
    np.testing.assert_allclose(r["prey_population"]["data"], ref[0], rtol=1e-7)
    np.testing.assert_allclose(r["predator_population"]["data"], ref[1], rtol=1e-7)


@pytest.mark.parametrize("integrator, order", [("euler", 1), ("rk4", 4)])
def test_convergence_order(integrator, order):
    """Halving dt divides the max-norm global error by ~2**order. (For RK4,
    dt below ~0.005 reaches the reference solver's error floor of ~1e-11.)"""
    t_end, dts = 5.0, [0.04, 0.02, 0.01]
    errors = []
    for dt in dts:
        r = lv(t_end, dt, integrator=integrator)
        ref = lv_reference(t_end, len(r["time"]["data"]))
        errors.append(np.max(np.abs(r["prey_population"]["data"] - ref[0])))
    observed = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert observed == pytest.approx([order, order], abs=0.15)


def test_rk4_conserves_lotka_volterra_invariant():
    r = lv(50.0, 0.005, integrator="rk4")
    v = lv_invariant(r["prey_population"]["data"], r["predator_population"]["data"])
    assert np.max(np.abs(v - v[0])) / abs(v[0]) < 1e-8


def test_legacy_euler_dt1_drifts_as_documented():
    """The default (0.1.0-compatible) configuration is not reference quality:
    forward Euler with dt = 1 visibly violates the conserved quantity. The test
    pins this so the behaviour stays intentional and documented."""
    cfg = dict(LV, n_steps=30, prey_growth_rate=0.25, prey_death_rate=0.01,
               predator_growth_rate=0.005, predator_death_rate=0.2,
               initial_prey=40.0, initial_predators=9.0)
    r = simulate(PredatorPreyExperiment, cfg)
    x, y = r["prey_population"]["data"], r["predator_population"]["data"]
    a, b, c, d = 0.25, 0.01, 0.005, 0.2
    v = c * x - d * np.log(x) + b * y - a * np.log(y)
    assert np.max(np.abs(v - v[0])) / abs(v[0]) > 1e-2


def test_defaults_reproduce_0_1_0_update_rule():
    cfg = dict(LV, n_steps=3)
    r = simulate(PredatorPreyExperiment, cfg)
    x, y = cfg["initial_prey"], cfg["initial_predators"]
    for k in range(1, 4):
        dx = LV["prey_growth_rate"] * x - LV["prey_death_rate"] * x * y
        dy = LV["predator_growth_rate"] * x * y - LV["predator_death_rate"] * y
        x, y = max(0, x + dx), max(0, y + dy)
        assert r["prey_population"]["data"][k] == pytest.approx(x)
        assert r["predator_population"]["data"][k] == pytest.approx(y)
    np.testing.assert_array_equal(r["time"]["data"], [0, 1, 2, 3])


# ------------------------------------------------------------------ counterfactual forms
def test_holling2_with_zero_handling_time_equals_linear():
    a = lv(5.0, 0.01, integrator="rk4")
    b = lv(5.0, 0.01, integrator="rk4", functional_response="holling2", handling_time=0.0)
    np.testing.assert_array_equal(a["prey_population"]["data"], b["prey_population"]["data"])


def test_holling2_matches_reference():
    kw = dict(functional_response="holling2", handling_time=0.5)
    r = lv(10.0, 0.01, integrator="rk4", **kw)
    ref = lv_reference(10.0, len(r["time"]["data"]), **kw)
    np.testing.assert_allclose(r["prey_population"]["data"], ref[0], rtol=1e-7)


def test_logistic_prey_without_predators_matches_closed_form():
    K, a, x0, t_end = 50.0, 0.8, 2.0, 15.0
    cfg = dict(LV, initial_prey=x0, initial_predators=0.0, prey_growth_rate=a, prey_carrying_capacity=K)
    r = simulate(PredatorPreyExperiment, dict(cfg, n_steps=1500, dt=0.01, integrator="rk4", clamp_negative=False))
    t = r["time"]["data"]
    exact = K / (1 + (K - x0) / x0 * np.exp(-a * t))
    np.testing.assert_allclose(r["prey_population"]["data"], exact, rtol=1e-8)
    assert t[-1] == pytest.approx(t_end)


@pytest.mark.parametrize("bad", [dict(dt=0), dict(integrator="leapfrog"), dict(functional_response="holling3"),
                                 dict(handling_time=0.3), dict(prey_carrying_capacity=-1)])
def test_invalid_model_options(bad):
    with pytest.raises(ValueError):
        PredatorPreyExperiment(dict(LV, n_steps=1, **bad))


def test_law_shift_to_holling2_through_engine(tmp_path):
    from simulator.engine import SimulatorEngine
    cfg = dict(LV, experiment_type="experiments.predator_prey.logic.PredatorPreyExperiment",
               n_steps=500, dt=0.01, integrator="rk4", generate_plots=False, seed=0,
               law_shift={"changes": {"functional_response": "holling2", "handling_time": 0.5},
                          "description": "saturating predation"})
    run = SimulatorEngine(output_dir=str(tmp_path), verbose=False).run_config(cfg)
    ref = lv_reference(5.0, 501, functional_response="holling2", handling_time=0.5)
    np.testing.assert_allclose(run.results["prey_population"]["data"], ref[0], rtol=1e-7)
    assert run.record.ground_truth["parameters"]["functional_response"] == "holling2"
    assert run.record.ground_truth["law_shift"]["changes"]["functional_response"]["original"] is None
