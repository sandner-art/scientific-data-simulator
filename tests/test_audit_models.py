# tests/test_audit_models.py
"""Pinning tests for the deliberately weak (``audit``) models: their defects must stay
present and measurable, and their reference counterparts must stay correct."""
import numpy as np
import pytest

from experiments.example_random_walk.logic import RANDU_MODULUS, randu
from experiments.pendulum.logic import PendulumExperiment, exact_period
from simulator.engine import simulate

PEND = {"experiment_type": "experiments.pendulum.logic.PendulumExperiment", "n_steps": 2000, "dt": 0.005,
        "generate_plots": False, "seed": 0}
WALK = {"experiment_type": "experiments.example_random_walk.logic.RandomWalkExperiment", "n_steps": 3000,
        "step_size": 1.0, "generate_plots": False, "seed": 5}


def _measured_period(t, theta):
    """Period from successive downward zero crossings (linear interpolation)."""
    idx = np.where((theta[:-1] > 0) & (theta[1:] <= 0))[0]
    crossings = t[idx] + theta[idx] / (theta[idx] - theta[idx + 1]) * (t[idx + 1] - t[idx])
    return np.mean(np.diff(crossings))


# ------------------------------------------------------------------ pendulum
@pytest.mark.parametrize("deg", [10, 60, 120, 170])
def test_reference_pendulum_period_matches_elliptic_integral(deg):
    sim = simulate(dict(PEND, theta0_deg=deg, n_steps=4000))
    t, th = sim.results["time"]["data"], sim.results["angle"]["data"]
    T = exact_period(1.0, 9.81, np.deg2rad(deg))
    assert _measured_period(t, th) == pytest.approx(T, rel=1e-6)
    assert sim.results["period"]["data"] == pytest.approx(T, rel=1e-12)
    e = sim.results["energy"]["data"]
    assert np.ptp(e) / e[0] < 1e-8
    assert sim.ground_truth["fidelity"]["level"] == "reference"


def test_small_angle_is_audit_and_its_error_grows_with_amplitude():
    errors = []
    for deg in (5, 30, 60, 90):
        sim = simulate(dict(PEND, theta0_deg=deg, approximation="small_angle"))
        t, th = sim.results["time"]["data"], sim.results["angle"]["data"]
        assert _measured_period(t, th) == pytest.approx(2 * np.pi / np.sqrt(9.81), rel=1e-9)
        errors.append(1 - _measured_period(t, th) / exact_period(1.0, 9.81, np.deg2rad(deg)))
        assert sim.ground_truth["fidelity"]["level"] == "audit"
    assert errors == sorted(errors)
    assert errors[0] < 1e-3 and 0.06 < errors[2] < 0.08 and 0.14 < errors[3] < 0.16
    note = simulate(dict(PEND, theta0_deg=60, approximation="small_angle")).ground_truth["fidelity"]["note"]
    assert "6.8 % too short" in note


def test_small_angle_and_reference_agree_for_tiny_swings_and_with_damping():
    a = simulate(dict(PEND, theta0=1e-4, damping=0.3, omega0=1e-4))
    b = simulate(dict(PEND, theta0=1e-4, damping=0.3, omega0=1e-4, approximation="small_angle"))
    np.testing.assert_allclose(a.results["angle"]["data"], b.results["angle"]["data"], atol=1e-11)
    for c in (2 * np.sqrt(9.81), 10.0):  # critical and over-damped closed forms vs. linear ODE
        lin = PendulumExperiment(dict(PEND, theta0=0.2, damping=c, approximation="small_angle"))
        t = np.linspace(0, 3, 50)
        np.testing.assert_allclose(lin.linear_solution(t)[0], lin.numerical_solution(t)[0], atol=1e-9)


def test_pendulum_validation_and_units():
    with pytest.raises(ValueError):
        PendulumExperiment(dict(PEND, approximation="taylor"))
    with pytest.raises(ValueError):
        PendulumExperiment(dict(PEND, theta0=0.1, theta0_deg=5))
    with pytest.raises(ValueError):
        PendulumExperiment(dict(PEND, length=0))
    pytest.importorskip("pint")
    sim = simulate(dict(PEND, length="50 cm", theta0_deg=10))
    assert sim.results["period"]["data"] == pytest.approx(exact_period(0.5, 9.81, np.deg2rad(10)))


# ------------------------------------------------------------------ RANDU walk
def test_randu_triples_lie_on_planes():
    u = randu(12345, 3000)
    x = np.rint(u * RANDU_MODULUS).astype(np.int64)
    assert np.all((x[2:] - 6 * x[1:-1] + 9 * x[:-2]) % RANDU_MODULUS == 0)
    # 9x - 6y + z is an integer in a small range: only 15 distinct planes in the unit cube
    planes = np.unique(np.rint(9 * u[:-2] - 6 * u[1:-1] + u[2:]))
    assert len(planes) <= 15
    rng = np.random.default_rng(0).random(3000)
    assert len(np.unique(np.rint(9 * rng[:-2] - 6 * rng[1:-1] + rng[2:]))) > 15


def test_randu_walk_passes_naive_checks_but_is_audit():
    sim = simulate(dict(WALK, generator="randu"))
    steps = np.diff(sim.results["position"]["data"])
    assert set(np.unique(steps)) <= {-1.0, 1.0} and abs(steps.mean()) < 0.1
    assert sim.ground_truth["fidelity"]["level"] == "audit"
    assert sim.results["uniforms"]["descriptor"].role.value == "auxiliary"
    again = simulate(dict(WALK, generator="randu"))
    np.testing.assert_array_equal(sim.results["position"]["data"], again.results["position"]["data"])
    ref = simulate(WALK)
    assert ref.ground_truth["fidelity"]["level"] == "reference" and "uniforms" not in ref.results
    with pytest.raises(ValueError):
        simulate(dict(WALK, generator="mt"))


def test_pendulum_needs_steps():
    with pytest.raises(ValueError):
        PendulumExperiment(dict(PEND, n_steps=0))
