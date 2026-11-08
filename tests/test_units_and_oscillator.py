# tests/test_units_and_oscillator.py
import numpy as np
import pytest

from experiments.oscillator.logic import OscillatorExperiment
from simulator.engine import SimulatorEngine, simulate

pint = pytest.importorskip("pint")
from simulator.units import check_units, convert_parameters  # noqa: E402

OSC = {"experiment_type": "experiments.oscillator.logic.OscillatorExperiment",
       "n_steps": 2000, "dt": 0.005, "mass": 1.0, "stiffness": 40.0, "damping": 0.0,
       "x0": 0.1, "v0": 0.0, "generate_plots": False, "seed": 1}


def model(**kw):
    return OscillatorExperiment(dict(OSC, **kw)).get_results()


# ------------------------------------------------------------------ units
@pytest.mark.parametrize("text, ok", [("m/s", True), ("individuals", True), ("time_unit", True),
                                      ("arbitrary_unit", True), ("dimensionless", True), (None, True),
                                      ("units", False), ("au", False), ("a.u.", False), ("furlongs_per_x", False)])
def test_check_units(text, ok):
    assert check_units(text)["ok"] is ok


def test_parameter_conversion_and_errors():
    cfg, conv = convert_parameters({"stiffness": "2 kN/m", "x0": {"value": 5, "units": "cm"}, "mass": 2.0,
                                    "integrator": "auto"}, OscillatorExperiment.parameter_units)
    assert cfg["stiffness"] == pytest.approx(2000.0) and cfg["x0"] == pytest.approx(0.05) and cfg["mass"] == 2.0
    assert conv["stiffness"] == {"given": "2 kN/m", "value": pytest.approx(2000.0), "units": "N/m"}
    with pytest.raises(ValueError, match="cannot convert"):
        convert_parameters({"stiffness": "2 kg"}, OscillatorExperiment.parameter_units)


def test_engine_converts_units_and_records_them(tmp_path):
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(
        dict(OSC, stiffness="0.04 kN/m", x0="10 cm", n_steps=10))
    gt = run.record.ground_truth
    assert gt["parameters"]["stiffness"] == pytest.approx(40.0) and gt["parameters"]["x0"] == pytest.approx(0.1)
    assert gt["parameter_units"]["stiffness"] == "N/m"
    assert run.record.config["stiffness"] == "0.04 kN/m"  # the recipe keeps what the user wrote
    np.testing.assert_allclose(run.results["position"]["data"],
                               model(n_steps=10)["position"]["data"])
    assert run.record.provenance["units"]["checked"] is True
    from simulator.units import parse_units
    energy_units = run.record.provenance["units"]["outputs"]["energy"]["canonical"]
    assert parse_units(energy_units).dimensionality == parse_units("kg*m**2/s**2").dimensionality


def test_unit_trap_is_reported_and_can_be_an_error(tmp_path):
    from simulator.base import ExperimentLogic
    from simulator.utils import DataDescriptor, DataType

    class Trap(ExperimentLogic):
        def __init__(self, config): pass
        def initialize(self, config): return {}
        def run_step(self, state, step): return state
        def get_results(self):
            return {"x": {"data": np.zeros(2), "descriptor": DataDescriptor("x", DataType.NDARRAY, units="units")}}
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config({"generate_plots": False}, experiment_logic_class=Trap)
    assert any("micronit" in m for m in run.record.log_messages)
    with pytest.raises(ValueError, match="Unit check failed"):
        SimulatorEngine(str(tmp_path), verbose=False).run_config({"generate_plots": False, "unit_check": "error"},
                                                                 experiment_logic_class=Trap)


def test_bundled_models_have_clean_units(tmp_path):
    configs = [
        dict(OSC, n_steps=5),
        {"experiment_type": "experiments.predator_prey.logic.PredatorPreyExperiment", "n_steps": 5,
         "initial_prey": 10, "initial_predators": 5, "prey_growth_rate": 1.1, "prey_death_rate": 0.4,
         "predator_growth_rate": 0.1, "predator_death_rate": 0.4},
        {"experiment_type": "experiments.example_random_walk.logic.RandomWalkExperiment", "n_steps": 5, "step_size": 1.0},
        {"experiment_type": "experiments.example_experiment.logic.ExampleExperiment", "n_steps": 5, "amplitude": 1.0},
        {"experiment_type": "experiments.linear_function.logic.LinearFunctionExperiment", "n_points": 3, "m": 1.0,
         "c": 0.0, "x_min": 0.0, "x_max": 1.0, "x_units": "s", "y_units": "m"},
    ]
    for cfg in configs:
        run = SimulatorEngine(str(tmp_path), verbose=False).run_config(dict(cfg, generate_plots=False, unit_check="error"))
        assert run.record.provenance["units"]["checked"]


# ------------------------------------------------------------------ oscillator verification
@pytest.mark.parametrize("damping", [0.0, 2.0, 12.649110640673518, 30.0])  # undamped, under, critical, over
def test_closed_form_matches_dop853(damping):
    exact = model(damping=damping, integrator="exact")
    numeric = model(damping=damping, integrator="dop853")
    np.testing.assert_allclose(exact["position"]["data"], numeric["position"]["data"], atol=1e-9)
    np.testing.assert_allclose(exact["velocity"]["data"], numeric["velocity"]["data"], atol=1e-8)


def test_closed_form_satisfies_the_ode():
    r = model(damping=2.0, integrator="exact", n_steps=4000, dt=0.0005)
    t, x, v = r["time"]["data"], r["position"]["data"], r["velocity"]["data"]
    a = np.gradient(v, t)
    residual = 1.0 * a[5:-5] + 2.0 * v[5:-5] + 40.0 * x[5:-5]
    assert np.max(np.abs(residual)) < 1e-4
    np.testing.assert_allclose(np.gradient(x, t)[5:-5], v[5:-5], atol=1e-5)


def test_energy_conserved_without_damping_and_decays_with_it():
    for nonlinearity in (0.0, 500.0):
        e = model(nonlinearity=nonlinearity)["energy"]["data"]
        assert np.max(np.abs(e - e[0])) / e[0] < 1e-8
    e = model(damping=1.0)["energy"]["data"]
    assert np.all(np.diff(e) <= 1e-12)


def test_euler_audit_model_gains_energy_as_documented(tmp_path):
    e = model(integrator="euler")["energy"]["data"]
    assert e[-1] > 1.5 * e[0]
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(dict(OSC, integrator="euler", n_steps=50))
    assert run.record.provenance["fidelity"]["level"] == "audit"


def test_duffing_law_shift_changes_the_period(tmp_path):
    base = simulate(dict(OSC))
    shifted = simulate(dict(OSC, law_shift={"changes": {"nonlinearity": "4000 N/m^3"},
                                            "description": "hardening spring"}))
    assert shifted.ground_truth["parameters"]["nonlinearity"] == pytest.approx(4000.0)

    def first_upcrossing_period(r):
        x, t = r.results["position"]["data"], r.results["time"]["data"]
        crossings = t[1:][(x[:-1] < 0) & (x[1:] >= 0)]
        return np.diff(crossings).mean()
    assert first_upcrossing_period(shifted) < first_upcrossing_period(base)  # hardening -> shorter period
    assert first_upcrossing_period(base) == pytest.approx(2 * np.pi / np.sqrt(40.0), rel=1e-2)


@pytest.mark.parametrize("bad", [dict(mass=0), dict(damping=-1), dict(integrator="rk2"),
                                 dict(integrator="exact", nonlinearity=1.0)])
def test_invalid_oscillator_options(bad):
    with pytest.raises(ValueError):
        OscillatorExperiment(dict(OSC, **bad))
