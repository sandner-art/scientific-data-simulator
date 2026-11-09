# tests/test_observation_and_ground_truth.py
import json
import os

import numpy as np
import pytest

from simulator.engine import SimulatorEngine
from simulator.observation import apply_observation, register_operator, VALUE_OPERATORS
from simulator.persistence import load_experiment_record
from simulator.utils import DataDescriptor, DataType, Role

LINEAR = {"experiment_type": "experiments.linear_function.logic.LinearFunctionExperiment",
          "n_points": 2000, "m": 2.0, "c": 5.0, "x_min": 0.0, "x_max": 10.0,
          "generate_plots": False, "seed": 2024}


def _run(tmp_path, **overrides):
    return SimulatorEngine(output_dir=str(tmp_path), verbose=False).run_config(dict(LINEAR, **overrides))


def _latent(n=1000, value=10.0):
    data = np.full(n, value)
    return {
        "t": {"data": np.arange(n, dtype=float), "descriptor": DataDescriptor("t", DataType.NDARRAY, role="coordinate")},
        "x": {"data": data, "descriptor": DataDescriptor("x", DataType.NDARRAY, x_axis="t", role="latent",
                                                         group="time_series", plot_type="line")},
    }


# ------------------------------------------------------------------ operators (statistics)
def test_gaussian_sigma_is_realised(tmp_path):
    run = _run(tmp_path, observation={"operators": [{"type": "gaussian", "sigma": 0.5}]})
    residual = run.results["y_observed"]["data"] - run.results["y"]["data"]
    assert residual.std() == pytest.approx(0.5, rel=0.05)
    assert abs(residual.mean()) < 5 * 0.5 / np.sqrt(len(residual))


def test_gaussian_snr_db_sets_sigma_from_signal_rms(tmp_path):
    run = _run(tmp_path, observation={"operators": [{"type": "gaussian", "snr_db": 20}]})
    y = run.results["y"]["data"]
    realized = run.record.ground_truth["observation"]["targets"]["y"]["operators"][0]
    assert realized["sigma"] == pytest.approx(np.sqrt(np.mean(y ** 2)) / 10.0)
    residual = run.results["y_observed"]["data"] - y
    assert residual.std() == pytest.approx(realized["sigma"], rel=0.05)


def test_poisson_mean_and_variance():
    added, _ = apply_observation(_latent(20000, 7.0), {"operators": [{"type": "poisson"}]}, np.random.default_rng(1))
    obs = added["x_observed"]["data"]
    assert obs.mean() == pytest.approx(7.0, rel=0.02) and obs.var() == pytest.approx(7.0, rel=0.05)
    assert np.all(obs == np.round(obs))


def test_multiplicative_noise_has_unit_mean():
    added, _ = apply_observation(_latent(20000, 3.0), {"operators": [{"type": "multiplicative", "sigma": 0.2}]},
                                 np.random.default_rng(2))
    ratio = added["x_observed"]["data"] / 3.0
    assert ratio.mean() == pytest.approx(1.0, abs=0.01) and np.all(ratio > 0)


def test_missing_quantize_outliers_clip():
    ops = [{"type": "gaussian", "sigma": 1.0}, {"type": "quantize", "step": 0.5},
           {"type": "outliers", "rate": 0.05, "scale": 10.0}, {"type": "clip", "low": 0.0},
           {"type": "missing", "rate": 0.1}]
    added, realized = apply_observation(_latent(20000), {"operators": ops}, np.random.default_rng(3))
    obs = added["x_observed"]["data"]
    assert np.isnan(obs).mean() == pytest.approx(0.1, abs=0.01)
    finite = obs[np.isfinite(obs)]
    assert finite.min() >= 0.0
    steps = {s["type"]: s for s in realized["targets"]["x"]["operators"]}
    assert len(steps["outliers"]["indices"]) == pytest.approx(1000, rel=0.15)
    assert steps["missing"]["n_missing"] == int(np.isnan(obs).sum())


def test_sampling_every_adds_coordinates_and_index():
    added, realized = apply_observation(_latent(100), {"sampling": {"every": 10, "offset": 5}},
                                        np.random.default_rng(0))
    np.testing.assert_array_equal(added["observation_index"]["data"], np.arange(5, 100, 10))
    np.testing.assert_array_equal(added["t_observed"]["data"], np.arange(5, 100, 10))
    assert added["x_observed"]["descriptor"].x_axis == "t_observed"
    assert added["t_observed"]["descriptor"].role is Role.COORDINATE
    assert realized["sampling"] == {"method": "every", "every": 10, "offset": 5}


@pytest.mark.parametrize("sampling", [{"n": 17, "method": "random"}, {"n": 17, "method": "uniform"}])
def test_sampling_n(sampling):
    added, _ = apply_observation(_latent(100), {"sampling": sampling}, np.random.default_rng(0))
    idx = added["observation_index"]["data"]
    assert len(idx) == 17 and np.all(np.diff(idx) > 0)


@pytest.mark.parametrize("obs", [
    {"operators": [{"type": "nonsense"}]},
    {"operators": [{"sigma": 1}]},
    {"sampling": {"rate": 2}},
    {"bogus": 1},
    {"targets": ["missing_output"]},
])
def test_observation_config_errors(obs):
    with pytest.raises((ValueError, KeyError)):
        apply_observation(_latent(10), obs, np.random.default_rng(0))


def test_custom_operator_registration():
    name = "test_offset_operator"
    if name not in VALUE_OPERATORS:
        register_operator(name, lambda x, rng, p: (x + p["by"], {"by": p["by"]}))
    added, _ = apply_observation(_latent(5, 1.0), {"operators": [{"type": name, "by": 2.0}]}, np.random.default_rng(0))
    np.testing.assert_array_equal(added["x_observed"]["data"], np.full(5, 3.0))


# ------------------------------------------------------------------ engine integration
def test_noise_does_not_change_latent_and_is_seeded(tmp_path):
    clean = _run(tmp_path, observation={"operators": [{"type": "gaussian", "sigma": 0.1}]})
    noisy = _run(tmp_path, observation={"operators": [{"type": "gaussian", "sigma": 5.0}]})
    again = _run(tmp_path, observation={"operators": [{"type": "gaussian", "sigma": 5.0}]})
    np.testing.assert_array_equal(clean.results["y"]["data"], noisy.results["y"]["data"])
    np.testing.assert_array_equal(noisy.results["y_observed"]["data"], again.results["y_observed"]["data"])


def test_ground_truth_export(tmp_path):
    run = _run(tmp_path, observation={"sampling": {"every": 100}, "operators": [{"type": "gaussian", "sigma": 1.0}]})
    gt_path = os.path.join(run.experiment_dir, "ground_truth.json")
    with open(gt_path) as f:
        gt = json.load(f)
    assert gt["parameters"] == {"n_points": 2000, "m": 2.0, "c": 5.0, "x_min": 0.0, "x_max": 10.0}
    assert gt["seed"] == 2024 and gt["synthetic"] is True
    assert gt["ground_truth_fields"] == ["y"]
    assert gt["observation_fields"] == ["y_observed"]
    assert gt["fields"]["x_observed"]["role"] == "coordinate"
    assert gt["observation"]["targets"]["y"]["operators"][0]["sigma"] == 1.0
    stored = load_experiment_record(str(tmp_path), run.experiment_id)
    assert stored.output_data["y_observed"]["descriptor"].role is Role.OBSERVATION
    assert stored.ground_truth == gt


def test_law_shift_changes_model_and_is_recorded(tmp_path):
    run = _run(tmp_path, law_shift={"changes": {"m": -1.5}, "description": "counterfactual slope"})
    x, y = run.results["x"]["data"], run.results["y"]["data"]
    np.testing.assert_allclose(y, -1.5 * x + 5.0)
    gt = run.record.ground_truth
    assert gt["parameters"]["m"] == -1.5
    assert gt["law_shift"] == {"description": "counterfactual slope",
                               "changes": {"m": {"original": 2.0, "shifted": -1.5}}}
    # the stored recipe keeps the original value plus the shift
    assert run.record.config["m"] == 2.0 and run.record.config["law_shift"]["changes"]["m"] == -1.5


def test_law_shift_flat_form_and_new_keys(tmp_path):
    run = _run(tmp_path, law_shift={"c": 0.0, "new_term": "quadratic"})
    assert run.record.ground_truth["law_shift"]["changes"]["new_term"] == {"original": None, "shifted": "quadratic"}


def test_observation_targets_default_to_latent_outputs(tmp_path):
    run = _run(tmp_path, observation={"operators": [{"type": "gaussian", "sigma": 1.0}]})
    assert "y_observed" in run.results and "x_observed" not in run.results


def test_real_data_experiment_is_flagged_not_synthetic(tmp_path):
    import pandas as pd
    csv = tmp_path / "d.csv"
    pd.DataFrame({"time": [0, 1], "value": [1.0, 2.0]}).to_csv(csv, index=False)
    run = SimulatorEngine(output_dir=str(tmp_path / "o"), verbose=False).run_config(
        {"experiment_type": "experiments.data_analysis.logic.DataAnalysisExperiment",
         "input_data_path": str(csv), "generate_plots": False})
    assert run.record.ground_truth["synthetic"] is False
    assert "input_data_path" not in run.record.ground_truth["parameters"]


def test_observed_series_are_plotted(tmp_path):
    run = _run(tmp_path, generate_plots=True, n_points=50,
               observation={"sampling": {"every": 5}, "operators": [{"type": "gaussian", "sigma": 1.0}]})
    assert os.path.exists(os.path.join(run.experiment_dir, "time_series_matplotlib.png"))
