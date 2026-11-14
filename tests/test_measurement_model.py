# tests/test_measurement_model.py - correlated noise, drift, calibration, censoring, informative
# missingness, jitter, and the field (image) operators
import numpy as np
import pytest

from scidatasim.engine import simulate
from scidatasim.observation import apply_observation, validate_observation_config
from scidatasim.utils import DataDescriptor, DataType, Role


def _series(n=2000, x=None, y=None):
    x = np.linspace(0.0, 10.0, n) if x is None else x
    y = np.sin(x) + 2.0 if y is None else y
    return {"x": {"data": x, "descriptor": DataDescriptor("x", DataType.NDARRAY, role=Role.COORDINATE)},
            "y": {"data": y, "descriptor": DataDescriptor("y", DataType.NDARRAY, role=Role.LATENT, x_axis="x",
                                                          plot_type="line")}}


def _observe(ops, results=None, seed=0, **obs):
    results = _series() if results is None else results
    added, realized = apply_observation(results, dict(obs, operators=ops), np.random.default_rng(seed))
    name = next(iter(realized["targets"]))
    return added[f"{name}_observed"]["data"], realized["targets"][name]["operators"], results[name]["data"]


def test_ar1_noise_is_correlated_with_the_stated_level():
    obs, info, y = _observe([{"type": "ar1", "sigma": 0.1, "rho": 0.8}])
    err = obs - y
    assert abs(np.std(err) - 0.1) < 0.02
    assert abs(np.corrcoef(err[:-1], err[1:])[0, 1] - 0.8) < 0.05
    iid, _, _ = _observe([{"type": "gaussian", "sigma": 0.1}])
    assert abs(np.corrcoef((iid - y)[:-1], (iid - y)[1:])[0, 1]) < 0.1


def test_ar1_correlation_length_follows_the_coordinate():
    x = np.sort(np.random.default_rng(1).uniform(0, 100, 3000))
    obs, info, y = _observe([{"type": "ar1", "sigma": 1.0, "length": 2.0}], _series(x=x, y=np.zeros_like(x)))
    assert info[0]["length"] == 2.0 and abs(np.std(obs) - 1.0) < 0.15
    with pytest.raises(ValueError):
        _observe([{"type": "ar1", "sigma": 1.0, "rho": 0.5, "length": 1.0}])
    with pytest.raises(ValueError):
        _observe([{"type": "ar1", "sigma": 1.0, "rho": 1.0}])


def test_drift_is_linear_in_the_coordinate_and_recorded():
    obs, info, y = _observe([{"type": "drift", "rate": 0.5}])
    assert np.allclose(obs - y, 0.5 * np.linspace(0.0, 10.0, 2000))
    assert info[0]["per"] == "coordinate unit" and info[0]["final_offset"] == pytest.approx(5.0)
    walk, info, y = _observe([{"type": "drift", "random_walk": 0.01}])
    assert info[0]["random_walk"] == 0.01 and np.std(np.diff(walk - y)) == pytest.approx(0.01, rel=0.1)
    with pytest.raises(ValueError):
        _observe([{"type": "drift"}])


def test_calibration_error_is_one_gain_and_offset_per_run():
    obs, info, y = _observe([{"type": "calibration", "gain_sd": 0.05, "offset_sd": 0.2}])
    gain, offset = info[0]["gain"], info[0]["offset"]
    assert np.allclose(obs, gain * y + offset) and gain != 1.0
    fixed, info, y = _observe([{"type": "calibration", "gain": 1.1, "offset": -0.3}])
    assert np.allclose(fixed, 1.1 * y - 0.3)


@pytest.mark.parametrize("replace, below_value", [("nan", np.nan), ("limit", 1.5), ("half", 0.75)])
def test_censor_applies_the_detection_limit(replace, below_value):
    obs, info, y = _observe([{"type": "censor", "low": 1.5, "replace": replace}])
    below = y < 1.5
    assert info[0]["n_below"] == below.sum() > 0
    if replace == "nan":
        assert np.isnan(obs[below]).all()
    else:
        assert np.allclose(obs[below], below_value)
    assert np.allclose(obs[~below], y[~below])


def test_missing_not_at_random_removes_high_values_more_often():
    obs, info, y = _observe([{"type": "missing", "rate": 0.2, "mechanism": "mnar", "strength": 2.0}])
    lost = np.isnan(obs)
    assert abs(lost.mean() - 0.2) < 0.03
    assert y[lost].mean() > y[~lost].mean() + 0.3
    low, _, y = _observe([{"type": "missing", "rate": 0.2, "mechanism": "mnar", "direction": "low"}])
    assert y[np.isnan(low)].mean() < y[~np.isnan(low)].mean()
    mar, info, y = _observe([{"type": "missing", "rate": 0.2, "mechanism": "mar"}])
    x = np.linspace(0.0, 10.0, 2000)
    assert x[np.isnan(mar)].mean() > x[~np.isnan(mar)].mean() + 1.0  # late points lost more often
    mcar, info, _ = _observe([{"type": "missing", "rate": 0.2}])
    assert info[0]["mechanism"] == "mcar" and abs(np.isnan(mcar).mean() - 0.2) < 0.03


def test_jitter_measures_at_shifted_positions():
    obs, info, y = _observe([{"type": "jitter", "sigma": 0.05}])
    assert 0.03 < info[0]["rms_shift"] < 0.07
    err = obs - y  # about slope * shift: |cos(x)| * 0.05
    assert 0 < np.std(err) < 0.06 and np.max(np.abs(err)) < 0.25
    with pytest.raises(ValueError):  # it starts from the exact values, so it must come first
        validate_observation_config({"operators": [{"type": "gaussian", "sigma": 1}, {"type": "jitter", "sigma": 1}]})


def test_series_operators_reject_fields_and_need_coordinates():
    field = {"f": {"data": np.ones((8, 8)), "descriptor": DataDescriptor("f", DataType.NDARRAY, role=Role.LATENT)}}
    for op in ({"type": "ar1", "sigma": 1, "rho": 0.5}, {"type": "drift", "rate": 1.0}):
        with pytest.raises(ValueError):
            _observe([op], field, targets=["f"])
    no_x = {"y": {"data": np.arange(10.0), "descriptor": DataDescriptor("y", DataType.NDARRAY, role=Role.LATENT)}}
    with pytest.raises(ValueError):
        _observe([{"type": "jitter", "sigma": 0.1}], no_x)
    with pytest.raises(ValueError):
        _observe([{"type": "gaussian", "sigma": 1}], field, targets=["f"], sampling={"every": 2})


# ------------------------------------------------------------------ fields
def _field(shape=(4, 32, 32)):
    rng = np.random.default_rng(3)
    data = 100.0 + 50.0 * rng.random(shape)
    return {"u": {"data": data, "descriptor": DataDescriptor("u", DataType.NDARRAY, role=Role.LATENT, group="field")}}


def test_psf_blur_smooths_each_frame_and_keeps_the_mean():
    obs, info, u = _observe([{"type": "psf_blur", "sigma": 2.0, "boundary": "wrap"}], _field(), targets=["u"])
    assert obs.shape == u.shape and np.std(obs) < 0.5 * np.std(u)
    assert np.allclose(obs.mean(axis=(1, 2)), u.mean(axis=(1, 2)))  # frames stay separate


def test_detector_noise_saturation_and_bits():
    obs, info, u = _observe([{"type": "detector", "scale": 100.0, "read_noise": 5.0, "gain": 2.0}],
                            _field(), targets=["u"])
    assert abs(np.mean(obs - u)) < 0.05
    assert np.std(obs - u) == pytest.approx(np.sqrt(np.mean(u) * 100 + 25) / 100, rel=0.1)
    sat, info, u = _observe([{"type": "detector", "scale": 100.0, "full_well": 12000.0}], _field(), targets=["u"])
    assert info[0]["n_saturated"] > 0 and sat.max() <= 120.0
    adc, info, _ = _observe([{"type": "detector", "scale": 1.0, "bits": 8}], _field(), targets=["u"])
    assert len(np.unique(adc)) <= 256 and np.allclose(adc, np.round(adc))


def test_bin_vignetting_and_shift():
    binned, info, u = _observe([{"type": "bin", "factor": 3}], _field(), targets=["u"])
    assert binned.shape == (4, 10, 10) and info[0]["cropped"] == [2, 2]
    assert binned[0, 0, 0] == pytest.approx(u[0, :3, :3].mean())
    ones = {"u": {"data": np.ones((16, 16)), "descriptor": DataDescriptor("u", DataType.NDARRAY, role=Role.LATENT)}}
    vig, info, _ = _observe([{"type": "vignetting", "strength": 0.4}], ones, targets=["u"])
    assert vig[0, 0] == pytest.approx(0.6) and vig[8, 8] > 0.99
    spot = np.zeros((32, 32))
    spot[16, 16] = 1.0
    moved, info, _ = _observe([{"type": "shift", "dx": 2.0, "dy": -1.0}],
                              {"u": {"data": spot, "descriptor": DataDescriptor("u", DataType.NDARRAY, role=Role.LATENT)}},
                              targets=["u"])
    assert np.unravel_index(np.argmax(moved), moved.shape) == (15, 18)
    with pytest.raises(ValueError):
        _observe([{"type": "bin", "factor": 2}])  # a 1-D series is not an image


def test_a_2d_heat_field_is_observed_like_a_camera_and_reproducibly():
    cfg = {"experiment_type": "scidatasim.experiments.heat.logic.HeatExperiment", "dims": 2, "nx": 32,
           "n_steps": 5, "dt": 0.001, "alpha": 0.01, "length": 1.0, "seed": 7, "generate_plots": False,
           "observation": {"targets": ["u"], "operators": [
               {"type": "psf_blur", "sigma": 1.0, "boundary": "wrap"},
               {"type": "detector", "scale": 1000.0, "read_noise": 3.0, "bits": 16},
               {"type": "missing", "rate": 0.01}]}}
    a, b = simulate(cfg), simulate(cfg)
    obs = a.results["u_observed"]
    assert obs["data"].shape == a.results["u"]["data"].shape and obs["descriptor"].role is Role.OBSERVATION
    assert np.array_equal(obs["data"], b.results["u_observed"]["data"], equal_nan=True)
    assert 0 < np.isnan(obs["data"]).mean() < 0.03
    assert "u_probe_observed" not in a.results  # only the named target


def test_observed_fields_are_stored_with_the_run(tmp_path):
    from scidatasim.engine import SimulatorEngine
    cfg = {"experiment_type": "scidatasim.experiments.heat.logic.HeatExperiment", "dims": 2, "nx": 16,
           "n_steps": 3, "dt": 0.001, "alpha": 0.01, "length": 1.0, "seed": 1, "generate_plots": False,
           "observation": {"targets": ["u"], "operators": [{"type": "bin", "factor": 2},
                                                           {"type": "gaussian", "snr_db": 30}]}}
    engine = SimulatorEngine(output_dir=str(tmp_path), verbose=False)
    record = engine.run_config(cfg)
    loaded = engine.load_experiment_record(record.experiment_id)
    assert loaded.output_data["u_observed"]["data"].shape == (4, 8, 8)
    ops = loaded.ground_truth["observation"]["targets"]["u"]["operators"]
    assert [o["type"] for o in ops] == ["bin", "gaussian"]
