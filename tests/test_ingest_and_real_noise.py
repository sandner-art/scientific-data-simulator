"""Ingest of external files, the real_noise operator and the external experiment."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from scidatasim import ingest
from scidatasim.engine import simulate
from scidatasim.observation import apply_observation, validate_observation_config

LINEAR = dict(experiment_type="scidatasim.experiments.linear_function.logic.LinearFunctionExperiment",
              n_points=200, m=0.0, c=1.0, x_min=0.0, x_max=199.0)
EXTERNAL = "scidatasim.experiments.external.logic.ExternalSeriesExperiment"


@pytest.fixture
def light_curve(tmp_path):
    """A 2-minute-cadence 'light curve': red-ish noise around 1000, a gap and flagged points."""
    rng = np.random.default_rng(0)
    t = 1000.0 + np.arange(3000) * (2 / 1440)
    noise = np.convolve(rng.normal(0, 1, t.size + 9), np.ones(10) / np.sqrt(10), "valid")[: t.size]
    flux = 1000.0 * (1 + 3e-4 * noise)
    flux[1200:1300] = np.nan  # a data gap
    quality = np.zeros(t.size, int)
    quality[500:510] = 128
    frame = pd.DataFrame({"time": t, "flux": flux, "flux_err": 0.3, "quality": quality})
    path = tmp_path / "lc.csv"
    frame.to_csv(path, index=False)
    return str(path), frame


def test_read_table_formats_and_provenance(tmp_path, light_curve):
    path, frame = light_curve
    table = ingest.read_table(path, columns=["time", "flux"])
    assert list(table.frame.columns) == ["time", "flux"] and table.provenance["rows"] == 3000
    assert len(table.provenance["sha256"]) == 64 and table.provenance["format"] == "csv"
    with pytest.raises(KeyError):
        ingest.read_table(path, columns=["nope"])
    np.savez(tmp_path / "a.npz", time=frame["time"].to_numpy(), flux=frame["flux"].to_numpy())
    assert ingest.read_table(str(tmp_path / "a.npz")).frame.shape == (3000, 2)
    (tmp_path / "a.ecsv").write_text("# %ECSV 1.0\n# ---\ntime flux\n1.0 2.0\n2.0 3.0\n")
    assert ingest.read_table(str(tmp_path / "a.ecsv")).frame["flux"].tolist() == [2.0, 3.0]
    (tmp_path / "a.dat").write_text("1.0   2.0\n2.0   3.5\n")
    assert ingest.read_table(str(tmp_path / "a.dat"), header=None).frame.shape == (2, 2)
    if "parquet" in ingest.available_formats():
        frame.to_parquet(tmp_path / "a.parquet")
        assert ingest.read_table(str(tmp_path / "a.parquet"), columns=["flux"]).frame.shape == (3000, 1)
    if "hdf5" in ingest.available_formats():
        import h5py
        with h5py.File(tmp_path / "a.h5", "w") as f:
            f["lc/time"], f["lc/flux"] = frame["time"].to_numpy(), frame["flux"].to_numpy()
            f["image"] = np.ones((4, 5))
        assert set(ingest.read_table(str(tmp_path / "a.h5"), key="lc").frame.columns) == {"time", "flux"}
        assert ingest.read_field(str(tmp_path / "a.h5"), key="image")["data"].shape == (4, 5)
    if "fits" in ingest.available_formats():
        from astropy.table import Table
        Table.from_pandas(frame).write(tmp_path / "a.fits")
        s = ingest.read_series(str(tmp_path / "a.fits"), value="flux", time="time", quality="quality")
        assert s.meta["flagged"] == 10 and np.isnan(s.value[500])


def test_read_series_quality_and_normalize(light_curve):
    path, _ = light_curve
    s = ingest.read_series(path, value="flux", time="time", error="flux_err", quality="quality", normalize="median")
    assert s.meta["flagged"] == 10 and np.isnan(s.value[505]) and np.isnan(s.value[1250])
    assert abs(np.nanmedian(s.value) - 1.0) < 1e-12 and s.error[0] == pytest.approx(0.3 / s.meta["median"])
    assert s.cadence == pytest.approx(2 / 1440) and s.mask.sum() == 3000 - 100 - 10


def test_real_noise_by_index_keeps_noise_and_gaps(light_curve):
    path, frame = light_curve
    cfg = dict(LINEAR, seed=3, observation={"targets": ["y"], "operators": [
        {"type": "real_noise", "source": path, "column": "flux", "normalize": "median", "combine": "multiply",
         "align": "index", "offset": 1150}]})
    run = simulate(cfg)
    y = run.results["y_observed"]["data"]
    bg = frame["flux"].to_numpy()[1150:1350] / np.nanmedian(frame["flux"])
    assert np.allclose(y, bg, equal_nan=True) and np.isnan(y[50:150]).all()
    info = run.ground_truth["observation"]["targets"]["y"]["operators"][0]
    assert info["offset"] == 1150 and info["n_gaps"] == 100 and info["source"]["sha256"] == \
        ingest.sha256_file(path)


def test_real_noise_by_time_and_reproducible(light_curve):
    path, frame = light_curve
    obs = {"targets": ["y"], "operators": [{"type": "real_noise", "source": path, "time": "time",
                                            "column": "flux", "normalize": "subtract_median"}]}
    # model coordinate: days at the background's cadence
    cfg = dict(LINEAR, n_points=300, x_min=0.0, x_max=299 * 2 / 1440, m=0.0, c=0.0, observation=obs)
    a, b = simulate(dict(cfg, seed=5)), simulate(dict(cfg, seed=5))
    ya, yb = a.results["y_observed"]["data"], b.results["y_observed"]["data"]
    assert np.array_equal(ya, yb, equal_nan=True)
    info = a.ground_truth["observation"]["targets"]["y"]["operators"][0]
    assert info["align"] == "time" and info["shift"] >= 1000.0 - 1e-9
    # each point takes one real sample: values are background residuals at consecutive times
    resid = frame["flux"].to_numpy() - np.nanmedian(frame["flux"])
    finite = ya[np.isfinite(ya)]
    assert np.isin(np.round(finite, 9), np.round(resid[np.isfinite(resid)], 9)).all()
    assert not np.array_equal(ya, simulate(dict(cfg, seed=6)).results["y_observed"]["data"], equal_nan=True)


def test_real_noise_field_window_and_errors():
    rng = np.random.default_rng(1)
    bg = rng.normal(0, 1, (40, 50))
    results = {"img": {"data": np.zeros((10, 12)), "descriptor": __import__("scidatasim.utils", fromlist=["x"])
               .DataDescriptor("img", "NDARRAY", role="latent")}}
    added, realized = apply_observation(results, {"targets": ["img"], "operators": [
        {"type": "real_noise", "values": bg.tolist(), "offset": [5, 7]}]}, rng)
    assert np.array_equal(added["img_observed"]["data"], bg[5:15, 7:19])
    with pytest.raises(ValueError):
        apply_observation(results, {"targets": ["img"], "operators": [
            {"type": "real_noise", "values": np.zeros((5, 5)).tolist()}]}, rng)
    with pytest.raises(ValueError):
        apply_observation({"y": {"data": np.zeros(10), "descriptor": results["img"]["descriptor"]}},
                          {"targets": ["y"], "operators": [{"type": "real_noise", "values": [1.0, 2.0]}]}, rng)
    assert validate_observation_config({"operators": [{"type": "real_noise", "values": [0.0]}]})


def test_external_series_experiment(light_curve):
    path, _ = light_curve
    run = simulate(dict(experiment_type=EXTERNAL, source=path, value="flux", time="time", error="flux_err",
                        quality="quality", normalize="median", name="flux", units={"time": "d"}, seed=1))
    r = run.results
    assert set(r) == {"time", "flux", "flux_error"} and r["flux"]["descriptor"].role.value == "observation"
    assert len(r["flux"]["data"]) == 2890 and np.all(np.diff(r["time"]["data"]) > 0)
    assert run.ground_truth["fidelity"]["level"] == "observed" and not run.ground_truth["synthetic"]
    assert run.ground_truth["parameters"]["source.sha256"] == ingest.sha256_file(path)
    assert run.ground_truth["parameters"]["source.rows_used"] == 2890
    with pytest.raises(ValueError):
        simulate(dict(experiment_type=EXTERNAL, value="flux", seed=1))
