# tests/test_heat_and_the_well.py
"""Heat equation model (verification) and the The Well exporter."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from experiments.heat.logic import HeatExperiment
from simulator.checker import check_model
from simulator.engine import SimulatorEngine, simulate

HEAT = "experiments.heat.logic.HeatExperiment"
MODES_1D = [{"k": 1, "a": 1.0, "b": 0.3}, {"k": 3, "a": 0.2, "b": -0.5}]
MODES_2D = [{"k": [1, 0], "a": 1.0, "b": 0.0}, {"k": [1, 2], "a": 0.3, "b": -0.4}, {"k": [-2, 1], "a": 0.0, "b": 0.5}]
BASE = {"experiment_type": HEAT, "alpha": 1e-3, "length": 1.0, "nx": 32, "dt": 0.1, "n_steps": 200,
        "seed": 1, "generate_plots": False}


def heat(**kw):
    model = HeatExperiment(dict(BASE, **kw))
    return model, model.get_results()


def spectral_laplacian(u, length):
    """Exact Laplacian of a band-limited periodic field (last 1 or 2 axes)."""
    axes = tuple(range(1, u.ndim))
    k = [2 * np.pi * np.fft.fftfreq(n, d=length / n) for n in u.shape[1:]]
    k2 = sum(np.meshgrid(*[kk ** 2 for kk in k], indexing="ij"))
    return np.real(np.fft.ifftn(-k2 * np.fft.fftn(u, axes=axes), axes=axes))


def semi_discrete(model, t):
    """Exact-in-time solution of the FD-in-space system (1-D): mode k decays with 4 sin^2(k dx/2)/dx^2."""
    x, u = model.coordinates(), model.mean
    for m in model.modes:
        k = 2 * np.pi * m["k"][0] / model.length
        lam = 4 * np.sin(k * model.dx / 2) ** 2 / model.dx ** 2
        u = u + np.exp(-model.alpha * lam * t) * (m["a"] * np.cos(k * x) + m["b"] * np.sin(k * x))
    return u


# ---------------------------------------------------------------------------- heat: verification
@pytest.mark.parametrize("dims, modes", [(1, MODES_1D), (2, MODES_2D), (1, None), (2, None)])
def test_exact_solution_satisfies_the_pde(dims, modes):
    _, r = heat(dims=dims, modes=modes, k_max=3, dt=0.01, n_steps=300)
    u, t = r["u"]["data"], r["time"]["data"]
    u_t = (u[2:] - u[:-2]) / (t[2:, None] - t[:-2, None]).reshape((-1,) + (1,) * dims)
    residual = u_t - 1e-3 * spectral_laplacian(u[1:-1], 1.0)
    assert np.max(np.abs(residual)) / np.max(np.abs(u_t)) < 1e-5


def test_each_fourier_mode_decays_at_the_exact_rate():
    model, r = heat(dims=2, modes=MODES_2D, n_steps=100)
    u = r["u"]["data"]
    spectrum = np.fft.fft2(u, axes=(1, 2))
    t = r["time"]["data"]
    for m in MODES_2D:
        i, j = m["k"][0] % 32, m["k"][1] % 32
        k2 = (2 * np.pi) ** 2 * (m["k"][0] ** 2 + m["k"][1] ** 2)
        ratio = np.abs(spectrum[:, i, j]) / np.abs(spectrum[0, i, j])
        np.testing.assert_allclose(ratio, np.exp(-1e-3 * k2 * t), rtol=1e-10)
    assert np.allclose(u[0], model.exact_solution(np.array([0.0]))[0])


@pytest.mark.parametrize("dims", [1, 2])
@pytest.mark.parametrize("integrator", ["exact", "ftcs"])
def test_mean_conserved_and_l2_norm_non_increasing(dims, integrator):
    _, r = heat(dims=dims, integrator=integrator, k_max=5)
    mean, l2 = r["mean"]["data"], r["l2_norm"]["data"]
    assert np.max(np.abs(mean - mean[0])) < 1e-13
    assert mean[0] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(l2) <= 1e-13)
    assert l2[-1] < l2[0]


def test_ftcs_is_first_order_in_time():
    errors = []
    for dt in (0.4, 0.2, 0.1, 0.05):  # nx = 32: r = 0.41 ... 0.05, all stable
        model, r = heat(modes=MODES_1D, integrator="ftcs", dt=dt, n_steps=int(round(20 / dt)))
        errors.append(np.max(np.abs(r["u"]["data"][-1] - semi_discrete(model, 20.0))))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    np.testing.assert_allclose(orders, 1.0, atol=0.02)  # measured 1.002, 1.001, 1.000


def test_ftcs_converges_to_the_exact_solution_second_order_in_space():
    errors = []
    for nx in (16, 32, 64, 128):  # fixed r = 0.25: dt ~ dx^2, so the total error is O(dx^2) = O(dt)
        dt = 0.25 * (1.0 / nx) ** 2 / 1e-3
        n = int(round(20 / dt))
        _, exact = heat(modes=MODES_1D, nx=nx, dt=dt, n_steps=n)
        _, ftcs = heat(modes=MODES_1D, nx=nx, dt=dt, n_steps=n, integrator="ftcs")
        errors.append(np.max(np.abs(exact["u"]["data"][-1] - ftcs["u"]["data"][-1])))
    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    np.testing.assert_allclose(orders, 2.0, atol=0.05)  # measured 2.00, 2.00, 2.00
    assert errors[-1] < 1e-4


@pytest.mark.parametrize("dims, r_value", [(1, 0.6), (2, 0.3)])
def test_ftcs_beyond_the_stability_limit_blows_up_as_documented(dims, r_value):
    nx = 32
    dt = r_value * (1.0 / nx) ** 2 / 1e-3
    model, r = heat(dims=dims, nx=nx, dt=dt, n_steps=400, integrator="ftcs", k_max=3)
    assert model.r == pytest.approx(r_value) and model.r > model.r_limit
    fidelity = model.fidelity({})
    assert fidelity["level"] == "audit" and "UNSTABLE" in fidelity["note"]
    u = r["u"]["data"]
    assert np.max(np.abs(u[0])) < 10
    assert not np.all(np.isfinite(u[-1])) or np.max(np.abs(u[-1])) > 1e20  # round-off grows like |1-4dr|^n
    # at the limit itself the scheme is stable
    _, stable = heat(dims=dims, nx=nx, dt=0.5 / dims * (1.0 / nx) ** 2 / 1e-3, n_steps=400,
                     integrator="ftcs", k_max=3)
    assert np.all(np.diff(stable["l2_norm"]["data"]) <= 1e-13)
    assert "UNSTABLE" not in HeatExperiment(dict(BASE, integrator="ftcs")).fidelity({})["note"]


# ---------------------------------------------------------------------------- heat: framework
def test_outputs_roles_and_shapes():
    _, r1 = heat(n_steps=10)
    _, r2 = heat(dims=2, n_steps=10, nx=16)
    assert r1["u"]["data"].shape == (11, 32) and r2["u"]["data"].shape == (11, 16, 16)
    assert r2["y"]["descriptor"].role.value == "coordinate"
    assert r1["u_probe"]["data"].shape == (11,) and r1["u_probe"]["descriptor"].x_axis == "time"
    np.testing.assert_array_equal(r2["u_probe"]["data"], r2["u"]["data"][:, 16 // 3, 16 // 7])
    roles = {n: i["descriptor"].role.value for n, i in r2.items()}
    assert roles == {"time": "coordinate", "x": "coordinate", "y": "coordinate", "u": "latent",
                     "u_probe": "latent", "mean": "label", "l2_norm": "label"}


def test_seeded_reproducibility_and_recorded_modes():
    a = simulate(dict(BASE, dims=2, n_steps=20))
    b = simulate(dict(BASE, dims=2, n_steps=20))
    c = simulate(dict(BASE, dims=2, n_steps=20, seed=2))
    np.testing.assert_array_equal(a.results["u"]["data"], b.results["u"]["data"])
    assert not np.array_equal(a.results["u"]["data"], c.results["u"]["data"])
    modes = a.ground_truth["parameters"]["modes"]
    assert len(modes) == 4 and all(len(m["k"]) == 2 and 0 < max(map(abs, m["k"])) <= 4 for m in modes)
    # the recorded modes reproduce the run without the seed
    replay = HeatExperiment(dict(BASE, dims=2, n_steps=20, modes=modes, seed=99)).get_results()
    np.testing.assert_allclose(replay["u"]["data"], a.results["u"]["data"], atol=1e-14)
    assert a.ground_truth["fidelity"]["level"] == "reference"


def test_units_conversion():
    pytest.importorskip("pint")
    si = simulate(dict(BASE, alpha=1e-6, length=0.1, dt=2.0, n_steps=20))
    units = simulate(dict(BASE, alpha="1 mm^2/s", length="10 cm", dt="2000 ms", n_steps=20))
    assert units.model_config["alpha"] == pytest.approx(1e-6)
    assert units.ground_truth["parameter_units"]["alpha"] == "m**2/s"
    np.testing.assert_allclose(units.results["u"]["data"], si.results["u"]["data"], rtol=1e-12)


def test_model_checker_accepts_both_integrators():
    exact = check_model(HEAT, dict(BASE, n_steps=50))
    assert exact.accepted, exact.feedback()
    assert exact.get("convergence").status == "skip"  # random initial condition: stochastic in the seed
    ftcs = check_model(HEAT, dict(BASE, n_steps=50, dt=0.5, integrator="ftcs", modes=MODES_1D))
    assert ftcs.accepted, ftcs.feedback()
    assert ftcs.get("convergence").status == "pass"
    assert ftcs.get("convergence").data["orders"][-1] == pytest.approx(1.0, abs=0.15)  # measured 1.09
    assert ftcs.get("sensitivity").status == "pass" and ftcs.get("global_rng").status == "pass"
    assert ftcs.get("invariants").status == "pass"


def test_engine_run_with_observation_and_files(tmp_path):
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(dict(
        BASE, dims=2, nx=16, n_steps=20, observation={"operators": [{"type": "gaussian", "sigma": 0.01}]}))
    assert "u_probe_observed" in run.results and "u_observed" not in run.results
    assert os.path.exists(os.path.join(run.experiment_dir, "data", "ground_truth", "u.npy"))
    gt = json.load(open(os.path.join(run.experiment_dir, "ground_truth.json")))
    assert gt["parameters"]["stability_number"] == pytest.approx(1e-3 * 0.1 * 16 ** 2)


@pytest.mark.parametrize("bad", [dict(dims=3), dict(integrator="rk4"), dict(k_max=16), dict(alpha=-1.0),
                                 dict(modes=[{"k": 16, "a": 1.0}]), dict(modes=[{"k": [1, 0], "a": 1.0}]),
                                 dict(probe_index=32)])
def test_invalid_heat_options(bad):
    with pytest.raises(ValueError):
        HeatExperiment(dict(BASE, **bad))


# ---------------------------------------------------------------------------- The Well export
@pytest.fixture(scope="module")
def heat_design(tmp_path_factory):
    from simulator.doe import create_doe_table, run_design
    out = str(tmp_path_factory.mktemp("design"))
    table = create_doe_table({"alpha": [0.001, 0.002]}, "full_factorial")
    run_design(dict(BASE, dims=2, nx=16, dt=1.0, n_steps=12), table, output_dir=out, replicates=3, base_seed=7,
               generate_plots=False)
    return out


def _export(src, out, **kw):
    from simulator.export.the_well import export_the_well
    return export_the_well(src, out, "sds_heat", **kw)


def test_the_well_schema(heat_design, tmp_path):
    h5py = pytest.importorskip("h5py")
    yaml = pytest.importorskip("yaml")
    out = str(tmp_path / "well")
    export = _export(heat_design, out, splits={"train": 2 / 3, "test": 1 / 3})
    assert export.simulation_parameters == ["alpha"] and export.spatial_dims == ["x", "y"]
    assert export.files["train"] == ["data/train/sds_heat_alpha_0.001.hdf5", "data/train/sds_heat_alpha_0.002.hdf5"]
    assert export.n_trajectories == {"train": 4, "test": 2}
    manifest = json.load(open(os.path.join(out, "sds_export.json")))
    with h5py.File(os.path.join(out, export.files["train"][1]), "r") as f:
        assert f.attrs["dataset_name"] == "sds_heat" and f.attrs["grid_type"] == "cartesian"
        assert f.attrs["n_spatial_dims"] == 2 and f.attrs["n_trajectories"] == 2
        assert list(f.attrs["simulation_parameters"]) == ["alpha"] and f.attrs["alpha"] == 0.002
        assert list(f["dimensions"].attrs["spatial_dims"]) == ["x", "y"]
        assert f["dimensions/time"].shape == (13,) and not f["dimensions/time"].attrs["sample_varying"]
        np.testing.assert_allclose(f["dimensions/x"][:], np.arange(16) / 16)
        bc = f["boundary_conditions/y_periodic"]
        assert bc.attrs["bc_type"] == "PERIODIC" and list(bc.attrs["associated_dims"]) == ["y"]
        assert bc["mask"][:].tolist() == [True] + [False] * 14 + [True]
        assert f["scalars/alpha"][()] == 0.002 and not f["scalars/alpha"].attrs["time_varying"]
        u = f["t0_fields/u"]
        assert u.shape == (2, 13, 16, 16) and u.dtype == np.float32
        assert list(u.attrs["dim_varying"]) == [True, True]
        assert u.attrs["sample_varying"] and u.attrs["time_varying"] and u.attrs["units"] == "K"
        assert list(f["t1_fields"].attrs["field_names"]) == [] and list(f["t2_fields"].attrs["field_names"]) == []
        # trajectory 0 is the run recorded for it, with its exact field
        entry = next(e for e in manifest["trajectories"] if e["file"] == export.files["train"][1] and e["trajectory"] == 0)
        run_dir = next(os.path.join(heat_design, d) for d in os.listdir(heat_design) if d.endswith(entry["experiment_id"]))
        np.testing.assert_allclose(u[0], np.load(os.path.join(run_dir, "data", "ground_truth", "u.npy")), rtol=1e-6)
        assert list(f.attrs["sds_fidelity"]) == ["reference", "reference"]
    stats = yaml.safe_load(open(os.path.join(out, "stats.yaml")))
    assert set(stats) == {"mean", "std", "mean_delta", "std_delta", "rms", "rms_delta"}
    assert stats["mean"]["u"] == pytest.approx(1.0, abs=1e-6)  # the conserved mean of every run


def test_the_well_export_is_deterministic_and_guarded(heat_design, tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    export = _export(heat_design, a, split="valid")
    _export(heat_design, b, split="valid")
    for rel in export.files["valid"] + ["stats.yaml", "README.md", "sds_export.json"]:
        assert open(os.path.join(a, rel), "rb").read() == open(os.path.join(b, rel), "rb").read()
    with pytest.raises(FileExistsError):
        _export(heat_design, a)
    _export(heat_design, a, overwrite=True)
    assert sorted(os.listdir(os.path.join(a, "data"))) == ["train"]
    with pytest.raises(ValueError, match="Unknown split"):
        _export(heat_design, str(tmp_path / "c"), split="validation")


def test_the_well_export_rejects_mixed_grids_and_models(heat_design, tmp_path):
    engine = SimulatorEngine(str(tmp_path / "other"), verbose=False)
    engine.run_config(dict(BASE, dims=2, nx=8, k_max=3, dt=1.0, n_steps=12))
    with pytest.raises(ValueError, match="grid"):
        _export([heat_design, str(tmp_path / "other")], str(tmp_path / "w1"))
    ftcs = SimulatorEngine(str(tmp_path / "ftcs"), verbose=False)
    ftcs.run_config(dict(BASE, dims=2, nx=16, dt=1.0, n_steps=12, alpha=0.001, integrator="ftcs"))
    with pytest.raises(ValueError, match="integrator"):
        _export([heat_design, str(tmp_path / "ftcs")], str(tmp_path / "w2"))


def test_the_well_loader_reads_the_export(heat_design, tmp_path):
    pytest.importorskip("the_well")
    from the_well.data import WellDataset
    from the_well.data.normalization import ZScoreNormalization

    out = str(tmp_path / "well")
    _export(heat_design, out, splits={"train": 2 / 3, "test": 1 / 3})
    ds = WellDataset(path=out, well_split_name="train", n_steps_input=4, n_steps_output=2)
    assert ds.metadata.n_spatial_dims == 2 and ds.metadata.spatial_resolution == (16, 16)
    assert ds.metadata.field_names[0] == ["u"] and ds.metadata.constant_scalar_names == ["alpha"]
    assert ds.metadata.boundary_condition_types == ["PERIODIC"]
    assert len(ds) == 4 * (13 - 6 + 1)
    samples = [ds[i] for i in range(len(ds))]
    s = samples[0]
    assert tuple(s["input_fields"].shape) == (4, 16, 16, 1) and tuple(s["output_fields"].shape) == (2, 16, 16, 1)
    assert s["boundary_conditions"].tolist() == [[2.0, 2.0], [2.0, 2.0]]  # BoundaryCondition.PERIODIC
    assert float(s["constant_scalars"][0]) == pytest.approx(0.001)
    assert tuple(s["space_grid"].shape) == (16, 16, 2)
    for sample in samples:  # the conserved mean survives the round trip
        assert float(sample["input_fields"].mean()) == pytest.approx(1.0, abs=1e-5)
    normed = WellDataset(path=out, well_split_name="test", use_normalization=True,
                         normalization_type=ZScoreNormalization)
    assert abs(float(normed[0]["input_fields"].mean())) < 1e-3


def test_fractional_split_with_one_run_per_parameter_set(tmp_path):
    pytest.importorskip("h5py")
    from simulator.doe import create_doe_table, run_design
    out = str(tmp_path / "design")
    table = create_doe_table({"alpha": [0.0005, 0.003]}, "lhs", n_samples=10, seed=1)
    run_design(dict(BASE, nx=16, dt=1.0, n_steps=8), table, output_dir=out, base_seed=3, generate_plots=False)
    export = _export(out, str(tmp_path / "well"), splits={"train": 0.6, "valid": 0.2, "test": 0.2})
    assert export.n_trajectories == {"train": 6, "valid": 2, "test": 2}
