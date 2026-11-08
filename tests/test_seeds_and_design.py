# tests/test_seeds_and_design.py
import json
import os

import numpy as np
import pandas as pd
import pytest

from simulator.doe import (add_replicates_and_seeds, apply_overrides, create_doe_table,
                           get_by_path, run_design, run_parameter_sweep, set_by_path)
from simulator.engine import SimulatorEngine
from simulator.persistence import load_experiment_record
from simulator.rng import RunRandomness, derive_seeds
from simulator.schema import validate_engine_config

WALK = {"experiment_type": "experiments.example_random_walk.logic.RandomWalkExperiment",
        "n_steps": 50, "step_size": 1.0, "generate_plots": False}


def _engine(tmp_path):
    return SimulatorEngine(output_dir=str(tmp_path), verbose=False)


# ------------------------------------------------------------------ seeds
def test_same_seed_reproduces_run(tmp_path):
    a = _engine(tmp_path).run_config(dict(WALK, seed=42))
    b = _engine(tmp_path).run_config(dict(WALK, seed=42))
    np.testing.assert_array_equal(a.results["position"]["data"], b.results["position"]["data"])
    assert a.record.seed == 42 and a.record.randomness["seed_generated"] is False


def test_missing_seed_is_drawn_recorded_and_replayable(tmp_path):
    first = _engine(tmp_path).run_config(dict(WALK))
    assert first.record.randomness["seed_generated"] is True
    stored = load_experiment_record(str(tmp_path), first.experiment_id)
    assert stored.config["seed"] == first.record.seed
    replay = _engine(tmp_path).run_config(stored.config)  # the stored config is a full recipe
    np.testing.assert_array_equal(first.results["position"]["data"], replay.results["position"]["data"])


def test_streams_are_independent_and_deterministic():
    s1 = RunRandomness(5, False).streams()
    s2 = RunRandomness(5, False).streams()
    assert s1["model"].random() == s2["model"].random()
    assert s1["observation"].random() != RunRandomness(5, False).streams()["model"].random()


def test_derive_seeds_is_deterministic_and_distinct():
    a, b = derive_seeds(1, 100), derive_seeds(1, 100)
    assert a == b and len(set(a)) == 100
    assert all(0 <= s < 2 ** 63 for s in a)


def test_record_provenance(tmp_path):
    run = _engine(tmp_path).run_config(dict(WALK, seed=1))
    prov = run.record.provenance
    assert prov["synthetic"] is True
    assert len(prov["config_sha256"]) == 64
    assert prov["code"]["class"] == "RandomWalkExperiment"
    assert len(prov["code"]["source_sha256"]) == 64


# ------------------------------------------------------------------ schema
@pytest.mark.parametrize("bad", [{"n_steps": -1}, {"seed": -3}, {"static_plot_format": "gif"},
                                 {"generate_plots": "maybe"}])
def test_engine_config_validation(bad):
    with pytest.raises(ValueError, match="Invalid configuration"):
        validate_engine_config(dict(WALK, **bad))


def test_engine_config_accepts_legacy_null_format():
    assert validate_engine_config(dict(WALK, static_plot_format="null")).static_plot_format is None


# ------------------------------------------------------------------ config paths
def test_set_and_get_by_path():
    cfg = {"observation": {"operators": [{"type": "gaussian", "sigma": 1.0}]}}
    set_by_path(cfg, "observation.operators.0.sigma", 2.0)
    set_by_path(cfg, "new.nested.key", 3)
    assert get_by_path(cfg, "observation.operators.0.sigma") == 2.0
    assert cfg["new"]["nested"]["key"] == 3
    base = {"a": {"b": 1}}
    out = apply_overrides(base, {"a.b": np.float64(2.0)})
    assert base["a"]["b"] == 1 and out["a"]["b"] == 2.0 and type(out["a"]["b"]) is float


# ------------------------------------------------------------------ designs
def test_lhs_is_stratified_in_every_dimension():
    n = 20
    t = create_doe_table({"a": [0.0, 1.0], "b": [10.0, 20.0]}, "lhs", n_samples=n, seed=3)
    assert len(t) == n
    for col, (lo, hi) in {"a": (0, 1), "b": (10, 20)}.items():
        bins = np.floor((t[col] - lo) / (hi - lo) * n).astype(int)
        assert sorted(bins) == list(range(n)), "one sample per stratum"


@pytest.mark.parametrize("design", ["sobol", "halton", "random", "lhs"])
def test_sampled_designs_respect_bounds_and_seed(design):
    spec = {"k": {"low": 1e-3, "high": 1e1, "scale": "log"}, "n": {"low": 2, "high": 5, "integer": True}}
    t1 = create_doe_table(spec, design, n_samples=64, seed=11)
    t2 = create_doe_table(spec, design, n_samples=64, seed=11)
    pd.testing.assert_frame_equal(t1, t2)
    assert t1["k"].between(1e-3, 1e1).all()
    assert set(t1["n"]) <= {2, 3, 4, 5}


def test_design_validation_errors():
    with pytest.raises(ValueError):
        create_doe_table({"a": [0, 1]}, "lhs")  # n_samples missing
    with pytest.raises(ValueError):
        create_doe_table({"a": [1, 0]}, "lhs", n_samples=4)
    with pytest.raises(ValueError):
        create_doe_table({"a": [0, 1]}, "taguchi", n_samples=4)


def test_replicates_and_seed_modes():
    base = pd.DataFrame({"x": [1, 2, 3]})
    per_row = add_replicates_and_seeds(base, replicates=2, base_seed=9)
    assert list(per_row["design_point"]) == [0, 0, 1, 1, 2, 2]
    assert list(per_row["replicate"]) == [0, 1, 0, 1, 0, 1]
    assert per_row["seed"].nunique() == 6
    crn = add_replicates_and_seeds(base, replicates=2, base_seed=9, seed_mode="per_replicate")
    assert crn.groupby("replicate")["seed"].nunique().eq(1).all()
    assert crn["seed"].nunique() == 2


# ------------------------------------------------------------------ running designs
def test_run_design_writes_tidy_outputs(tmp_path):
    table = create_doe_table({"step_size": [1.0, 2.0]})
    design = run_design(dict(WALK, seed=100), table, output_dir=str(tmp_path), replicates=3)
    assert len(design.table) == 6 and design.base_seed == 100
    assert (design.table["status"] == "ok").all()
    for f in ("design.csv", "design_manifest.json", "design_results.csv"):
        assert os.path.exists(tmp_path / f)
    with open(tmp_path / "design_manifest.json") as f:
        assert json.load(f)["base_seed"] == 100
    # each row can be re-run on its own from its recorded seed
    row = design.table.iloc[4]
    rerun = _engine(tmp_path / "rerun").run_config(dict(WALK, step_size=row["step_size"], seed=int(row["seed"])))
    np.testing.assert_array_equal(rerun.results["position"]["data"], design.results[4]["position"]["data"])


def test_run_design_is_reproducible_from_base_seed(tmp_path):
    table = create_doe_table({"step_size": [1.0, 2.0]})
    d1 = run_design(WALK, table, output_dir=str(tmp_path / "a"), replicates=2, base_seed=5)
    d2 = run_design(WALK, table, output_dir=str(tmp_path / "b"), replicates=2, base_seed=5)
    assert list(d1.table["seed"]) == list(d2.table["seed"])
    for r1, r2 in zip(d1.results, d2.results):
        np.testing.assert_array_equal(r1["position"]["data"], r2["position"]["data"])


def test_parallel_matches_serial(tmp_path):
    table = create_doe_table({"step_size": [0.5, 1.0, 1.5, 2.0]})
    serial = run_design(WALK, table, output_dir=str(tmp_path / "s"), base_seed=8)
    parallel = run_design(WALK, table, output_dir=str(tmp_path / "p"), base_seed=8, n_jobs=2)
    for r1, r2 in zip(serial.results, parallel.results):
        np.testing.assert_array_equal(r1["position"]["data"], r2["position"]["data"])


def test_run_design_continue_on_error(tmp_path):
    table = pd.DataFrame({"step_size": [1.0, None]})
    base = dict(WALK)
    base["experiment_type"] = "experiments.linear_function.logic.LinearFunctionExperiment"
    design = run_design(base, table, output_dir=str(tmp_path), on_error="continue", base_seed=1)
    assert (design.table["status"] == "failed").all()
    assert design.table["error"].str.contains("KeyError").all()


def test_sweep_supports_nested_noise_keys_and_replicates(tmp_path):
    from experiments.linear_function.logic import LinearFunctionExperiment
    base = {"n_points": 5, "m": 1.0, "c": 0.0, "x_min": 0.0, "x_max": 1.0, "generate_plots": False,
            "extra": {"level": 0}}
    runs = run_parameter_sweep(LinearFunctionExperiment, base, {"extra.level": [1, 2]},
                               output_dir=str(tmp_path), replicates=2, base_seed=3)
    assert len(runs) == 4
    for run in runs:
        assert load_experiment_record(str(tmp_path), run["record_id"]).config["extra"]["level"] == run["params"]["extra.level"]
