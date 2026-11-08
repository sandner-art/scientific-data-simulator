# tests/test_review_fixes.py
"""Regression tests for issues found in an independent review of 0.2.0.dev0."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from simulator.base import ExperimentLogic
from simulator.doe import create_doe_table, run_design
from simulator.engine import SimulatorEngine
from simulator.observation import apply_observation
from simulator.persistence import load_experiment_record
from simulator.utils import DataDescriptor, DataType

WALK = {"experiment_type": "experiments.example_random_walk.logic.RandomWalkExperiment",
        "n_steps": 30, "step_size": 1.0, "generate_plots": False}
LV = {"experiment_type": "experiments.predator_prey.logic.PredatorPreyExperiment",
      "n_steps": 100, "dt": 0.01, "integrator": "rk4", "clamp_negative": False,
      "initial_prey": 10.0, "initial_predators": 5.0, "prey_growth_rate": 1.1, "prey_death_rate": 0.4,
      "predator_growth_rate": 0.1, "predator_death_rate": 0.4, "generate_plots": False, "seed": 3}


def _engine(path):
    return SimulatorEngine(output_dir=str(path), verbose=False)


def test_rerunning_saved_design_csv_reproduces_seeds_and_data(tmp_path):
    """design.csv mixes float and 63-bit int columns; seeds must survive exactly."""
    table = create_doe_table({"step_size": [0.5, 1.5]})
    first = run_design(WALK, table, output_dir=str(tmp_path / "a"), base_seed=123, replicates=2)
    saved = pd.read_csv(tmp_path / "a" / "design.csv")
    again = run_design(WALK, saved, output_dir=str(tmp_path / "b"))
    assert [int(s) for s in again.table["seed"]] == [int(s) for s in first.table["seed"]]
    for r1, r2 in zip(first.results, again.results):
        np.testing.assert_array_equal(r1["position"]["data"], r2["position"]["data"])
    rec = load_experiment_record(str(tmp_path / "b"), again.table["experiment_id"][0])
    assert rec.config["seed"] == int(first.table["seed"][0])


def test_integer_design_columns_stay_integers(tmp_path):
    table = pd.DataFrame({"n_steps": [10, 20], "step_size": [0.5, 1.0]})
    out = run_design(WALK, table, output_dir=str(tmp_path), base_seed=1)
    rec = load_experiment_record(str(tmp_path), out.table["experiment_id"][0])
    assert type(rec.config["n_steps"]) is int


@pytest.mark.parametrize("key", ["seed", "observation.operators.0.sigma", "storage.format", "experiment_type"])
def test_law_shift_rejects_engine_settings(tmp_path, key):
    with pytest.raises(ValueError, match="law_shift may only change model parameters"):
        _engine(tmp_path).run_config(dict(WALK, law_shift={key: 1}))


def test_law_shift_on_n_steps_takes_effect(tmp_path):
    run = _engine(tmp_path).run_config(dict(WALK, seed=1, law_shift={"n_steps": 5}))
    assert len(run.results["position"]["data"]) == 6


def test_colliding_output_names_get_distinct_files(tmp_path):
    class Clash(ExperimentLogic):
        def __init__(self, config): pass
        def initialize(self, config): return {}
        def run_step(self, state, step): return state
        def get_results(self):
            return {"a b": {"data": np.zeros(3), "descriptor": DataDescriptor("a b", DataType.NDARRAY, role="latent")},
                    "a_b": {"data": np.ones(3), "descriptor": DataDescriptor("a_b", DataType.NDARRAY, role="latent")},
                    "A_B": {"data": np.full(3, 2.0), "descriptor": DataDescriptor("A_B", DataType.NDARRAY, role="latent")}}
    run = _engine(tmp_path).run_config({"generate_plots": False, "seed": 0}, experiment_logic_class=Clash)
    loaded = load_experiment_record(str(tmp_path), run.experiment_id)
    np.testing.assert_array_equal(loaded.output_data["a b"]["data"], np.zeros(3))
    np.testing.assert_array_equal(loaded.output_data["a_b"]["data"], np.ones(3))
    np.testing.assert_array_equal(loaded.output_data["A_B"]["data"], np.full(3, 2.0))


class _Strings(ExperimentLogic):
    def __init__(self, config): pass
    def initialize(self, config): return {}
    def run_step(self, state, step): return state
    def get_results(self):
        return {"labels": {"data": np.array(["ab", "cd"]), "descriptor": DataDescriptor("labels", DataType.NDARRAY, role="label")},
                "x": {"data": np.arange(2.0), "descriptor": DataDescriptor("x", DataType.NDARRAY, role="latent")}}


def test_hdf5_keeps_string_arrays_inline(tmp_path):
    pytest.importorskip("h5py")
    run = _engine(tmp_path).run_config({"generate_plots": False, "seed": 0, "storage": {"format": "hdf5"}},
                                       experiment_logic_class=_Strings)
    loaded = load_experiment_record(str(tmp_path), run.experiment_id)
    assert list(loaded.output_data["labels"]["data"]) == ["ab", "cd"]
    np.testing.assert_array_equal(loaded.output_data["x"]["data"], np.arange(2.0))


def test_failed_data_write_still_leaves_a_record(tmp_path, monkeypatch):
    import simulator.persistence as persistence

    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(persistence, "write_outputs", broken)
    with pytest.raises(OSError, match="disk full"):
        _engine(tmp_path).run_config(dict(WALK, seed=1))
    (run_dir,) = os.listdir(tmp_path)
    with open(tmp_path / run_dir / "experiment_record.json") as f:
        record = json.load(f)
    assert any("Writing data files failed" in m for m in record["log_messages"])


def test_save_failure_does_not_mask_model_error(tmp_path, monkeypatch):
    import simulator.engine as engine_module

    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(engine_module, "save_experiment_record", broken)
    with pytest.raises(KeyError):  # the model's own error, not the save error
        _engine(tmp_path).run_config({"experiment_type": "experiments.linear_function.logic.LinearFunctionExperiment"})


@pytest.mark.parametrize("offset", [-2, 10, 99])
def test_sampling_offset_must_be_in_range(offset):
    results = {"x": {"data": np.arange(10.0), "descriptor": DataDescriptor("x", DataType.NDARRAY, role="latent")}}
    with pytest.raises(ValueError, match="offset"):
        apply_observation(results, {"sampling": {"every": 3, "offset": offset}}, np.random.default_rng(0))


def test_results_csv_aligns_sampled_observations_with_their_time(tmp_path):
    cfg = dict(LV, save_csv=True, observation={"sampling": {"every": 25, "offset": 5},
                                               "operators": [{"type": "gaussian", "sigma": 0.1}]})
    run = _engine(tmp_path).run_config(cfg)
    df = pd.read_csv(os.path.join(run.experiment_dir, "results.csv"))
    assert len(df) == 101
    observed_rows = df[df["prey_population_observed"].notna()]
    assert list(observed_rows.index) == [5, 30, 55, 80]
    np.testing.assert_allclose(observed_rows["time_observed"], observed_rows["time"])


@pytest.mark.parametrize("value, expected", [("no", False), ("false", False), ("yes", True), ("1", True)])
def test_string_booleans_from_yaml_or_env_are_respected(tmp_path, value, expected):
    run = _engine(tmp_path).run_config(dict(WALK, seed=1, save_csv=value))
    assert os.path.exists(os.path.join(run.experiment_dir, "results.csv")) is expected


def test_explicit_class_is_named_in_the_stored_config(tmp_path):
    from experiments.linear_function.logic import LinearFunctionExperiment
    cfg = dict(WALK, n_points=3, m=1.0, c=0.0, x_min=0.0, x_max=1.0)  # names RandomWalkExperiment
    run = _engine(tmp_path).run_config(cfg, experiment_logic_class=LinearFunctionExperiment)
    assert run.record.config["experiment_type"] == "experiments.linear_function.logic.LinearFunctionExperiment"
    assert any("replaced by the explicitly passed class" in m for m in run.record.log_messages)
    # re-running the stored config runs the same model
    rerun = _engine(tmp_path).run_config(run.record.config)
    np.testing.assert_array_equal(rerun.results["y"]["data"], run.results["y"]["data"])


def test_model_cannot_corrupt_the_stored_config(tmp_path):
    class Greedy(ExperimentLogic):
        def __init__(self, config):
            self.k = config.pop("k")  # mutates its config
        def initialize(self, config): return {}
        def run_step(self, state, step): return state
        def get_results(self):
            return {"k": {"data": float(self.k), "descriptor": DataDescriptor("k", DataType.FLOAT)}}
    run = _engine(tmp_path).run_config({"k": 2.0, "generate_plots": False}, experiment_logic_class=Greedy)
    assert run.record.config["k"] == 2.0
    _engine(tmp_path).run_config(run.record.config, experiment_logic_class=Greedy)


def test_integer_parameters_need_integer_bounds():
    with pytest.raises(ValueError, match="integer bounds"):
        create_doe_table({"n": {"low": 0.5, "high": 2.5, "integer": True}}, "lhs", n_samples=4, seed=0)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_dataframe_index_round_trips(tmp_path, fmt):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")

    class Frame(ExperimentLogic):
        def __init__(self, config): pass
        def initialize(self, config): return {}
        def run_step(self, state, step): return state
        def get_results(self):
            df = pd.DataFrame({"v": [1.0, 2.0]}, index=pd.Index(["a", "b"], name="site"))
            return {"table": {"data": df, "descriptor": DataDescriptor("table", DataType.DATAFRAME)}}
    run = _engine(tmp_path).run_config({"generate_plots": False, "storage": {"dataframes": fmt}},
                                       experiment_logic_class=Frame)
    loaded = load_experiment_record(str(tmp_path), run.experiment_id)
    pd.testing.assert_frame_equal(loaded.output_data["table"]["data"], run.results["table"]["data"])
