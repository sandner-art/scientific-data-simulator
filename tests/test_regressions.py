# tests/test_regressions.py
"""Regression tests for defects found in the 0.1.0 code audit."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from simulator.data_handler import results_to_dataframe, save_csv
from simulator.engine import SimulatorEngine
from simulator.experiment_record import ExperimentRecord
from simulator.persistence import load_experiment_record, save_experiment_record
from simulator.utils import DataDescriptor, DataType


def test_descriptor_serialises_enum_value_not_name():
    d = DataDescriptor("x", DataType.DATAFRAME)
    assert d.to_dict()["data_type"] == "pd.DataFrame"


@pytest.mark.parametrize("raw", ["pd.DataFrame", "DATAFRAME", "DataType.DATAFRAME", DataType.DATAFRAME])
def test_descriptor_accepts_legacy_type_strings(raw):
    d = DataDescriptor.from_dict({"name": "x", "data_type": raw})
    assert d.data_type is DataType.DATAFRAME


def test_record_round_trip_restores_types(tmp_path):
    record = ExperimentRecord({"experiment_type": "a.B"}, None)
    record.add_output_data("arr", np.arange(4.0), DataDescriptor("arr", DataType.NDARRAY))
    record.add_output_data("df", pd.DataFrame({"a": [1, 2]}), DataDescriptor("df", DataType.DATAFRAME))
    record.add_output_data("n", np.int64(3), DataDescriptor("n", DataType.INT))  # numpy scalar
    save_experiment_record(record, str(tmp_path))
    loaded = load_experiment_record(str(tmp_path), record.experiment_id)
    assert isinstance(loaded.output_data["arr"]["data"], np.ndarray)
    assert isinstance(loaded.output_data["df"]["data"], pd.DataFrame)
    assert loaded.output_data["n"]["data"] == 3
    assert loaded.output_data["df"]["descriptor"].data_type is DataType.DATAFRAME


def test_validation_accepts_numpy_scalars_and_int_valued_floats(tmp_path):
    engine = SimulatorEngine(output_dir=str(tmp_path), verbose=False)
    engine._validate_results({
        "max": {"data": np.int64(5), "descriptor": DataDescriptor("max", DataType.FLOAT)},
        "mean": {"data": np.float64(2.5), "descriptor": DataDescriptor("mean", DataType.FLOAT)},
        "count": {"data": np.int32(2), "descriptor": DataDescriptor("count", DataType.INT)},
    })
    with pytest.raises(TypeError):
        engine._validate_results({"flag": {"data": True, "descriptor": DataDescriptor("flag", DataType.INT)}})


def test_validation_fills_missing_shape_and_rejects_wrong_shape(tmp_path):
    engine = SimulatorEngine(output_dir=str(tmp_path), verbose=False)
    desc = DataDescriptor("a", DataType.NDARRAY)
    validated = engine._validate_results({"a": {"data": np.zeros(3), "descriptor": desc}})
    assert validated["a"]["descriptor"].shape == (3,)
    assert desc.shape is None, "a shared descriptor must not be modified"
    # the same descriptor can then be reused with data of another length
    engine._validate_results({"a": {"data": np.zeros(7), "descriptor": desc}})
    with pytest.raises(ValueError):
        engine._validate_results({"a": {"data": np.zeros(3), "descriptor": DataDescriptor("a", DataType.NDARRAY, shape=(4,))}})


def test_save_csv_mixed_scalars_arrays_and_dataframes(tmp_path):
    results = {
        "t": {"data": np.arange(3), "descriptor": DataDescriptor("t", DataType.NDARRAY)},
        "short": {"data": np.arange(2), "descriptor": DataDescriptor("short", DataType.NDARRAY)},
        "mean": {"data": 1.5, "descriptor": DataDescriptor("mean", DataType.FLOAT)},
        "df": {"data": pd.DataFrame({"a": [7, 8, 9]}), "descriptor": DataDescriptor("df", DataType.DATAFRAME)},
    }
    path = tmp_path / "r.csv"
    save_csv(results, str(path))
    df = pd.read_csv(path)
    assert list(df.columns) == ["t", "short", "df.a", "mean"]
    assert len(df) == 3 and np.isnan(df["short"].iloc[2])
    assert (df["mean"] == 1.5).all()


def test_data_analysis_experiment_runs_end_to_end_with_csv(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"time": [0, 1, 2], "value": [10, 12, 11]}).to_csv(csv, index=False)
    engine = SimulatorEngine(output_dir=str(tmp_path / "out"), verbose=False)
    run = engine.run_config({
        "experiment_type": "experiments.data_analysis.logic.DataAnalysisExperiment",
        "input_data_path": str(csv), "save_csv": True, "generate_plots": False,
    })
    assert os.path.exists(os.path.join(run.experiment_dir, "results.csv"))
    assert not any("Could not save CSV" in m for m in run.record.log_messages)


def test_calibration_descriptor_shapes_match_truncated_data(tmp_path):
    from experiments.predator_prey_calibration.logic import PredatorPreyCalibrationExperiment
    obs = tmp_path / "obs.csv"
    pd.DataFrame({"time": [0, 1, 2], "prey_population": [1, 2, 3],
                  "predator_population": [1, 1, 1]}).to_csv(obs, index=False)
    cfg = dict(n_steps=10, initial_prey=10, initial_predators=2, prey_growth_rate=0.1,
               prey_death_rate=0.01, predator_growth_rate=0.01, predator_death_rate=0.1,
               observed_data_path=str(obs))
    exp = PredatorPreyCalibrationExperiment(cfg)
    state = exp.initialize(cfg)
    for i in range(cfg["n_steps"]):
        state = exp.run_step(state, i)
    res = exp.get_results()
    for key in ("time", "prey_population", "predator_population"):
        assert res[key]["descriptor"].shape == res[key]["data"].shape == (3,)


def test_failed_run_still_writes_record(tmp_path):
    engine = SimulatorEngine(output_dir=str(tmp_path), verbose=False)
    with pytest.raises(KeyError):
        engine.run_config({"experiment_type": "experiments.linear_function.logic.LinearFunctionExperiment"})
    (run_dir,) = os.listdir(tmp_path)
    with open(tmp_path / run_dir / "experiment_record.json") as f:
        record = json.load(f)
    assert any("Experiment failed" in m for m in record["log_messages"])
    assert record["end_time"] is not None


def test_software_versions_are_recorded(tmp_path):
    engine = SimulatorEngine(output_dir=str(tmp_path), verbose=False)
    run = engine.run_config({"experiment_type": "experiments.linear_function.logic.LinearFunctionExperiment",
                             "n_points": 3, "m": 1.0, "c": 0.0, "x_min": 0.0, "x_max": 1.0,
                             "generate_plots": False})
    assert "numpy" in run.record.software_versions
