# tests/test_storage.py
import json
import os

import numpy as np
import pandas as pd
import pytest

from simulator.engine import SimulatorEngine
from simulator.persistence import load_experiment_record
from simulator.storage import ChecksumError, resolve_storage_config

LINEAR = {"experiment_type": "experiments.linear_function.logic.LinearFunctionExperiment",
          "n_points": 100, "m": 2.0, "c": 5.0, "x_min": 0.0, "x_max": 10.0,
          "generate_plots": False, "seed": 1,
          "observation": {"operators": [{"type": "gaussian", "sigma": 1.0}]}}


def _run(tmp_path, **overrides):
    return SimulatorEngine(output_dir=str(tmp_path), verbose=False).run_config(dict(LINEAR, **overrides))


def _record_json(run):
    with open(os.path.join(run.experiment_dir, "experiment_record.json")) as f:
        return json.load(f)


def test_arrays_are_stored_by_role_with_checksums(tmp_path):
    run = _run(tmp_path)
    rec = _record_json(run)
    paths = {name: out["storage"]["path"] for name, out in rec["output_data"].items()}
    assert paths == {"x": "data/coordinate/x.npy", "y": "data/ground_truth/y.npy",
                     "y_observed": "data/observation/y_observed.npy"}
    for out in rec["output_data"].values():
        assert out["data"] is None and len(out["storage"]["sha256"]) == 64
    loaded = load_experiment_record(str(tmp_path), run.experiment_id)
    np.testing.assert_array_equal(loaded.output_data["y_observed"]["data"], run.results["y_observed"]["data"])


def test_tampered_file_is_detected(tmp_path):
    run = _run(tmp_path)
    path = os.path.join(run.experiment_dir, "data", "observation", "y_observed.npy")
    arr = np.load(path)
    arr[0] += 1.0
    np.save(path, arr)
    with pytest.raises(ChecksumError):
        load_experiment_record(str(tmp_path), run.experiment_id)
    # verification can be switched off explicitly
    load_experiment_record(str(tmp_path), run.experiment_id, verify=False)


def test_mmap_loading(tmp_path):
    run = _run(tmp_path)
    loaded = load_experiment_record(str(tmp_path), run.experiment_id, mmap=True)
    assert isinstance(loaded.output_data["y"]["data"], np.memmap)


def test_hdf5_storage(tmp_path):
    pytest.importorskip("h5py")
    run = _run(tmp_path, storage={"format": "hdf5"})
    rec = _record_json(run)
    assert rec["output_data"]["y"]["storage"]["dataset"] == "ground_truth/y"
    assert os.path.exists(os.path.join(run.experiment_dir, "data.h5"))
    loaded = load_experiment_record(str(tmp_path), run.experiment_id)
    np.testing.assert_array_equal(loaded.output_data["y"]["data"], run.results["y"]["data"])


def test_inline_storage_keeps_legacy_layout(tmp_path):
    run = _run(tmp_path, storage={"format": "inline"})
    rec = _record_json(run)
    assert isinstance(rec["output_data"]["y"]["data"], list)
    assert not os.path.exists(os.path.join(run.experiment_dir, "data"))
    loaded = load_experiment_record(str(tmp_path), run.experiment_id)
    np.testing.assert_allclose(loaded.output_data["y"]["data"], run.results["y"]["data"])


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_dataframe_outputs(tmp_path, fmt):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    csv = tmp_path / "in.csv"
    pd.DataFrame({"time": [0, 1, 2], "value": [1.5, 2.5, 3.5]}).to_csv(csv, index=False)
    run = SimulatorEngine(output_dir=str(tmp_path / "o"), verbose=False).run_config(
        {"experiment_type": "experiments.data_analysis.logic.DataAnalysisExperiment",
         "input_data_path": str(csv), "generate_plots": False, "storage": {"dataframes": fmt}})
    rec = _record_json(run)
    assert rec["output_data"]["loaded_data"]["storage"]["format"] == fmt
    assert rec["output_data"]["mean"]["data"] == 2.5  # scalars stay inline
    loaded = load_experiment_record(str(tmp_path / "o"), run.experiment_id)
    pd.testing.assert_frame_equal(loaded.output_data["loaded_data"]["data"], run.results["loaded_data"]["data"])


def test_storage_config_errors(tmp_path):
    with pytest.raises(ValueError):
        resolve_storage_config({"format": "zip"})
    with pytest.raises(ValueError):
        resolve_storage_config({"compression": "gzip"})
    with pytest.raises(ValueError):
        _run(tmp_path, storage={"format": "zip"})


def test_legacy_inline_record_still_loads(tmp_path):
    """Records written by 0.1.0 (inline data, 'DataType.X' strings) remain readable."""
    run_dir = tmp_path / "2025-03-16_10-00-00_legacy-id"
    run_dir.mkdir()
    legacy = {
        "experiment_id": "legacy-id", "start_time": "2025-03-16T10:00:00", "end_time": None,
        "config": {"experiment_type": "a.B"}, "experiment_logic_class_name": "B",
        "experiment_logic_module": "a", "experiment_description": "",
        "input_data_descriptors": {},
        "output_data": {"y": {"data": [1.0, 2.0],
                              "descriptor": {"name": "y", "data_type": "DataType.NDARRAY", "shape": [2],
                                             "units": None, "group": "default", "plot_type": None, "x_axis": None}}},
        "log_messages": [], "system_info": {}, "software_versions": {}, "llm_usage": {},
    }
    (run_dir / "experiment_record.json").write_text(json.dumps(legacy))
    rec = load_experiment_record(str(tmp_path), "legacy-id")
    np.testing.assert_array_equal(rec.output_data["y"]["data"], [1.0, 2.0])
    assert rec.ground_truth == {} and rec.randomness == {}
