# tests/test_doe.py
import os

import pytest

from experiments.linear_function.logic import LinearFunctionExperiment
from simulator.doe import (append_results_to_doe_table, create_doe_table,
                           generate_parameter_combinations, run_parameter_sweep)
from simulator.persistence import load_experiment_record

BASE = {
    "experiment_type": "experiments.linear_function.logic.LinearFunctionExperiment",
    "n_points": 4, "x_min": 0.0, "x_max": 3.0,
}
RANGES = {"m": [1.0, 2.0], "c": [0.0, 1.0]}


def test_generate_parameter_combinations():
    combos = generate_parameter_combinations(RANGES)
    assert combos == [{"m": 1.0, "c": 0.0}, {"m": 1.0, "c": 1.0},
                      {"m": 2.0, "c": 0.0}, {"m": 2.0, "c": 1.0}]
    with pytest.raises(ValueError):
        generate_parameter_combinations({})


def test_sweep_runs_through_engine_with_full_provenance(tmp_path):
    out = str(tmp_path)
    runs = run_parameter_sweep(LinearFunctionExperiment, BASE, RANGES, output_dir=out,
                               generate_plots=False)
    assert len(runs) == 4
    for run in runs:
        record = load_experiment_record(out, run["record_id"])
        assert record.system_info, "sweep runs must record system info"
        assert record.software_versions, "sweep runs must record software versions"
        assert record.config["m"] == run["params"]["m"]
        assert record.end_time is not None
    # the base config is not mutated
    assert "m" not in BASE


def test_sweep_nested_output(tmp_path):
    nested = run_parameter_sweep(LinearFunctionExperiment, BASE, RANGES, output_dir=str(tmp_path),
                                 output_transform="nested", generate_plots=False)
    assert set(nested) == {"m=1.0, c=0.0", "m=1.0, c=1.0", "m=2.0, c=0.0", "m=2.0, c=1.0"}


def test_sweep_rejects_bad_arguments(tmp_path):
    with pytest.raises(TypeError):
        run_parameter_sweep(dict, BASE, RANGES, output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        run_parameter_sweep(LinearFunctionExperiment, BASE, RANGES, output_dir=str(tmp_path),
                            output_transform="bogus")


def test_append_results_to_doe_table(tmp_path):
    runs = run_parameter_sweep(LinearFunctionExperiment, BASE, RANGES, output_dir=str(tmp_path),
                               generate_plots=False)
    table = append_results_to_doe_table(create_doe_table(RANGES), runs)
    assert len(table) == 4
    # y_3 = m * 3 + c
    for _, row in table.iterrows():
        assert row["y_3"] == pytest.approx(row["m"] * 3 + row["c"])
