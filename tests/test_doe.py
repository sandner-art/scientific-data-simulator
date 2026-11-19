# tests/test_doe.py
import os

import pytest

from scidatasim.experiments.linear_function.logic import LinearFunctionExperiment
from scidatasim.doe import (append_results_to_doe_table, create_doe_table,
                           generate_parameter_combinations, run_parameter_sweep)
from scidatasim.persistence import load_experiment_record

BASE = {
    "experiment_type": "scidatasim.experiments.linear_function.logic.LinearFunctionExperiment",
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


def test_one_at_a_time_and_morris():
    import numpy as _np

    from scidatasim.doe import create_doe_table, elementary_effects
    oat = create_doe_table({"a": [1, 2, 3], "b": {"levels": [10, 20], "baseline": 20}}, "one_at_a_time")
    assert oat.to_dict("records") == [{"a": 1, "b": 20, "varied": "baseline"}, {"a": 2, "b": 20, "varied": "a"},
                                      {"a": 3, "b": 20, "varied": "a"}, {"a": 1, "b": 10, "varied": "b"}]
    m = create_doe_table({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": {"low": 1.0, "high": 100.0, "scale": "log"}},
                         "morris", n_samples=20, seed=3)
    assert len(m) == 20 * 4 and set(m["varied"]) == {"start", "x", "y", "z"}
    assert m["x"].between(0, 1).all() and m["z"].between(1, 100).all()
    for _, traj in m.groupby("trajectory"):  # each step changes exactly the named parameter
        diffs = traj[["x", "y", "z"]].diff().iloc[1:].ne(0)
        assert (diffs.sum(axis=1) == 1).all()
    m["out"] = 5 * m["x"] + m["y"] ** 2 + 0 * m["z"]
    ee = elementary_effects(m, "out").set_index("parameter")
    assert ee.loc["x", "mu_star"] == _np.float64(5.0) or abs(ee.loc["x", "mu_star"] - 5.0) < 1e-9
    assert ee.loc["z", "mu_star"] == 0.0 and ee.loc["y", "sigma"] > 0 and ee.index[0] == "x"


def test_morris_design_runs(tmp_path):
    from scidatasim.doe import create_doe_table, elementary_effects, run_design
    table = create_doe_table({"m": [0.5, 2.0], "c": [-1.0, 1.0]}, "morris", n_samples=3, seed=1)
    base = {"experiment_type": "scidatasim.experiments.linear_function.logic.LinearFunctionExperiment",
            "n_points": 5, "m": 1.0, "c": 0.0, "x_min": 0.0, "x_max": 1.0}
    out = run_design(base, table, output_dir=str(tmp_path), base_seed=4, generate_plots=False)
    frame = out.table if hasattr(out, "table") else out
    assert len(frame) == 9 and "varied" in frame.columns
