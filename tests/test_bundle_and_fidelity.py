# tests/test_bundle_and_fidelity.py
import json
import os
import runpy

import numpy as np
import pandas as pd
import pytest

from simulator.bundle import check_answer, resolve_bundle_config
from simulator.doe import create_doe_table, run_design
from simulator.engine import SimulatorEngine
from simulator.persistence import load_experiment_record
from simulator.provenance import file_sha256

LV = {"experiment_type": "experiments.predator_prey.logic.PredatorPreyExperiment",
      "n_steps": 3000, "dt": 0.01, "integrator": "rk4", "clamp_negative": False,
      "initial_prey": 10.0, "initial_predators": 5.0, "prey_growth_rate": 1.1, "prey_death_rate": 0.4,
      "predator_growth_rate": 0.1, "predator_death_rate": 0.4, "generate_plots": False, "seed": 11,
      "observation": {"sampling": {"every": 25}, "operators": [{"type": "gaussian", "snr_db": 20}]},
      "bundle": True}


def _run(path, **overrides):
    return SimulatorEngine(output_dir=str(path), verbose=False).run_config(dict(LV, **overrides))


def _bundle(run):
    return os.path.join(run.experiment_dir, "bundle")


def _qa(run):
    with open(os.path.join(_bundle(run), "qa.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------ contents
def test_bundle_files_and_visibility(tmp_path):
    run = _run(tmp_path)
    files = run.record.bundle["files"]
    public = {n for n, f in files.items() if f["visibility"] == "public"}
    private = {n for n, f in files.items() if f["visibility"] == "private"}
    assert public == {"figure.png", "plot_figure.py", "caption.txt", "questions.json"}
    assert private == {"qa.json", "graph.json", "graph.dot", "description.md", "table.csv"}
    training = _run(tmp_path / "training", bundle={"public_table": True})
    assert training.record.bundle["files"]["table.csv"]["visibility"] == "public"
    for name, entry in files.items():
        assert file_sha256(os.path.join(run.experiment_dir, entry["path"])) == entry["sha256"]
    # the record on disk carries the manifest too
    assert load_experiment_record(str(tmp_path), run.experiment_id).bundle == run.record.bundle


def test_public_questions_carry_no_answers(tmp_path):
    run = _run(tmp_path)
    with open(os.path.join(_bundle(run), "questions.json")) as f:
        public = json.load(f)
    assert public and all("answer" not in q for q in public)
    assert [q["id"] for q in public] == [q["id"] for q in _qa(run)]


def test_answers_match_the_exact_arrays(tmp_path):
    run = _run(tmp_path)
    t = run.results["time"]["data"]
    prey = run.results["prey_population"]["data"]
    by_template = {(q["template"], q["series"] if isinstance(q["series"], str) else None): q for q in _qa(run)}
    assert by_template[("max", "prey_population")]["answer"] == pytest.approx(prey.max())
    assert by_template[("argmax", "prey_population")]["answer"] == pytest.approx(t[np.argmax(prey)])
    assert by_template[("final", "prey_population")]["answer"] == pytest.approx(prey[-1])
    assert by_template[("peak_count", "prey_population")]["answer"] == 2
    assert by_template[("peak_count", "predator_population")]["answer"] == 3
    params = {q["parameter"]: q["answer"] for q in _qa(run) if q["template"] == "parameter"}
    assert params == {"prey_growth_rate": 1.1, "prey_death_rate": 0.4,
                      "predator_growth_rate": 0.1, "predator_death_rate": 0.4}


def test_value_at_questions_quote_a_grid_point(tmp_path):
    run = _run(tmp_path)
    t, prey = run.results["time"]["data"], run.results["prey_population"]["data"]
    for q in _qa(run):
        if q["template"] == "value_at" and q["series"] == "prey_population":
            x = float(q["question"].rsplit("= ", 1)[1].rstrip("?"))
            assert prey[np.argmin(np.abs(t - x))] == pytest.approx(q["answer"], abs=q["tolerance"])


def test_check_answer():
    num = {"answer_type": "number", "answer": 10.0, "tolerance": 0.5}
    assert check_answer(num, 10.4) and check_answer(num, "9.6") and not check_answer(num, 10.6)
    assert not check_answer(num, "ten")
    assert check_answer({"answer_type": "integer", "answer": 3}, "3")
    assert check_answer({"answer_type": "choice", "answer": "prey_population"}, " Prey_Population ")
    assert check_answer({"answer_type": "number", "answer": 2.0, "tolerance": None}, 2.0)


def test_table_is_exactly_what_is_drawn(tmp_path):
    run = _run(tmp_path)
    table = pd.read_csv(os.path.join(_bundle(run), "table.csv"))
    np.testing.assert_allclose(table["prey_population"], run.results["prey_population"]["data"])
    observed = table["prey_population_observed"].dropna().to_numpy()
    np.testing.assert_allclose(observed, run.results["prey_population_observed"]["data"])
    np.testing.assert_allclose(table["time_observed"].dropna(), run.results["time_observed"]["data"])


def test_shipped_script_reproduces_the_figure_byte_for_byte(tmp_path):
    run = _run(tmp_path)
    regenerated = tmp_path / "regenerated.png"
    runpy.run_path(os.path.join(_bundle(run), "plot_figure.py"), run_name="__test__")["main"](str(regenerated))
    assert file_sha256(str(regenerated)) == file_sha256(os.path.join(_bundle(run), "figure.png"))


def test_bundle_is_deterministic_for_a_seed(tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    for name in ("figure.png", "table.csv", "qa.json", "questions.json", "caption.txt", "graph.json",
                 "plot_figure.py"):
        assert a.record.bundle["files"][name]["sha256"] == b.record.bundle["files"][name]["sha256"], name
    c = _run(tmp_path / "c", seed=12)
    assert a.record.bundle["files"]["figure.png"]["sha256"] != c.record.bundle["files"]["figure.png"]["sha256"]


@pytest.mark.parametrize("fmt", ["svg", "pdf"])
def test_vector_formats(tmp_path, fmt):
    run = _run(tmp_path, bundle={"figure": {"format": fmt}})
    assert os.path.exists(os.path.join(_bundle(run), f"figure.{fmt}"))


def test_hiding_latent_curves_turns_figure_questions_into_inference(tmp_path):
    run = _run(tmp_path, bundle={"figure": {"show_latent": False}})
    table = pd.read_csv(os.path.join(_bundle(run), "table.csv"))
    assert "prey_population" not in table.columns and "prey_population_observed" in table.columns
    assert {q["answerable_from"] for q in _qa(run)} == {"inference"}
    with open(os.path.join(_bundle(run), "caption.txt")) as f:
        caption = f.read()
    assert "Lines show" not in caption and "Markers show observations" in caption


def test_caption_does_not_leak_parameter_values(tmp_path):
    run = _run(tmp_path)
    with open(os.path.join(_bundle(run), "caption.txt")) as f:
        caption = f.read()
    for value in ("1.1", "0.4", "0.1"):
        assert value not in caption


def test_graph_reflects_law_shift(tmp_path):
    run = _run(tmp_path, law_shift={"changes": {"functional_response": "holling2", "handling_time": 0.3}})
    with open(os.path.join(_bundle(run), "graph.json")) as f:
        graph = json.load(f)
    terms = " ".join(e["term"] for e in graph["edges"])
    assert "Holling II" in terms and any(e["parameter"] == "handling_time" for e in graph["edges"])
    with open(os.path.join(_bundle(run), "description.md")) as f:
        assert "Law shift" in f.read()


def test_qa_options(tmp_path):
    run = _run(tmp_path, bundle={"qa": {"value_queries": 0, "parameters": ["prey_growth_rate"]}})
    templates = [q["template"] for q in _qa(run)]
    assert "value_at" not in templates and templates.count("parameter") == 1
    no_qa = _run(tmp_path / "noqa", bundle={"qa": {"enabled": False}, "graph": False})
    assert set(no_qa.record.bundle["files"]) == {"figure.png", "plot_figure.py", "table.csv",
                                                 "caption.txt", "description.md"}
    assert no_qa.record.bundle["files"]["table.csv"]["visibility"] == "private"


@pytest.mark.parametrize("bad", [{"colour": "red"}, {"figure": {"format": "gif"}},
                                 {"figure": {"show_latent": False, "show_observed": False}},
                                 {"figure": {"dpi": 0}}, {"qa": {"value_queries": -1}}, "yes"])
def test_bundle_config_errors(bad):
    with pytest.raises(ValueError):
        resolve_bundle_config(bad)


def test_bundle_without_time_series_still_writes_caption_and_qa(tmp_path):
    from simulator.base import ExperimentLogic
    from simulator.utils import DataDescriptor, DataType

    class Scalar(ExperimentLogic):
        inferable_parameters = ("k",)
        def __init__(self, config): self.k = config["k"]
        def initialize(self, config): return {}
        def run_step(self, state, step): return state
        def get_results(self):
            return {"k2": {"data": float(self.k) ** 2, "descriptor": DataDescriptor("k2", DataType.FLOAT, role="label")}}
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(
        {"k": 3.0, "bundle": True, "generate_plots": False}, experiment_logic_class=Scalar)
    assert run.record.bundle["figure"] is None
    assert [q["parameter"] for q in _qa(run)] == ["k"]


def test_design_collects_a_question_benchmark(tmp_path):
    base = dict(LV, n_steps=500)
    design = create_doe_table({"observation.operators.0.snr_db": [30, 10]})
    out = run_design(base, design, output_dir=str(tmp_path), replicates=2, base_seed=4)
    with open(tmp_path / "qa.jsonl") as f:
        private = [json.loads(line) for line in f]
    with open(tmp_path / "questions.jsonl") as f:
        public = [json.loads(line) for line in f]
    assert len(private) == len(public) > 0
    assert {p["experiment_id"] for p in private} == set(out.table["experiment_id"])
    assert all("answer" not in q for q in public) and all("answer" in q for q in private)
    assert all(os.path.exists(tmp_path / q["figure"]) for q in public)


# ------------------------------------------------------------------ fidelity
def test_fidelity_reference_vs_audit(tmp_path):
    ref = _run(tmp_path, bundle=None)
    assert ref.record.provenance["fidelity"]["level"] == "reference"
    assert ref.record.ground_truth["fidelity"]["level"] == "reference"
    audit = _run(tmp_path, bundle=None, integrator="euler", dt=1.0, n_steps=30, clamp_negative=True)
    fid = audit.record.provenance["fidelity"]
    assert fid["level"] == "audit" and "Forward Euler" in fid["note"] and "deliberately weak" in fid["note"]
    assert any("Model fidelity 'audit'" in m for m in audit.record.log_messages)
    coarse = _run(tmp_path, bundle=None, dt=0.1, n_steps=300)
    assert coarse.record.provenance["fidelity"]["level"] == "unverified"


@pytest.mark.parametrize("model, level", [
    ("experiments.example_experiment.logic.ExampleExperiment", "reference"),
    ("experiments.linear_function.logic.LinearFunctionExperiment", "reference"),
    ("experiments.example_random_walk.logic.RandomWalkExperiment", "reference"),
])
def test_bundled_models_declare_fidelity(tmp_path, model, level):
    cfg = {"experiment_type": model, "n_steps": 5, "amplitude": 1.0, "step_size": 1.0, "n_points": 5,
           "m": 1.0, "c": 0.0, "x_min": 0.0, "x_max": 1.0, "generate_plots": False, "seed": 0}
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(cfg)
    assert run.record.provenance["fidelity"]["level"] == level


def test_unknown_fidelity_level_is_rejected(tmp_path):
    from experiments.linear_function.logic import LinearFunctionExperiment

    class Bad(LinearFunctionExperiment):
        fidelity_level = "excellent"
    with pytest.raises(ValueError, match="unknown level"):
        SimulatorEngine(str(tmp_path), verbose=False).run_config(
            {"n_points": 3, "m": 1.0, "c": 0.0, "x_min": 0.0, "x_max": 1.0, "generate_plots": False},
            experiment_logic_class=Bad)


def test_public_files_do_not_leak_answers_through_tolerances(tmp_path):
    """A parameter question's tolerance is 5 % of the answer; it must stay private."""
    from simulator.export.tasks import collect_bundle_tasks, public_view
    run = _run(tmp_path)
    with open(os.path.join(_bundle(run), "questions.json")) as f:
        public = json.load(f)
    assert all("tolerance" not in q and "answer" not in q for q in public)
    assert all(q["tolerance_rule"] for q in public)
    assert {q["tolerance_rule"] for q in public if q["template"] == "parameter"} == {"5 % of the true value"}
    for task in collect_bundle_tasks(run.experiment_dir):
        assert "tolerance" not in public_view(task) and "tolerance" in task
