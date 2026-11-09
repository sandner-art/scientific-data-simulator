# tests/test_envs_tasks_sbi.py
import json
import os

import numpy as np
import pytest
import yaml

from simulator.doe import create_doe_table, run_design
from simulator.engine import SimulatorEngine, simulate
from simulator.envs import ExperimentEnv, ExperimentTaskSpec
from simulator.export.sbi import (BatchSimulator, generate_training_data, load_training_data, sample_box_prior,
                                  training_data_from_runs)
from simulator.export.tasks import collect_bundle_tasks, public_view, read_tasks_jsonl, score, write_tasks_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "examples", "example_9_discovery_env", "env.yaml")) as f:
    SPEC = yaml.safe_load(f)
LV = dict(SPEC["base_config"], n_steps=500)


# ------------------------------------------------------------------ simulate()
def test_simulate_matches_engine(tmp_path):
    cfg = dict(LV, seed=4)
    a = simulate(cfg)
    b = SimulatorEngine(str(tmp_path), verbose=False).run_config(dict(cfg, generate_plots=False))
    np.testing.assert_array_equal(a.results["prey_population_observed"]["data"],
                                  b.results["prey_population_observed"]["data"])
    assert a.ground_truth["parameters"] == b.record.ground_truth["parameters"]
    assert not os.listdir(tmp_path / os.listdir(tmp_path)[0] / "data") == []  # engine wrote files; simulate did not


# ------------------------------------------------------------------ environment
def test_episode_is_reproducible_and_scored():
    env = ExperimentEnv(SPEC)
    first = env.reset(seed=11)
    assert "prey_growth_rate" in first["task"] and first["budget"] == 3
    truth = env.truth["prey_growth_rate"]
    assert 0.6 <= truth <= 1.6
    r1 = env.step({"type": "experiment", "inputs": {"initial_prey": 5, "initial_predators": 0}})
    env2 = ExperimentEnv(SPEC)
    env2.reset(seed=11)
    r2 = env2.step({"type": "experiment", "inputs": {"initial_prey": 5, "initial_predators": 0}})
    assert r1 == r2 and env2.truth == env.truth
    done = env.step({"type": "answer", "value": truth * 1.02})
    assert done["done"] and done["reward"] == 1.0 and done["info"]["correct"]
    with pytest.raises(RuntimeError):
        env.step({"type": "answer", "value": 1})


def test_wrong_answer_bad_inputs_and_budget():
    env = ExperimentEnv(SPEC)
    env.reset(seed=1)
    assert "outside" in env.step({"type": "experiment", "inputs": {"initial_prey": 500}})["observation"]["error"]
    assert "unknown inputs" in env.step({"type": "experiment", "inputs": {"seed": 1}})["observation"]["error"]
    assert "action must be" in env.step({"type": "guess"})["observation"]["error"]
    assert env.state["experiments_used"] == 0  # invalid actions do not consume the budget
    for _ in range(3):
        env.step({"type": "experiment", "inputs": {"initial_prey": 10}})
    out = env.step({"type": "experiment", "inputs": {"initial_prey": 10}})
    assert out["done"] and out["reward"] == 0.0
    env.reset(seed=1)
    assert env.step({"type": "answer", "value": "not a number"})["reward"] == 0.0


def test_categorical_target():
    spec = dict(SPEC, hidden={"functional_response": {"choices": ["linear", "holling2"]}},
                target="functional_response")
    spec["base_config"] = dict(SPEC["base_config"], handling_time=0.0)
    env = ExperimentEnv(spec)
    env.reset(seed=2)
    truth = env.truth["functional_response"]
    assert env.step({"type": "answer", "value": truth.upper()})["reward"] == 1.0


@pytest.mark.parametrize("bad", [dict(target="unknown"), dict(hidden={}), dict(budget=0),
                                 dict(controls={"prey_growth_rate": [0, 1]}), dict(hidden={"a": [2, 1]})])
def test_invalid_specs(bad):
    with pytest.raises(ValueError):
        ExperimentTaskSpec.from_dict(dict(SPEC, **bad))


def test_scripted_baseline_solves_the_example():
    from examples.example_9_discovery_env.run_experiment import baseline_agent
    env = ExperimentEnv(SPEC)
    rewards = [baseline_agent(env, seed)[0]["reward"] for seed in range(5)]
    assert np.mean(rewards) >= 0.8


# ------------------------------------------------------------------ tasks
@pytest.fixture(scope="module")
def design_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("design")
    base = dict(LV, bundle=True, generate_plots=False)
    base["observation"] = {"sampling": {"every": 25}, "operators": [{"type": "gaussian", "sigma": 0.2}]}
    run_design(base, create_doe_table({"prey_growth_rate": [0.9, 1.3]}), output_dir=str(out), base_seed=3)
    return str(out)


def test_collect_and_score_tasks(design_dir, tmp_path):
    tasks = collect_bundle_tasks(design_dir, relative_to=design_dir)
    assert len({t["experiment_id"] for t in tasks}) == 2
    for task in tasks:
        assert task["prompt"].endswith(("number.", "integer.")) or "exactly one of" in task["prompt"]
        assert os.path.exists(os.path.join(design_dir, task["image"]))
        assert "/" in task["image"] and "\\" not in task["image"]  # portable task files
        assert score(task, task["answer"]) == 1.0
    numeric = next(t for t in tasks if t["template"] == "max")
    assert score(numeric, f"I read about {numeric['answer']:.6f} from the plot.") == 1.0
    assert score(numeric, numeric["answer"] + 10 * numeric["tolerance"]) == 0.0
    assert score(numeric, "no idea") == 0.0
    choice = next((t for t in tasks if t["answer_type"] == "choice"), None)
    if choice:
        assert score(choice, f"The answer is {choice['answer']}.") == 1.0


def test_task_files(design_dir, tmp_path):
    tasks = collect_bundle_tasks(design_dir)
    public, private = tmp_path / "tasks.jsonl", tmp_path / "tasks_private.jsonl"
    assert write_tasks_jsonl(tasks, str(public)) == len(tasks)
    write_tasks_jsonl(tasks, str(private), include_answers=True)
    assert all("answer" not in t and "tolerance" not in t for t in read_tasks_jsonl(str(public)))
    assert read_tasks_jsonl(str(private)) == json.loads(json.dumps(tasks))


# ------------------------------------------------------------------ sbi
PRIOR = {"prey_growth_rate": [0.8, 1.4], "predator_death_rate": [0.3, 0.5]}


def test_sbi_training_data_is_reproducible(tmp_path):
    a = generate_training_data(LV, PRIOR, 6, seed=1, output_dir=str(tmp_path))
    b = generate_training_data(LV, PRIOR, 6, seed=1)
    np.testing.assert_array_equal(a["x"], b["x"])
    assert a["theta"].shape == (6, 2) and a["x"].shape[0] == 6
    assert [l["name"] for l in a["x_layout"]] == ["predator_population_observed", "prey_population_observed"]
    stored = load_training_data(str(tmp_path))
    np.testing.assert_array_equal(stored["x"], a["x"])
    assert stored["prior"]["bounds"] == PRIOR
    assert np.all((a["theta"] >= [0.8, 0.3]) & (a["theta"] <= [1.4, 0.5]))


def test_batch_simulator_gives_fresh_noise_and_respects_theta():
    sim = BatchSimulator(LV, ["prey_growth_rate"], seed=5)
    x1 = sim(np.array([[1.0], [1.0]]))
    assert not np.array_equal(x1[0], x1[1])  # independent noise per simulation
    assert len(sim.seeds) == 2 and len(set(sim.seeds)) == 2
    noiseless = BatchSimulator(dict(LV, observation={"sampling": {"every": 25}}), ["prey_growth_rate"], seed=5)
    lo, hi = noiseless(np.array([[0.8], [1.4]]))
    assert not np.allclose(lo, hi)
    with pytest.raises(ValueError):
        sim(np.ones((1, 3)))


def test_training_data_from_runs(tmp_path):
    out = run_design(dict(LV, generate_plots=False), create_doe_table({"prey_growth_rate": [0.9, 1.2]}),
                     output_dir=str(tmp_path), base_seed=2)
    data = training_data_from_runs(str(tmp_path), list(out.table["experiment_id"]), ["prey_growth_rate"])
    np.testing.assert_allclose(data["theta"][:, 0], [0.9, 1.2])
    prey = next(l for l in data["x_layout"] if l["name"] == "prey_population_observed")
    np.testing.assert_array_equal(data["x"][0][prey["start"]:prey["start"] + prey["length"]],
                                  out.results[0]["prey_population_observed"]["data"])


def test_nan_policy():
    cfg = dict(LV)
    cfg["observation"] = {"sampling": {"every": 25}, "operators": [{"type": "missing", "rate": 0.5}]}
    with pytest.raises(ValueError, match="NaN"):
        BatchSimulator(cfg, ["prey_growth_rate"], seed=1, nan_policy="error")(np.array([[1.0]]))
    x = BatchSimulator(cfg, ["prey_growth_rate"], seed=1, nan_policy="zero")(np.array([[1.0]]))
    assert not np.isnan(x).any()
    assert sample_box_prior(PRIOR, 3, seed=0).shape == (3, 2)
