# tests/test_law_discovery.py
import json
import os

import numpy as np
import pytest

from experiments.law_discovery.logic import LAWS, HiddenLawExperiment, law_expression, law_suite_design
from simulator.bundle import check_answer
from simulator.config import yaml_load
from simulator.engine import SimulatorEngine, simulate
from simulator.envs import ExperimentEnv
from simulator.export.tasks import collect_bundle_tasks, score
from simulator.expressions import ExpressionError, evaluate, matches, normalise, parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAW = {"experiment_type": "experiments.law_discovery.logic.HiddenLawExperiment", "n_points": 30,
       "generate_plots": False, "seed": 3}


# ------------------------------------------------------------------ expressions
@pytest.mark.parametrize("candidate", ["2.3*x**1.5", "y = 2.3 x^1.5", "2.3*sqrt(x)**3", r"$2.3\cdot x^{1.5}$",
                                       r"\boxed{2.3x^{1.5}}", "exp(log(2.3) + 1.5*log(x))", "2.3e0*x**1.5"])
def test_equivalent_forms_match(candidate):
    assert matches(candidate, "2.3*x**1.5", (0.5, 5.0))


@pytest.mark.parametrize("candidate", ["2.35*x**1.5", "2.3*x**1.6", "2.3*x", "", "y ="])
def test_different_laws_do_not_match(candidate):
    assert not matches(candidate, "2.3*x**1.5", (0.5, 5.0))


@pytest.mark.parametrize("attack", ["__import__('os').system('x')", "x.real", "(lambda: 1)()", "open('f')",
                                    "x[0]", "[x for x in y]", "exp(x, 2)", "numpy.exp(x)", "z*2", "'a'*3",
                                    "x" + "+x" * 300])
def test_disallowed_constructs_are_rejected(attack):
    with pytest.raises(ExpressionError):
        parse(attack)
    assert not matches(attack, "x", (0, 1))


def test_evaluate_and_normalise():
    np.testing.assert_allclose(evaluate("4*exp(-0.5*x)*cos(2*x)", x=np.array([0.0, 1.0])),
                               [4.0, 4 * np.exp(-0.5) * np.cos(2)])
    assert normalise("y = 2x^2 + 1e-3") == "2 * x ** 2 + 1e-3"
    assert evaluate("pi + e") == pytest.approx(np.pi + np.e)


# ------------------------------------------------------------------ model
@pytest.mark.parametrize("family", sorted(LAWS))
def test_label_expression_reproduces_y(family):
    sim = simulate(dict(LAW, law=family))
    x, y = sim.results["x"]["data"], sim.results["y"]["data"]
    np.testing.assert_allclose(evaluate(sim.results["law_expression"]["data"], x=x), y, rtol=1e-5)
    assert sim.results["law_family"]["data"] == family
    assert np.all(np.isfinite(y))


def test_counterfactual_moves_structural_constants_off_textbook():
    for seed in range(20):
        sim = simulate(dict(LAW, law="inverse_square", counterfactual=True, seed=seed))
        p = sim.ground_truth["parameters"]["law.p"]
        assert abs(p - 2.0) >= 0.3 - 1e-9 and 1.4 <= p <= 2.8
    textbook = simulate(dict(LAW, law="inverse_square"))
    assert textbook.ground_truth["parameters"]["law.p"] == 2.0


def test_random_family_is_seeded_and_law_seed_fixes_the_law():
    a, b = simulate(dict(LAW, law="random", seed=9)), simulate(dict(LAW, law="random", seed=9))
    assert a.results["law_expression"]["data"] == b.results["law_expression"]["data"]
    families = {simulate(dict(LAW, law="random", seed=s)).results["law_family"]["data"] for s in range(30)}
    assert len(families) >= 5
    c = simulate(dict(LAW, law="power_law", law_seed=5, seed=1))
    d = simulate(dict(LAW, law="power_law", law_seed=5, seed=2))
    assert c.results["law_expression"]["data"] == d.results["law_expression"]["data"]


def test_fixed_parameters_and_ranges():
    sim = simulate(dict(LAW, law="power_law", parameters={"a": 2.0, "b": 1.25}, x_min=1.0, x_max=2.0))
    assert sim.results["law_expression"]["data"] == "2*x**1.25"
    assert sim.results["x"]["data"][0] == 1.0 and sim.results["x"]["data"][-1] == 2.0
    with pytest.raises(ValueError):
        HiddenLawExperiment(dict(LAW, law="unknown"))
    with pytest.raises(ValueError):
        simulate(dict(LAW, law="power_law", parameters={"zzz": 1.0}))


# ------------------------------------------------------------------ bundles and tasks
def test_law_bundle_questions(tmp_path):
    cfg = dict(LAW, law="hill", counterfactual=True, bundle=True,
               observation={"operators": [{"type": "gaussian", "snr_db": 30}]})
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(cfg)
    tasks = {t["template"]: t for t in collect_bundle_tasks(run.experiment_dir)}
    truth = run.results["law_expression"]["data"]
    assert tasks["law_family"]["answer"] == "hill" and "hill" in tasks["law_family"]["choices"]
    assert check_answer(tasks["law_formula"], truth)
    assert score(tasks["law_formula"], f"My fit gives `y = {truth}`") == 1.0
    assert score(tasks["law_formula"], "y = x") == 0.0
    assert "law_parameter" not in tasks  # off by default: they would reveal the family
    with open(os.path.join(run.experiment_dir, "bundle", "questions.json")) as f:
        public = json.load(f)
    assert all("answer" not in q for q in public)
    assert not any(truth in json.dumps(q) for q in public)


def test_parameter_questions_on_request(tmp_path):
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(
        dict(LAW, law="power_law", counterfactual=True, bundle=True, parameter_questions=True))
    templates = [t["template"] for t in collect_bundle_tasks(run.experiment_dir)]
    assert templates.count("law_parameter") == 2  # a and the counterfactual exponent b


def test_suite_design():
    design = law_suite_design(families=["power_law", "hill"], snr_db=(40, 10))
    assert len(design) == 8 and set(design.columns) == {"law", "counterfactual", "observation.operators.0.snr_db"}
    with pytest.raises(ValueError):
        law_suite_design(families=["nope"])


# ------------------------------------------------------------------ interactive environment
def _env():
    with open(os.path.join(ROOT, "examples", "example_10_law_discovery", "env.yaml")) as f:
        return ExperimentEnv(yaml_load(f))


def test_expression_target_environment():
    env = _env()
    obs = env.reset(seed=4)
    assert "formula" in obs["task"]
    truth = env.truth["law_expression"]
    out = env.step({"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}})
    assert len(out["observation"]["measurements"]["y_observed"]["values"]) == 30
    done = env.step({"type": "answer", "value": f"y = {truth}"})
    assert done["reward"] == 1.0 and done["info"]["relative_rms_error"] < 1e-5
    env.reset(seed=4)
    assert env.step({"type": "answer", "value": "x"})["reward"] == 0.0
    env.reset(seed=4)
    assert env.step({"type": "answer", "value": "__import__('os')"})["reward"] == 0.0


def test_failed_experiments_cost_budget_and_reveal_nothing():
    env = _env()
    env.reset(seed=1)
    out = env.step({"type": "experiment", "inputs": {"x_min": 5.0, "x_max": 1.0}})
    assert out["observation"]["error"] == "experiment failed for these inputs (no data)"
    assert env.state["experiments_used"] == 1 and out["observation"]["budget_left"] == 2
    assert env.truth["law_expression"] not in json.dumps(out)


def test_same_episode_seed_same_hidden_law():
    a, b = _env(), _env()
    a.reset(seed=7)
    b.reset(seed=7)
    assert a.truth == b.truth


def test_classical_baseline_solves_textbook_power_law():
    from examples.example_10_law_discovery.run_experiment import fit_best_law
    sim = simulate(dict(LAW, law="power_law", parameters={"a": 2.0, "b": 1.7},
                        observation={"operators": [{"type": "gaussian", "snr_db": 50}]}))
    family, formula = fit_best_law(sim.results["x"]["data"], sim.results["y_observed"]["data"])
    assert family == "power_law" and matches(formula, "2*x**1.7", (0.5, 5.0))


# ------------------------------------------------------------------ review regressions
@pytest.mark.parametrize("hostile", ["exp + x", "9" * 400, "exp", "x + log", "1e999999 * x"])
def test_hostile_input_is_rejected_not_crashing(hostile):
    assert not matches(hostile, "x", (1, 2))
    task = {"answer_type": "expression", "answer": "x", "domain": [1, 2], "variable": "x", "tolerance": 0.01}
    assert score(task, hostile) == 0.0


@pytest.mark.parametrize("candidate,reference", [
    ("log10(x)", "log(x)/log(10)"), ("x + 2*e-1", "x + 2*2.718281828459045-1"), ("3x(1-x)", "3*x*(1-x)"),
    ("3 x (1 - x)", "3*x*(1-x)"), (r"\frac{3x}{1}(1-x)", "3*x*(1-x)"), ("3x(1-x).", "3*x*(1-x)"),
    ("1e-3x + 1", "0.001*x + 1"), ("(x+1)(x-1)", "x**2 - 1")])
def test_written_forms(candidate, reference):
    assert matches(candidate, reference, (1.0, 2.0))


@pytest.mark.parametrize("text", ["y=2x^1.5\n\nThis fits the data well, exponent 1.5",
                                  "Answer: 2*x**1.5 (exponent ~1.5)", "The law is y = 2 x^1.5."])
def test_formula_found_in_prose(text):
    task = {"answer_type": "expression", "answer": "2*x**1.5", "domain": [0.5, 5], "variable": "x",
            "tolerance": 0.01}
    assert score(task, text) == 1.0


def test_negative_values_are_parenthesised_and_bad_ranges_rejected():
    expr = law_expression("hill", {"vmax": 2.0, "k": -2.0, "n": 2.0})
    assert "(-2)**2" in expr
    np.testing.assert_allclose(evaluate(expr, x=np.array([1.0])), [2.0 / 5.0])
    with pytest.raises(ValueError, match="not finite"):
        simulate(dict(LAW, law="inverse_square", x_range=[0.0, 5.0]))
    with pytest.raises(ValueError, match="fixed law family"):
        HiddenLawExperiment(dict(LAW, law="random", parameters={"a": 2.0}))


def test_shared_domain_and_shuffled_suite():
    ranges = {tuple(simulate(dict(LAW, law=f)).results["x"]["data"][[0, -1]]) for f in LAWS}
    assert ranges == {(0.5, 8.0)}  # the x range does not identify the family
    natural = simulate(dict(LAW, law="michaelis_menten", x_range="natural")).results["x"]["data"]
    assert natural[-1] == 10.0
    a, b = law_suite_design(seed=1), law_suite_design(seed=1)
    assert a.equals(b) and not a.equals(law_suite_design(shuffle=False))
    assert sorted(map(tuple, a.values.tolist())) == sorted(map(tuple, law_suite_design(shuffle=False).values.tolist()))
