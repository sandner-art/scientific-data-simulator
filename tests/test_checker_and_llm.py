# tests/test_checker_and_llm.py
"""The model checker must accept sound models and reject typical generated-code bugs."""
import textwrap

import pytest

from simulator.checker import check_model, check_source, lint_source
from simulator.llm import CallableLLMClient, draft_model, extract_code

LV = {"n_steps": 1000, "dt": 0.02, "integrator": "rk4", "clamp_negative": False, "initial_prey": 10.0,
      "initial_predators": 5.0, "prey_growth_rate": 1.1, "prey_death_rate": 0.4,
      "predator_growth_rate": 0.1, "predator_death_rate": 0.4}
DECAY_CONFIG = {"n_steps": 200, "dt": 0.05, "k": 0.7, "x0": 2.0}

GOOD = textwrap.dedent('''
    import numpy as np
    from simulator.base import ExperimentLogic
    from simulator.utils import DataDescriptor, DataType

    class Decay(ExperimentLogic):
        fidelity_level = "unverified"
        fidelity_note = "LLM-generated; not verified"
        inferable_parameters = ("k",)
        parameter_units = {"k": "1/s", "x0": "m", "dt": "s"}

        def __init__(self, config):
            self.k, self.x0, self.dt = config["k"], config["x0"], config["dt"]

        def initialize(self, config):
            self.t, self.x = [0.0], [self.x0]
            return {"x": self.x0}

        def run_step(self, state, step):
            x = state["x"]
            k1 = -self.k * x
            k2 = -self.k * (x + 0.5 * self.dt * k1)
            x = x + self.dt * k2                      # midpoint rule, order 2
            self.t.append((step + 1) * self.dt)
            self.x.append(x)
            return {"x": x}

        def get_results(self):
            return {
                "time": {"data": np.array(self.t), "descriptor": DataDescriptor(
                    "time", DataType.NDARRAY, units="s", group="time_series", role="coordinate")},
                "x": {"data": np.array(self.x), "descriptor": DataDescriptor(
                    "x", DataType.NDARRAY, units="m", group="time_series", plot_type="line",
                    x_axis="time", role="latent")},
            }
''')


def variant(old, new):
    assert old in GOOD
    return GOOD.replace(old, new)


# ------------------------------------------------------------------ bundled models
@pytest.mark.parametrize("model, config", [
    ("experiments.predator_prey.logic.PredatorPreyExperiment", LV),
    ("experiments.oscillator.logic.OscillatorExperiment", {"n_steps": 500, "dt": 0.01, "stiffness": 40.0, "x0": 0.1}),
    ("experiments.example_random_walk.logic.RandomWalkExperiment", {"n_steps": 50, "step_size": 1.0}),
    ("experiments.linear_function.logic.LinearFunctionExperiment",
     {"n_points": 5, "m": 2.0, "c": 1.0, "x_min": 0.0, "x_max": 1.0}),
])
def test_bundled_models_pass_the_gate(model, config):
    report = check_model(model, config)
    assert report.accepted, report.feedback()


def test_convergence_order_is_measured():
    report = check_model("experiments.predator_prey.logic.PredatorPreyExperiment", LV)
    assert report.get("convergence").data["orders"][-1] == pytest.approx(4.0, abs=0.2)
    euler = check_model("experiments.predator_prey.logic.PredatorPreyExperiment", dict(LV, integrator="euler"))
    assert euler.get("convergence").data["orders"][-1] == pytest.approx(1.0, abs=0.2)
    assert euler.get("invariants").status == "warn"  # the audit model drifts, as documented


# ------------------------------------------------------------------ generated code
def test_good_generated_model_is_accepted():
    report = check_source(GOOD, "Decay", DECAY_CONFIG)
    assert report.accepted, report.feedback()
    assert report.get("convergence").data["orders"][-1] == pytest.approx(2.0, abs=0.2)


@pytest.mark.parametrize("old, new, failing_check", [
    ('x = x + self.dt * k2', 'x = x + self.dt * k2 + 0.01 * np.random.normal()', "lint"),
    ('x = x + self.dt * k2', 'import random\n            x = x + self.dt * k2 * random.random()', "lint"),
    ('k1 = -self.k * x', 'import os; os.system("echo hi"); k1 = -self.k * x', "lint"),
    ('x = x + self.dt * k2', 'x = x - 0.7 * x * 0.05', "sensitivity"),           # ignores k (dead parameter)
    ('x = x + self.dt * k2', 'x = x + k2 * 0.05', "convergence"),                # step not scaled by dt
    ('x = x + self.dt * k2', 'x = x + self.dt * k2 + self.rng.normal() * 1e-3', None),  # fine: uses self.rng
])
def test_typical_bugs_are_rejected(old, new, failing_check):
    report = check_source(variant(old, new), "Decay", DECAY_CONFIG)
    if failing_check is None:
        assert report.accepted, report.feedback()
    else:
        assert not report.accepted
        assert failing_check in [r.name for r in report.failures], report.feedback()


def test_hidden_state_breaks_determinism():
    source = variant('        x = state["x"]\n', '        x = state["x"] + 1e-3 * _CALLS[0]\n        _CALLS[0] += 1\n')
    source = source.replace("class Decay(ExperimentLogic):", "_CALLS = [0]\n\n\nclass Decay(ExperimentLogic):")
    assert "_CALLS[0] += 1" in source
    report = check_source(source, "Decay", DECAY_CONFIG)
    assert "determinism" in [r.name for r in report.failures]


def test_lint_blocks_execution_and_timeout_is_enforced():
    dangerous = variant('k1 = -self.k * x', 'import subprocess; k1 = -self.k * x')
    report = check_source(dangerous, "Decay", DECAY_CONFIG)
    assert report.get("runs").status == "skip"  # never executed
    endless = variant('k1 = -self.k * x', 'while True: pass')
    report = check_source(endless, "Decay", DECAY_CONFIG, timeout=10)
    assert report.get("runs").status == "fail" and "timed out" in report.get("runs").detail


def test_lint_details():
    assert lint_source("from numpy.random import normal\n")[0].status == "fail"
    assert lint_source("import numpy as np\nrng = np.random.default_rng(1)\n")[0].status == "pass"
    assert lint_source("import time\nt = time.time()\n")[0].status == "warn"
    assert lint_source("def f(:\n")[0].status == "fail"


def test_report_markdown_and_feedback():
    report = check_source(variant('x = x + self.dt * k2', 'x = x - 0.7 * x * 0.05'), "Decay", DECAY_CONFIG)
    md = report.to_markdown()
    assert "rejected" in md and "sensitivity" in md
    assert "sensitivity" in report.feedback()


# ------------------------------------------------------------------ drafting loop
def test_draft_model_retries_with_checker_feedback(tmp_path):
    answers = iter(["Here you go:\n```python\n" + variant('x = x + self.dt * k2', 'x = x - 0.7 * x * 0.05') + "```",
                    "Fixed:\n```python\n" + GOOD + "```"])
    prompts = []

    def fake_llm(prompt):
        prompts.append(prompt)
        return next(answers)
    client = CallableLLMClient(fake_llm, name="fake")
    draft = draft_model(client, "Exponential decay dx/dt = -k x.", DECAY_CONFIG, class_name="Decay")
    assert draft.accepted and len(draft.attempts) == 2
    assert "sensitivity" in prompts[1]  # the second prompt carries the checker's findings
    assert "sensitivity" in draft.provenance()["attempts"][0]["failures"]
    assert draft.usage["calls"] == 2
    path = draft.save(str(tmp_path / "decay" / "logic.py"))
    assert (tmp_path / "decay" / "logic_check.md").exists() and (tmp_path / "decay" / "__init__.py").exists()
    assert open(path).read() == GOOD.strip() + "\n"


def test_rejected_draft_cannot_be_saved(tmp_path):
    client = CallableLLMClient(lambda p: "```python\nimport socket\n```")
    draft = draft_model(client, "anything", DECAY_CONFIG, class_name="Decay", max_attempts=2)
    assert not draft.accepted and len(draft.attempts) == 2
    with pytest.raises(ValueError, match="failed the model checker"):
        draft.save(str(tmp_path / "x.py"))


def test_extract_code():
    assert extract_code("text\n```python\nx = 1\n```\nmore") == "x = 1\n"
    assert extract_code("x = 2") == "x = 2\n"
