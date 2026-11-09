# tests/test_review2_fixes.py
"""Regression tests for issues found in an independent review of 0.3.0.dev0."""
import os
import random
import sys
import textwrap

import numpy as np
import pytest

from simulator.checker import check_model, check_source, lint_source
from simulator.config import load_config, yaml_load
from simulator.engine import simulate

DECAY = textwrap.dedent('''
    import numpy as np
    from simulator.base import ExperimentLogic
    from simulator.utils import DataDescriptor, DataType

    class M(ExperimentLogic):
        inferable_parameters = ("k",)
        def __init__(self, config):
            self.k = config["k"]
        def initialize(self, config):
            self.x = [1.0]
            return {"x": 1.0}
        def run_step(self, state, step):
            x = state["x"] * np.exp(-self.k)
            self.x.append(x)
            return {"x": x}
        def get_results(self):
            x = np.array(self.x)
            return {"t": {"data": np.arange(len(x), dtype=float), "descriptor": DataDescriptor(
                        "t", DataType.NDARRAY, role="coordinate")},
                    "x": {"data": x, "descriptor": DataDescriptor("x", DataType.NDARRAY, x_axis="t",
                                                                   role="latent")}}
''')


# ------------------------------------------------------------------ checker isolation
def test_forged_report_is_rejected():
    forged = 'import json\nprint("@@REPORT@@" + json.dumps({"model": "x", "results": []}))\n' + DECAY.replace(
        'self.k = config["k"]', 'raise RuntimeError("broken")')
    report = check_source(forged, "M", {"n_steps": 3, "k": 0.1})
    assert not report.accepted


def test_child_process_gets_no_secrets(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "sk-secret-123")
    leaking = DECAY.replace('self.k = config["k"]', 'import os; raise RuntimeError(str(os.environ.get("FAKE_API_KEY")))')
    report = check_source(leaking, "M", {"n_steps": 3, "k": 0.1})
    assert "sk-secret-123" not in report.get("runs").detail and "None" in report.get("runs").detail


def test_check_model_lints_before_importing(tmp_path, monkeypatch):
    marker = tmp_path / "ran.txt"
    module = tmp_path / "evil_model.py"
    module.write_text(f"import os\nos.system('echo x > {marker}')\n" + DECAY)
    monkeypatch.syspath_prepend(str(tmp_path))
    report = check_model("evil_model.M", {"n_steps": 3, "k": 0.1})
    assert report.get("lint").status == "fail" and report.get("runs").status == "skip"
    assert not marker.exists() and "evil_model" not in sys.modules


def test_check_model_restores_global_rng():
    np.random.seed(7)
    random.seed(7)
    expected_np, expected_py = np.random.random(), random.random()
    np.random.seed(7)
    random.seed(7)
    check_model("experiments.example_random_walk.logic.RandomWalkExperiment", {"n_steps": 5, "step_size": 1.0})
    assert np.random.random() == expected_np and random.random() == expected_py


@pytest.mark.parametrize("source", [
    "from os import system\nsystem('x')",
    "import os\ngetattr(os, 'system')('x')",
    "import importlib\nimportlib.import_module('subprocess')",
    "import os\nos.execvp('a', [])",
    "from pathlib import Path\nPath('a').unlink()",
    "import numpy as np\nr = np.random\nr.seed(1)",
    "import sys\nf = sys._getframe(1)",
    "def f(fr):\n    return fr.f_back.f_locals",
])
def test_lint_catches_escapes(source):
    assert lint_source(source)[0].status == "fail"


def test_seeded_stdlib_generator_is_allowed():
    assert lint_source("import random\nrng = random.Random(3)\nrng.random()")[0].status == "pass"


def test_sensitivity_tests_parameters_with_defaults():
    report = check_model("experiments.oscillator.logic.OscillatorExperiment",
                         {"n_steps": 200, "dt": 0.01, "stiffness": 40.0, "x0": 0.1})
    assert set(report.get("sensitivity").data["changes"]) == {"stiffness", "damping"}


# ------------------------------------------------------------------ units and YAML
def test_yaml_scientific_notation(tmp_path):
    assert yaml_load("a: 1e-3\nb: 5E6\nc: '1e-3'\nd: holling2") == {"a": 0.001, "b": 5e6, "c": "1e-3", "d": "holling2"}
    cfg = tmp_path / "c.yaml"
    cfg.write_text("experiment_type: experiments.oscillator.logic.OscillatorExperiment\n"
                   "n_steps: 10\nstiffness: 4e1\ndamping: 1e-3\n")
    config = load_config(str(cfg))
    assert config["damping"] == 0.001 and config["stiffness"] == 40.0


def test_numeric_strings_are_in_declared_units():
    pytest.importorskip("pint")
    sim = simulate({"experiment_type": "experiments.oscillator.logic.OscillatorExperiment", "n_steps": 10,
                    "stiffness": 40.0, "damping": "1e-3", "seed": 0})
    assert sim.ground_truth["parameters"]["damping"] == pytest.approx(1e-3)


def test_geometry_extent_follows_length_units():
    pytest.importorskip("pint")
    sim = simulate({"experiment_type": "experiments.geometry.logic.TriangleExperiment", "length_units": "m",
                    "extent": "200 cm", "seed": 1})
    assert sim.ground_truth["parameters"]["extent"] == pytest.approx(2.0)
    assert np.max(sim.results["vertices"]["data"]) <= 2.0
    assert sim.results["vertices"]["descriptor"].units == "m"


# ------------------------------------------------------------------ leaks
def test_env_state_does_not_reveal_seeds():
    from simulator.envs import ExperimentEnv
    from tests.test_envs_tasks_sbi import SPEC
    env = ExperimentEnv(SPEC)
    env.reset(seed=3)
    env.step({"type": "experiment", "inputs": {"initial_prey": 5}})
    assert "episode_seed" not in env.state and all("seed" not in h for h in env.state["history"])
    assert env.private_state["episode_seed"] == 3 and "truth" in env.private_state


def test_public_task_view_has_no_seed():
    from simulator.export.tasks import public_view
    task = {"id": "a/q1", "seed": 5, "answer": 1.0, "tolerance": 0.1, "answer_type": "number"}
    assert public_view(task) == {"id": "a/q1", "answer_type": "number"}
    assert public_view(task, include_seeds=True)["seed"] == 5


# ------------------------------------------------------------------ scoring and tolerances
@pytest.mark.parametrize("text", ["4.57 cm^2", r"\boxed{4.57\,\text{cm}^2}", "The area is 4.57 m².",
                                  "answer: 4.57 (I first thought 5)", "about 4.57 cm**2"])
def test_free_text_scoring_ignores_unit_exponents(text):
    from simulator.export.tasks import score
    assert score({"answer_type": "number", "answer": 4.57, "tolerance": 0.01}, text) == 1.0


def test_thousands_separator():
    from simulator.export.tasks import score
    assert score({"answer_type": "number", "answer": 1234.5, "tolerance": 0.1}, "1,234.5") == 1.0


def test_euler_distance_tolerance_scales_with_the_triangle():
    from experiments.geometry.logic import TriangleExperiment
    model = TriangleExperiment({"vertices": [[0, 0], [1, 0], [0.45, 0.8]],
                                "constructions": ["circumcircle", "centroid", "euler_line"]})
    model.rng = np.random.default_rng(0)
    model.initialize({})
    q = next(q for q in model.questions(model.get_results(), np.random.default_rng(0))
             if q["template"] == "euler_distance")
    assert q["tolerance"] < q["answer"]  # answering 0 must not be accepted
