# tests/test_export_rl.py
"""OpenEnv and verifiers exporters (simulator.export.rl).

The file-generation tests need only the simulator. Tests marked by
``pytest.importorskip`` run the generated packages against the real libraries
(openenv >= 0.12 with fastapi; verifiers >= 0.4 with its v1 API) when installed.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import sys

import pytest
import yaml

from simulator.engine import SimulatorEngine
from simulator.envs import ExperimentEnv, ExperimentTaskSpec
from simulator.export.rl import (
    DEFAULT_QA_SYSTEM_PROMPT,
    export_openenv,
    export_verifiers_qa,
    verifiers_openenv_taskset_config,
)
from simulator.export.tasks import collect_bundle_tasks, public_view

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CONFIG_PATH = os.path.join(ROOT, "examples", "example_7_reference_dataset", "config.yaml")

OPENENV_FILES = [
    "openenv.yaml", "models.py", "client.py", "__init__.py", "pyproject.toml", "README.md",
    ".dockerignore", "spec.json", "server/__init__.py", "server/app.py", "server/requirements.txt",
    "server/Dockerfile",
]
PLACEHOLDERS = ("__ENV_NAME__", "__ENV_TITLE__", "__PREFIX__", "__NAME__", "__TITLE__", "__UV_SOURCES__",
                "__SIMULATOR", "__OPENENV_REQ__", "__VERIFIERS_REQ__", "__VENDOR_TREE__")


def _base_config():
    with open(BASE_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _spec(**overrides):
    spec = {"base_config": _base_config(),
            "hidden": {"prey_growth_rate": [0.6, 1.6]},
            "controls": {"initial_prey": [1, 50], "initial_predators": [1, 20]},
            "target": "prey_growth_rate", "budget": 2, "tolerance": 0.05,
            "description": "Estimate the prey growth rate."}
    spec.update(overrides)
    return spec


def _files(path):
    out = []
    for dirpath, _, filenames in os.walk(path):
        out += [os.path.relpath(os.path.join(dirpath, n), path).replace(os.sep, "/") for n in filenames]
    return sorted(out)


def _toml(path):
    tomllib = pytest.importorskip("tomllib")
    with open(path, "rb") as f:
        return tomllib.load(f)


def _assert_python_compiles(path):
    for rel in _files(path):
        if rel.endswith(".py"):
            with open(os.path.join(path, rel), encoding="utf-8") as f:
                source = f.read()
            compile(source, rel, "exec")


def _assert_no_placeholders(path):
    for rel in _files(path):
        if rel.endswith((".png", ".jsonl")):
            continue
        with open(os.path.join(path, rel), encoding="utf-8") as f:
            text = f.read()
        left = [p for p in PLACEHOLDERS if p in text]
        assert not left, f"{rel} still contains {left}"


def _forget_modules(prefix):
    for key in [k for k in sys.modules if k == prefix or k.startswith(prefix + ".")]:
        del sys.modules[key]


@pytest.fixture(scope="module")
def bundle_runs(tmp_path_factory):
    """Two runs with figure bundles (Q&A answers computed from the exact data)."""
    out = tmp_path_factory.mktemp("bundle_runs")
    engine = SimulatorEngine(output_dir=str(out), verbose=False)
    for seed in (1, 2):
        config = dict(_base_config(), seed=seed, generate_plots=False, bundle=True)
        engine.run_config(config)
    return str(out)


# ============================================================================ OpenEnv: files
def test_openenv_package_layout_and_spec(tmp_path):
    path = export_openenv(_spec(), str(tmp_path))
    assert path == os.path.join(str(tmp_path), "sds_experiment_env")
    files = _files(path)
    for rel in OPENENV_FILES + ["server/sds_experiment_env_environment.py"]:
        assert rel in files, rel
    # The model (outside simulator/) is vendored with its parent package's __init__
    assert "vendor/experiments/__init__.py" in files
    assert "vendor/experiments/predator_prey/logic.py" in files
    assert not any("__pycache__" in f for f in files)

    with open(os.path.join(path, "spec.json"), encoding="utf-8") as f:
        written = json.load(f)
    assert "bundle" not in written["base_config"]
    expected = ExperimentTaskSpec.from_dict(_spec()).to_dict()
    expected["base_config"].pop("bundle")
    assert written == expected
    assert ExperimentTaskSpec.from_dict(written).to_dict() == written     # loadable as is

    _assert_python_compiles(path)
    _assert_no_placeholders(path)
    with open(os.path.join(path, "models.py"), encoding="utf-8") as f:
        models = f.read()
    for cls in ("SdsExperimentAction(Action)", "SdsExperimentObservation(Observation)", "SdsExperimentState(State)"):
        assert f"class {cls}" in models
    with open(os.path.join(path, "server", "sds_experiment_env_environment.py"), encoding="utf-8") as f:
        env_src = f.read()
    assert "class SdsExperimentEnvironment(Environment)" in env_src
    assert "from simulator.envs import ExperimentEnv" in env_src


def test_openenv_manifest_and_pyproject(tmp_path):
    path = export_openenv(_spec(), str(tmp_path), name="lv_env", max_concurrent_envs=4)
    with open(os.path.join(path, "openenv.yaml"), encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    assert manifest["name"] == "lv_env"
    assert manifest["app"] == "server.app:app"
    assert manifest["validation"]["reward"]["range"] == [0.0, 1.0]
    with open(os.path.join(path, "server", "app.py"), encoding="utf-8") as f:
        app_src = f.read()
    assert "max_concurrent_envs=4" in app_src and 'env_name="lv_env"' in app_src
    assert "def main(" in app_src and 'if __name__ == "__main__":' in app_src

    project = _toml(os.path.join(path, "pyproject.toml"))
    assert project["project"]["requires-python"] == ">=3.10"
    deps = project["project"]["dependencies"]
    assert any(d.startswith("openenv") for d in deps)
    assert "scientific-data-simulator" in deps
    assert project["project"]["scripts"]["server"] == "lv_env.server.app:main"
    assert project["tool"]["uv"]["sources"]["scientific-data-simulator"]["git"].startswith("https://github.com/")
    assert "spec.json" in project["tool"]["setuptools"]["package-data"]["lv_env"]


@pytest.mark.parametrize("source", [None, "local", "git+https://example.org/sds.git@v1.2"])
def test_openenv_simulator_source(tmp_path, source):
    if source == "local":
        source = ROOT
    path = export_openenv(_spec(), str(tmp_path), simulator_source=source, vendor_model=False)
    assert not os.path.exists(os.path.join(path, "vendor"))
    project = _toml(os.path.join(path, "pyproject.toml"))
    sources = project.get("tool", {}).get("uv", {}).get("sources", {})
    with open(os.path.join(path, "server", "requirements.txt"), encoding="utf-8") as f:
        requirements = f.read()
    if source is None:
        assert sources == {}
        assert "scientific-data-simulator\n" in requirements
    elif source == ROOT:
        assert sources["scientific-data-simulator"] == {"path": ROOT, "editable": True}
        assert f"-e {ROOT}" in requirements
    else:
        assert sources["scientific-data-simulator"] == {"git": "https://example.org/sds.git", "rev": "v1.2"}


def test_openenv_spec_json_drives_experiment_env(tmp_path):
    path = export_openenv(_spec(), str(tmp_path), check=True)
    with open(os.path.join(path, "spec.json"), encoding="utf-8") as f:
        env = ExperimentEnv(json.load(f))
    first = env.reset(seed=4)
    assert first["target"] == "prey_growth_rate" and first["budget"] == 2
    result = env.step({"type": "experiment", "inputs": {"initial_prey": 20, "initial_predators": 3}})
    assert result["observation"]["measurements"] and result["reward"] is None
    result = env.step({"type": "answer", "value": env.truth["prey_growth_rate"]})
    assert result["done"] and result["reward"] == 1.0


def test_openenv_spec_from_yaml_file_and_errors(tmp_path):
    spec = _spec()
    # relative to the spec file (copied next to it: on Windows the repo and the temp folder can sit
    # on different drives, where no relative path exists)
    (tmp_path / "configs").mkdir()
    shutil.copy(BASE_CONFIG_PATH, tmp_path / "configs" / "base.yaml")
    spec["base_config"] = "configs/base.yaml"
    spec_file = tmp_path / "env.yaml"
    spec_file.write_text(yaml.safe_dump(spec), encoding="utf-8")
    path = export_openenv(str(spec_file), str(tmp_path / "out"))
    with open(os.path.join(path, "spec.json"), encoding="utf-8") as f:
        assert json.load(f)["base_config"]["n_steps"] == 3000

    with pytest.raises(FileExistsError):
        export_openenv(_spec(), str(tmp_path / "out"))
    assert export_openenv(_spec(budget=7), str(tmp_path / "out"), overwrite=True) == path
    with open(os.path.join(path, "spec.json"), encoding="utf-8") as f:
        assert json.load(f)["budget"] == 7
    with pytest.raises(ValueError):
        export_openenv(_spec(), str(tmp_path), name="Bad-Name")
    with pytest.raises(ValueError):
        export_openenv(_spec(target="prey_death_rate"), str(tmp_path), name="x_env")


def test_openenv_categorical_target(tmp_path):
    spec = _spec(hidden={"integrator": {"choices": ["euler", "rk4"]}}, target="integrator")
    path = export_openenv(spec, str(tmp_path))
    with open(os.path.join(path, "README.md"), encoding="utf-8") as f:
        assert "one of euler, rk4" in f.read()


# ============================================================================ verifiers: files
def test_verifiers_package_layout(bundle_runs, tmp_path):
    tasks = collect_bundle_tasks(bundle_runs)
    assert len(tasks) > 10
    path = export_verifiers_qa(tasks, str(tmp_path), name="lv_qa")
    files = _files(path)
    for rel in ("pyproject.toml", "README.md", "lv_qa/__init__.py", "lv_qa/taskset.py", "lv_qa/data/tasks.jsonl"):
        assert rel in files, rel
    figures = [f for f in files if f.startswith("lv_qa/data/figures/")]
    assert len(figures) == 2                                        # one per run, deduplicated

    with open(os.path.join(path, "lv_qa", "data", "tasks.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert [r["id"] for r in rows] == [t["id"] for t in tasks]
    for row, task in zip(rows, tasks):
        assert "image" not in row
        assert row["answer"] == task["answer"] and row["answer_type"] == task["answer_type"]
        assert row["figure"].startswith("figures/")
        assert os.path.isfile(os.path.join(path, "lv_qa", "data", row["figure"]))

    _assert_python_compiles(path)
    _assert_no_placeholders(path)
    with open(os.path.join(path, "lv_qa", "taskset.py"), encoding="utf-8") as f:
        src = f.read()
    assert "class LvQaTaskset(vf.Taskset[LvQaTask, LvQaConfig])" in src
    assert "from simulator.export.tasks import score" in src
    assert json.dumps(DEFAULT_QA_SYSTEM_PROMPT) in src

    project = _toml(os.path.join(path, "pyproject.toml"))
    assert project["project"]["name"] == "lv-qa"
    assert project["project"]["requires-python"] == ">=3.11"
    deps = project["project"]["dependencies"]
    assert any(d.startswith("verifiers") for d in deps) and "scientific-data-simulator" in deps


def test_verifiers_export_options_and_errors(bundle_runs, tmp_path):
    path = export_verifiers_qa(bundle_runs, str(tmp_path), copy_figures=False, system_prompt="Be brief.")
    with open(os.path.join(path, "sds_bundle_qa", "data", "tasks.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert all(os.path.isabs(r["figure"]) and os.path.isfile(r["figure"]) for r in rows)
    assert not os.listdir(os.path.join(path, "sds_bundle_qa", "data", "figures"))
    with open(os.path.join(path, "sds_bundle_qa", "taskset.py"), encoding="utf-8") as f:
        assert 'SYSTEM_PROMPT = "Be brief."' in f.read()

    tasks = collect_bundle_tasks(bundle_runs)
    with pytest.raises(ValueError, match="answer"):
        export_verifiers_qa([public_view(t) for t in tasks], str(tmp_path), name="no_answers")
    with pytest.raises(ValueError):
        export_verifiers_qa([], str(tmp_path), name="empty")
    with pytest.raises(ValueError):
        export_verifiers_qa(tasks + tasks[:1], str(tmp_path), name="dupes")


def test_verifiers_openenv_taskset_config():
    config = verifiers_openenv_taskset_config(base_url="http://localhost:8000", seeds=range(3))
    assert config["id"] == "openenv" and config["base_url"] == "http://localhost:8000"
    assert config["resets"] == [{"seed": 0}, {"seed": 1}, {"seed": 2}]
    with pytest.raises(ValueError):
        verifiers_openenv_taskset_config()
    with pytest.raises(ValueError):
        verifiers_openenv_taskset_config(env="org/space", seeds=[])


# ============================================================================ against the libraries
def _version_tuple(version):
    return tuple(int(p) for p in version.split(".")[:2] if p.isdigit())


def test_openenv_environment_and_app(tmp_path, monkeypatch):
    pytest.importorskip("openenv.core.env_server.http_server")
    testclient = pytest.importorskip("fastapi.testclient")
    from importlib.metadata import version
    if _version_tuple(version("openenv")) < (0, 12):     # the generated package requires openenv>=0.12
        pytest.skip(f"openenv {version('openenv')} < 0.12")
    name = "sds_rl_test_env"
    export_openenv(_spec(), str(tmp_path), name=name)
    monkeypatch.syspath_prepend(str(tmp_path))
    _forget_modules(name)
    try:
        envmod = importlib.import_module(f"{name}.server.{name}_environment")
        models = importlib.import_module(f"{name}.models")
        Action = models.SdsRlTestAction
        env = envmod.SdsRlTestEnvironment()
        obs = env.reset(seed=4)
        assert obs.target == "prey_growth_rate" and obs.budget == 2 and not obs.done and obs.task
        obs = env.step(Action(type="experiment", inputs={"initial_prey": 20, "initial_predators": 3}))
        assert obs.experiment == 1 and obs.budget_left == 1 and obs.measurements and obs.reward is None
        obs = env.step(Action(type="experiment", inputs={"initial_prey": 999}))
        assert "outside" in obs.error and not obs.done
        assert env.state.experiments_used == 1 and not hasattr(env.state, "episode_seed")
        obs = env.step(Action(type="answer", value=1e6))
        assert obs.done and obs.reward == 0.0 and obs.correct is False
        truth = obs.metadata["truth"]["prey_growth_rate"]
        env.reset(seed=4)
        obs = env.step(Action(type="answer", value=truth))
        assert obs.done and obs.reward == 1.0 and obs.correct is True

        app = importlib.import_module(f"{name}.server.app").app
        client = testclient.TestClient(app)
        response = client.post("/reset", json={"seed": 4})
        assert response.status_code == 200
        body = response.json()
        assert body["done"] is False and body["observation"]["target"] == "prey_growth_rate"
        # plain HTTP: every request gets a fresh environment -> no episode in progress
        response = client.post("/step", json={"action": {"type": "experiment", "inputs": {"initial_prey": 5}}})
        assert response.status_code == 200 and response.json()["observation"]["error"]
        assert client.post("/step", json={"action": {"type": "dance"}}).status_code == 422
        response = client.get("/state")
        assert response.status_code == 200 and "experiments_used" in response.json()
        schema = client.get("/schema").json()
        assert set(schema["action"]["properties"]) >= {"type", "inputs", "value"}
        with client.websocket_connect("/ws") as ws:                     # a real multi-step episode
            ws.send_json({"type": "reset", "data": {"seed": 4}})
            assert ws.receive_json()["data"]["observation"]["budget"] == 2
            ws.send_json({"type": "step", "data": {"type": "experiment", "inputs": {"initial_prey": 20}}})
            assert ws.receive_json()["data"]["observation"]["budget_left"] == 1
            ws.send_json({"type": "step", "data": {"type": "answer", "value": truth}})
            message = ws.receive_json()["data"]
            assert message["done"] is True and message["reward"] == 1.0
    finally:
        _forget_modules(name)


def test_verifiers_taskset_rewards(bundle_runs, tmp_path, monkeypatch):
    vf = pytest.importorskip("verifiers.v1")
    from verifiers.v1.graph import MessageNode
    from verifiers.v1.utils.loaders import load_taskset, taskset_config_type

    name = "sds_rl_test_qa"
    path = export_verifiers_qa(bundle_runs, str(tmp_path), name=name)
    monkeypatch.syspath_prepend(path)
    _forget_modules(name)
    try:
        tasks = list(load_taskset(taskset_config_type(name).model_validate({"id": name})))
        assert len(tasks) == len(collect_bundle_tasks(bundle_runs))
        first = tasks[0]
        assert first.key == first.data.id and first.data.idx == 0
        content = first.data.prompt[0].content                          # multimodal by default
        assert content[1].image_url.url.startswith("data:image/png;base64,")
        text_only = list(load_taskset(taskset_config_type(name).model_validate({"id": name, "multimodal": False})))
        assert isinstance(text_only[0].data.prompt, str) and "attached" not in text_only[0].data.prompt

        def reward(task, reply):
            trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key,
                                               hash=task.hash),
                             agent=vf.AgentInfo(config=vf.AgentConfig()))
            trace.nodes.append(MessageNode(message={"role": "assistant", "content": reply}, sampled=True))
            asyncio.run(task.score(trace))
            return trace.rewards["correct"].score

        for kind in ("number", "integer", "choice"):
            task = next(t for t in tasks if t.data.answer_type == kind)
            d = task.data
            if kind == "number":
                right, wrong = f"Reading the curve.\n{d.answer:.6g}", f"Reading the curve.\n{d.answer + 3 * d.tolerance + 1:.6g}"
            elif kind == "integer":
                right, wrong = f"Counting.\n\\boxed{{{d.answer}}}", f"Counting.\n{d.answer + 1}"
            else:
                other = next(c for c in d.choices if c != d.answer)
                right, wrong = f"Both rise; {other} less.\n{d.answer}", f"Both rise.\n{other}"
            assert reward(task, right) == 1.0, (kind, right)
            assert reward(task, wrong) == 0.0, (kind, wrong)
    finally:
        _forget_modules(name)
