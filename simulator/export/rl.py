# simulator/export/rl.py
"""Reinforcement-learning environment packages: OpenEnv and verifiers.

Two exporters write self-contained packages that RL frameworks install and load.
Writing a package needs only the simulator; *using* it needs the target library.

* :func:`export_openenv` - an `OpenEnv <https://github.com/meta-pytorch/OpenEnv>`_
  environment (layout of the ``openenv init`` scaffold, openenv 0.12) that serves
  :class:`simulator.envs.ExperimentEnv` - the multi-turn "run experiments on a
  hidden system, then answer" task - over HTTP/WebSocket. The task specification
  is written to ``spec.json`` inside the package; the model code named by
  ``experiment_type`` is copied to ``vendor/`` unless it lives in ``simulator``.
* :func:`export_verifiers_qa` - a `verifiers <https://github.com/PrimeIntellect-ai/verifiers>`_
  (v1 API, verifiers 0.4) taskset of single-turn questions from figure bundles
  (:func:`simulator.export.tasks.collect_bundle_tasks`). Tasks, answers and the
  figures are copied into the package; the reward is
  :func:`simulator.export.tasks.score` applied to the model's final answer.
  Prompts are multimodal by default (caption + question text and the figure as a
  base64 PNG data URL); ``--env.taskset.multimodal false`` gives text-only prompts.

Multi-turn training with verifiers
----------------------------------
verifiers ships an ``openenv`` taskset (``pip install 'verifiers[openenv]'``) that
drives any OpenEnv server: each model reply is parsed as a JSON action, the next
observation is sent back as the user turn, and the per-step rewards are summed
into ``openenv_reward``. To train on the package from :func:`export_openenv`,
start its server (``uv run --project <package> server`` or ``uvicorn server.app:app``
inside it) and point the taskset at it, one task per reset seed::

    vf-run openenv --env.taskset.base-url http://localhost:8000 \\
        --env.taskset.resets '[{"seed": 0}, {"seed": 1}, {"seed": 2}]' \\
        --env.agent.harness.id null -m <model>

:func:`verifiers_openenv_taskset_config` returns that taskset configuration as a
dict (e.g. for a TOML run file).

Example::

    from simulator.export.rl import export_openenv, export_verifiers_qa
    from simulator.export.tasks import collect_bundle_tasks

    export_openenv({"base_config": "config.yaml", "hidden": {"prey_growth_rate": [0.6, 1.6]},
                    "controls": {"initial_prey": [1, 50]}, "target": "prey_growth_rate"}, "out/")
    export_verifiers_qa(collect_bundle_tasks("runs/"), "out/", name="lv_figure_qa")
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import keyword
import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import yaml

from ..config import yaml_load
from ..envs import ExperimentEnv, ExperimentTaskSpec
from .tasks import collect_bundle_tasks

DEFAULT_SIMULATOR_SOURCE = "git+https://github.com/sandner-art/scientific-data-simulator.git"
OPENENV_REQUIREMENT = "openenv>=0.12"
VERIFIERS_REQUIREMENT = "verifiers>=0.4.1"
SIMULATOR_DISTRIBUTION = "scientific-data-simulator"

SpecLike = Union[ExperimentTaskSpec, Dict[str, Any], str, "os.PathLike[str]"]


# ============================================================================ helpers
def _check_name(name: str) -> str:
    if not (isinstance(name, str) and re.fullmatch(r"[a-z_][a-z0-9_]*", name) and not keyword.iskeyword(name)):
        raise ValueError(f"name must be a snake_case Python identifier (e.g. 'my_env'), got {name!r}")
    return name


def _env_prefix(name: str) -> str:
    """Class-name prefix as ``openenv init`` derives it ('sds_experiment_env' -> 'SdsExperiment')."""
    base = name[:-4] if name.endswith("_env") else name
    base = base or name
    return "".join(part.capitalize() for part in base.split("_") if part) or "Sds"


def _pascal(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_") if part) or "Sds"


def _title(name: str) -> str:
    return " ".join(part.capitalize() for part in name.split("_") if part)


def _prepare_dir(output_dir: str, name: str, overwrite: bool) -> str:
    path = os.path.join(os.fspath(output_dir), name)
    if os.path.isfile(path):
        raise FileExistsError(f"'{path}' exists and is a file")
    if os.path.isdir(path) and os.listdir(path):
        if not overwrite:
            raise FileExistsError(f"'{path}' exists and is not empty (pass overwrite=True to replace it)")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _fill(template: str, replacements: Dict[str, str]) -> str:
    for key in sorted(replacements, key=len, reverse=True):
        template = template.replace(key, replacements[key])
    return template


def _toml_str(value: str) -> str:
    return json.dumps(value)  # a JSON string is a valid TOML basic string


def _simulator_source(source: Optional[str]) -> Dict[str, str]:
    """Requirement lines for the simulator, which is not on PyPI.

    ``source`` is a ``git+URL[@rev]`` or a local path; ``None`` leaves the
    dependency unpinned (install the simulator yourself first).
    """
    if not source:
        return {"uv_source": "", "requirement": SIMULATOR_DISTRIBUTION}
    source = os.fspath(source)
    if source.startswith("git+"):
        url, rev = source[4:], None
        match = re.match(r"^(.*\.git|[^@]*?)@([^/@]+)$", url)
        if match and not url.startswith("git@"):
            url, rev = match.group(1), match.group(2)
        entry = "{ git = " + _toml_str(url) + (", rev = " + _toml_str(rev) if rev else "") + " }"
        return {"uv_source": f"{SIMULATOR_DISTRIBUTION} = {entry}",
                "requirement": f"{SIMULATOR_DISTRIBUTION} @ {source}"}
    path = os.path.abspath(source)
    return {"uv_source": f"{SIMULATOR_DISTRIBUTION} = {{ path = {_toml_str(path)}, editable = true }}",
            "requirement": f"-e {path}"}


def _uv_sources_block(source: Dict[str, str]) -> str:
    if not source["uv_source"]:
        return ""
    return ("\n[tool.uv.sources]\n"
            "# The simulator is not on PyPI; uv (and `uv sync` in the Dockerfile) installs it from here.\n"
            f"{source['uv_source']}\n")


def _load_spec(spec: SpecLike) -> ExperimentTaskSpec:
    """An :class:`ExperimentTaskSpec` from a spec object, dict, or YAML/JSON file."""
    if isinstance(spec, ExperimentTaskSpec):
        data = copy.deepcopy(spec.to_dict())
    elif isinstance(spec, dict):
        data = copy.deepcopy(spec)
    elif isinstance(spec, (str, os.PathLike)):
        path = os.fspath(spec)
        with open(path, encoding="utf-8") as f:
            data = yaml_load(f)  # YAML is a superset of JSON; reads 1e-3 as a float
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a mapping")
        base = data.get("base_config")
        if isinstance(base, str) and not os.path.isabs(base) and not os.path.exists(base):
            candidate = os.path.join(os.path.dirname(os.path.abspath(path)), base)
            if os.path.exists(candidate):
                data["base_config"] = candidate
    else:
        raise TypeError(f"spec must be an ExperimentTaskSpec, a dict or a path, got {type(spec).__name__}")
    task = ExperimentTaskSpec.from_dict(data)       # loads a base_config path, validates
    out = task.to_dict()
    out["base_config"] = copy.deepcopy(out["base_config"])
    out["base_config"].pop("bundle", None)          # environments never write bundles
    return ExperimentTaskSpec.from_dict(out)


def _vendor_model(experiment_type: str, vendor_dir: str) -> List[str]:
    """Copy the package that defines ``experiment_type`` to ``vendor_dir``.

    Returns the copied paths relative to ``vendor_dir``; nothing is copied for
    models inside :mod:`simulator` (installed with the simulator).
    """
    module_name = experiment_type.rpartition(".")[0]
    if not module_name or module_name.split(".")[0] == "simulator":
        return []
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        raise ImportError(f"Cannot locate the source of '{module_name}' (needed to vendor the model); "
                          "pass vendor_model=False if the model is installed where the environment runs")
    origin = os.path.abspath(spec.origin)
    parts = module_name.split(".")
    is_package = os.path.basename(origin) == "__init__.py"
    package_parts = parts if is_package else parts[:-1]
    copied: List[str] = []
    if not package_parts:                           # a top-level single-file module
        shutil.copy2(origin, os.path.join(vendor_dir, os.path.basename(origin)))
        return [os.path.basename(origin)]
    leaf_dir = os.path.dirname(origin)
    root = leaf_dir                                 # the directory containing the top-level package
    for _ in range(len(package_parts)):
        root = os.path.dirname(root)
    for depth in range(1, len(package_parts)):      # ancestor packages: their __init__.py only
        rel = os.path.join(*package_parts[:depth])
        init = os.path.join(root, rel, "__init__.py")
        os.makedirs(os.path.join(vendor_dir, rel), exist_ok=True)
        if os.path.exists(init):
            shutil.copy2(init, os.path.join(vendor_dir, rel, "__init__.py"))
            copied.append(os.path.join(rel, "__init__.py"))
    rel_leaf = os.path.join(*package_parts)
    shutil.copytree(leaf_dir, os.path.join(vendor_dir, rel_leaf),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    for dirpath, _, filenames in os.walk(os.path.join(vendor_dir, rel_leaf)):
        copied += [os.path.relpath(os.path.join(dirpath, n), vendor_dir) for n in sorted(filenames)]
    return sorted(p.replace(os.sep, "/") for p in copied)


# ============================================================================ OpenEnv templates
_OE_OPENENV_YAML = """\
spec_version: 1
name: __ENV_NAME__
version: 0.1.0
type: space
runtime: fastapi
app: server.app:app
port: 8000
validation:
  reward:
    range: [0.0, 1.0]
    oracle_tolerance: 0.0
    floor_margin: 0.1
  resources:
    cpu: 1.0
    memory_mb: 1024
    disk_mb: 512
    episode_timeout_s: __EPISODE_TIMEOUT__
  capabilities:
    verifier:
      kind: reward_channel
  types:
    tags: [science, simulation]
"""

_OE_MODELS = '''\
# Generated by simulator.export.rl.export_openenv.

"""
Data models for the __ENV_TITLE__ Environment.

An agent runs a limited number of noisy experiments on a simulated system with
hidden parameters, then answers a question about it (see spec.json).
"""

from typing import Any, Dict, List, Literal, Optional, Union

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class __PREFIX__Action(Action):
    """Run an experiment (`type="experiment"`, `inputs`) or submit the answer (`type="answer"`, `value`)."""

    type: Literal["experiment", "answer"] = Field(
        ..., description='"experiment" to run one experiment, "answer" to submit the final answer'
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Experiment inputs (the observation's `controls`), each within [low, high]",
    )
    value: Optional[Union[float, str]] = Field(
        default=None, description="The answer: a number, or one of the listed choices"
    )


class __PREFIX__Observation(Observation):
    """What the agent sees. `task`/`controls`/`budget`/`target` come with reset; `measurements` with each experiment."""

    task: Optional[str] = Field(default=None, description="Task description (after reset)")
    controls: Dict[str, Any] = Field(default_factory=dict, description="Settable inputs and their bounds or choices")
    budget: Optional[int] = Field(default=None, description="Number of experiments allowed")
    target: Optional[str] = Field(default=None, description="Name of the quantity to report")
    experiment: Optional[int] = Field(default=None, description="1-based number of the experiment just run")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Inputs of the experiment just run")
    measurements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Observed outputs: {name: {units, values, <x axis>: [...]}}",
    )
    budget_left: Optional[int] = Field(default=None, description="Experiments still allowed")
    answer: Optional[Union[float, str]] = Field(default=None, description="The submitted answer")
    correct: Optional[bool] = Field(default=None, description="Whether the submitted answer is correct")
    error: Optional[str] = Field(default=None, description="Why the last action was rejected")
    data: Dict[str, Any] = Field(default_factory=dict, description="Any further observation fields")


class __PREFIX__State(State):
    """Episode bookkeeping; the hidden values are never part of the state."""

    experiments_used: int = Field(default=0, ge=0)
    budget_left: int = Field(default=0, ge=0)
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Inputs and seed of each experiment")
    done: bool = Field(default=True, description="Whether no episode is in progress")
'''

_OE_ENVIRONMENT = '''\
# Generated by simulator.export.rl.export_openenv.

"""
__ENV_TITLE__ Environment Implementation.

Serves simulator.envs.ExperimentEnv: each episode draws hidden model parameters
from the priors in spec.json (from the reset seed); the agent runs up to
`budget` noisy experiments and then submits an answer, rewarded 1.0 if correct
and 0.0 otherwise (also 0.0 when the budget is exhausted without an answer).
Experiments and invalid actions are not rewarded (reward None).

The spec path can be overridden with the SDS_ENV_SPEC environment variable.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..models import __PREFIX__Action, __PREFIX__Observation, __PREFIX__State
except ImportError:
    from models import __PREFIX__Action, __PREFIX__Observation, __PREFIX__State

PACKAGE_DIR = Path(__file__).resolve().parent.parent
SPEC_PATH = PACKAGE_DIR / "spec.json"
VENDOR_DIR = PACKAGE_DIR / "vendor"

# The model code (experiment_type) copied at export time takes precedence, so the
# environment simulates exactly the model the spec was written for.
if VENDOR_DIR.is_dir() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from simulator.envs import ExperimentEnv  # noqa: E402

_TYPED_FIELDS = tuple(
    name for name in __PREFIX__Observation.model_fields if name not in ("done", "reward", "metadata", "data")
)


def load_spec(path: Optional[str] = None) -> Dict[str, Any]:
    """The task specification (ExperimentTaskSpec as a dict)."""
    path = path or os.environ.get("SDS_ENV_SPEC") or SPEC_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_observation(result: Dict[str, Any]) -> __PREFIX__Observation:
    """An ExperimentEnv step result as a typed observation; `info` (incl. the truth,
    revealed at the end of an episode) goes to `metadata`."""
    obs = dict(result.get("observation") or {})
    done = bool(obs.pop("done", result.get("done", False)))
    typed = {name: obs.pop(name) for name in _TYPED_FIELDS if name in obs}
    return __PREFIX__Observation(
        **typed, data=obs, done=done, reward=result.get("reward"), metadata=dict(result.get("info") or {})
    )


class __PREFIX__Environment(Environment):
    """
    Experiment-design environment over a simulated system with hidden parameters.

    Example:
        >>> env = __PREFIX__Environment()
        >>> obs = env.reset(seed=1)
        >>> print(obs.task)
        >>> obs = env.step(__PREFIX__Action(type="experiment", inputs={...}))
        >>> print(obs.measurements)
        >>> obs = env.step(__PREFIX__Action(type="answer", value=1.0))
        >>> print(obs.correct, obs.reward)
    """

    # Every instance owns its ExperimentEnv; nothing is shared between sessions.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, spec: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._spec = spec if spec is not None else load_spec()
        self._env = ExperimentEnv(self._spec)
        self._state = __PREFIX__State(episode_id=str(uuid4()), step_count=0,
                                      budget_left=int(self._env.spec.budget))
        self._active = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> __PREFIX__Observation:
        """Start an episode; the seed fixes the hidden values and all measurement noise."""
        observation = self._env.reset(seed=seed)
        self._active = True
        self._sync_state(episode_id if episode_id is not None else str(uuid4()))
        return to_observation({"observation": observation, "reward": None, "done": False})

    def step(  # type: ignore[override]
        self,
        action: __PREFIX__Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> __PREFIX__Observation:
        """Run an experiment or submit the answer."""
        if not self._active:
            return __PREFIX__Observation(
                error="No episode in progress: call reset() first. Plain HTTP /step requests each get a "
                      "fresh environment; use a WebSocket session (the client) for multi-step episodes.",
                done=True,
                reward=None,
            )
        result = self._env.step({"type": action.type, "inputs": dict(action.inputs), "value": action.value})
        if result["done"]:
            self._active = False
        self._sync_state(self._state.episode_id)
        return to_observation(result)

    @property
    def state(self) -> __PREFIX__State:
        """Episode id, step count, experiments used and their inputs/seeds."""
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        readme = PACKAGE_DIR / "README.md"
        return EnvironmentMetadata(
            name="__ENV_NAME__",
            description=self._env.spec.description or "Experiment-design environment (scientific-data-simulator)",
            readme_content=readme.read_text(encoding="utf-8") if readme.exists() else None,
            version="0.1.0",
        )

    def _sync_state(self, episode_id: Optional[str]) -> None:
        inner = self._env.state
        self._state = __PREFIX__State(
            episode_id=episode_id,
            step_count=inner["step_count"],
            experiments_used=inner["experiments_used"],
            budget_left=max(int(inner["budget_left"]), 0),
            history=inner["history"],
            done=inner["done"],
        )


if __name__ == "__main__":
    env = __PREFIX__Environment()
    first = env.reset(seed=0)
    print(first.task)
    inputs = {}
    for name, bounds in first.controls.items():
        inputs[name] = bounds["choices"][0] if isinstance(bounds, dict) else (float(bounds[0]) + float(bounds[1])) / 2
    print(env.step(__PREFIX__Action(type="experiment", inputs=inputs)).model_dump_json(indent=1)[:2000])
    final = env.step(__PREFIX__Action(type="answer", value=0.0))
    print("correct:", final.correct, "reward:", final.reward, "truth:", final.metadata.get("truth"))
'''

_OE_APP = '''\
# Generated by simulator.export.rl.export_openenv.

"""
FastAPI application for the __ENV_TITLE__ Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation/state schemas
    - WS /ws: WebSocket endpoint for persistent sessions (multi-step episodes)

Plain HTTP /reset and /step each use a fresh environment instance, so an episode
of several steps needs the WebSocket session (the client in client.py).

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    # or
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\\n    uv sync\\n'"
    ) from e

try:
    from ..models import __PREFIX__Action, __PREFIX__Observation, __PREFIX__State
    from .__ENV_NAME___environment import __PREFIX__Environment
except ImportError:
    from models import __PREFIX__Action, __PREFIX__Observation, __PREFIX__State
    from server.__ENV_NAME___environment import __PREFIX__Environment


app = create_app(
    __PREFIX__Environment,
    __PREFIX__Action,
    __PREFIX__Observation,
    env_name="__ENV_NAME__",
    max_concurrent_envs=__MAX_CONCURRENT__,  # concurrent WebSocket sessions
    state_cls=__PREFIX__State,
)


def main():
    """Entry point for `uv run --project . server` and `python -m __ENV_NAME__.server.app`."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
'''

_OE_SERVER_INIT = '''\
# Generated by simulator.export.rl.export_openenv.

"""__ENV_TITLE__ environment server components."""

from .__ENV_NAME___environment import __PREFIX__Environment

__all__ = ["__PREFIX__Environment"]
'''

_OE_CLIENT = '''\
# Generated by simulator.export.rl.export_openenv.

"""__ENV_TITLE__ Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import __PREFIX__Action, __PREFIX__Observation, __PREFIX__State


class __PREFIX__Env(EnvClient[__PREFIX__Action, __PREFIX__Observation, __PREFIX__State]):
    """
    Client for the __ENV_TITLE__ Environment (persistent WebSocket session).

    Example:
        >>> with __PREFIX__Env(base_url="http://localhost:8000").sync() as client:
        ...     result = client.reset(seed=1)
        ...     print(result.observation.task)
        ...     result = client.step(__PREFIX__Action(type="experiment", inputs={...}))
        ...     result = client.step(__PREFIX__Action(type="answer", value=1.0))
        ...     print(result.reward, result.observation.correct)
    """

    def _step_payload(self, action: __PREFIX__Action) -> Dict:
        return {"type": action.type, "inputs": action.inputs, "value": action.value}

    def _parse_result(self, payload: Dict) -> StepResult[__PREFIX__Observation]:
        obs_data = dict(payload.get("observation", {}))
        fields = {
            k: v for k, v in obs_data.items()
            if k in __PREFIX__Observation.model_fields and k not in ("done", "reward", "metadata")
        }
        observation = __PREFIX__Observation(
            **fields,
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})) or {},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> __PREFIX__State:
        return __PREFIX__State(**payload)
'''

_OE_INIT = '''\
# Generated by simulator.export.rl.export_openenv.

"""__ENV_TITLE__ Environment."""

from .client import __PREFIX__Env
from .models import __PREFIX__Action, __PREFIX__Observation, __PREFIX__State

__all__ = [
    "__PREFIX__Action",
    "__PREFIX__Observation",
    "__PREFIX__State",
    "__PREFIX__Env",
]
'''

_OE_PYPROJECT = """\
# Generated by simulator.export.rl.export_openenv.

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "openenv-__ENV_NAME__"
version = "0.1.0"
description = "__ENV_TITLE__ environment for OpenEnv (scientific-data-simulator)"
requires-python = ">=3.10"
dependencies = [
    # Core OpenEnv runtime (provides FastAPI server + HTTP client types)
    "__OPENENV_REQ__",
    # The simulator that runs the experiments (not on PyPI; see [tool.uv.sources] and README)
    "__SIMULATOR_DIST__",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]

[project.scripts]
# Server entry point - enables running via: uv run --project . server
# or: python -m __ENV_NAME__.server.app
server = "__ENV_NAME__.server.app:main"

[tool.setuptools]
include-package-data = true
packages = ["__ENV_NAME__", "__ENV_NAME__.server"]
package-dir = { "__ENV_NAME__" = ".", "__ENV_NAME__.server" = "server" }

[tool.setuptools.package-data]
"__ENV_NAME__" = ["spec.json", "openenv.yaml", "README.md", "vendor/**/*"]
__UV_SOURCES__"""

_OE_REQUIREMENTS = """\
__OPENENV_REQ__
fastapi>=0.116.0
uvicorn>=0.24.0
__SIMULATOR_REQ__
"""

_OE_DOCKERFILE = """\
# Generated by simulator.export.rl.export_openenv (from the openenv init template).

ARG BASE_IMAGE=ghcr.io/huggingface/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# git is needed to install the simulator from its repository ([tool.uv.sources])
RUN apt-get update && \\
    apt-get install -y --no-install-recommends git && \\
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=__ENV_NAME__

COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \\
        curl -LsSf https://astral.sh/uv/install.sh | sh && \\
        mv /root/.local/bin/uv /usr/local/bin/uv && \\
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \\
    fi

RUN --mount=type=cache,target=/root/.cache/uv \\
    if [ -f uv.lock ]; then \\
        uv sync --frozen --no-install-project --no-editable; \\
    else \\
        uv sync --no-install-project --no-editable; \\
    fi

RUN --mount=type=cache,target=/root/.cache/uv \\
    if [ -f uv.lock ]; then \\
        uv sync --frozen --no-editable; \\
    else \\
        uv sync --no-editable; \\
    fi

RUN mv /app/env/.venv /app/.venv

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/env /app/env
RUN ln -s /app/.venv /app/env/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
"""

_OE_DOCKERIGNORE = """\
.venv
.git
.gitignore
.env
__pycache__/
*.pyc
*.pyo
*.pyd
"""

_OE_README = """\
---
title: __ENV_TITLE__ Environment Server
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# __ENV_TITLE__ Environment

__DESCRIPTION__

Generated by `simulator.export.rl.export_openenv` from
[scientific-data-simulator](https://github.com/sandner-art/scientific-data-simulator).
The environment wraps `simulator.envs.ExperimentEnv`; the task is defined in
`spec.json` (model configuration, hidden parameters and their priors, controls,
target, budget, tolerance).

- Model: `__MODEL__`
- Hidden: __HIDDEN__ (drawn per episode from the reset seed)
- Controls: __CONTROLS__
- Target: `__TARGET__` (__TARGET_RULE__)
- Budget: __BUDGET__ experiments

## Install

The simulator is not on PyPI. With uv, `[tool.uv.sources]` in `pyproject.toml`
says where to get it__SOURCE_NOTE__. With pip, install it first, then this package:

```bash
pip install "scientific-data-simulator @ git+https://github.com/sandner-art/scientific-data-simulator.git"
# or, from a local checkout:
pip install -e /path/to/scientific-data-simulator
pip install -e .
```

## Quick start

```python
from __ENV_NAME__ import __PREFIX__Action, __PREFIX__Env

with __PREFIX__Env(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=1)
    print(result.observation.task)
    result = env.step(__PREFIX__Action(type="experiment", inputs=__EXAMPLE_INPUTS__))
    print(result.observation.measurements)
    result = env.step(__PREFIX__Action(type="answer", value=__EXAMPLE_ANSWER__))
    print(result.reward, result.observation.correct)
```

Run the server with `uv run --project . server` or `uvicorn server.app:app --port 8000`
(from this directory), or in Docker: `docker build -t __ENV_NAME__:latest -f server/Dockerfile .`

Without a server: `python server/__ENV_NAME___environment.py` runs one episode.

Plain HTTP `POST /reset` and `POST /step` each create a fresh environment
instance, so multi-step episodes need the WebSocket session (`/ws`, used by the client).

## Environment details

### Action (`__PREFIX__Action`)
- `type`: `"experiment"` or `"answer"`
- `inputs`: experiment inputs, each within its bounds (`controls`)
- `value`: the answer (a number, or one of the listed choices)

### Observation (`__PREFIX__Observation`)
- after reset: `task` (text), `controls`, `budget`, `target`
- after an experiment: `experiment`, `inputs`, `measurements` (`{output: {units, values, <x axis>: [...]}}`), `budget_left`
- after the answer: `answer`, `correct`; `metadata` then holds the truth and the error
- `error` when an action is rejected (unknown or out-of-range input); `data` holds any other fields

### Reward
1.0 for a correct answer, 0.0 for a wrong one or when the budget is exhausted
without an answer, `None` for experiments. Range [0, 1] (`openenv.yaml`).

### Reproducibility
The reset seed fixes the hidden values and the noise of every experiment.
`state` (`GET /state`) records the episode seed and each experiment's inputs and seed.

## Training with verifiers

The `openenv` taskset of verifiers (`pip install 'verifiers[openenv]'`) can drive a
running server, one task per reset seed:

```bash
vf-run openenv --env.taskset.base-url http://localhost:8000 \\
    --env.taskset.resets '[{"seed": 0}, {"seed": 1}]' --env.agent.harness.id null -m <model>
```

## Project structure

```
__ENV_NAME__/
├── .dockerignore
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── spec.json              # The task specification (ExperimentTaskSpec)
├── client.py              # __PREFIX__Env client
├── models.py              # Action, Observation and State models
__VENDOR_TREE__└── server/
    ├── __init__.py
    ├── __ENV_NAME___environment.py  # Environment (wraps simulator.envs.ExperimentEnv)
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    ├── requirements.txt
    └── Dockerfile
```
"""


def _example_inputs(spec: ExperimentTaskSpec) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}
    for name, bounds in spec.controls.items():
        if isinstance(bounds, dict):
            inputs[name] = bounds["choices"][0]
        else:
            inputs[name] = float(f"{(float(bounds[0]) + float(bounds[1])) / 2:.6g}")
    return inputs


def _example_answer(spec: ExperimentTaskSpec) -> Any:
    prior = spec.hidden[spec.target]
    if isinstance(prior, dict):
        return prior["choices"][0]
    return float(f"{(float(prior[0]) + float(prior[1])) / 2:.6g}")


# ============================================================================ OpenEnv export
def export_openenv(spec: SpecLike, output_dir: str, name: str = "sds_experiment_env", *,
                   simulator_source: Optional[str] = DEFAULT_SIMULATOR_SOURCE,
                   vendor_model: bool = True, max_concurrent_envs: int = 1,
                   episode_timeout_s: float = 300.0, check: bool = False,
                   overwrite: bool = False) -> str:
    """Write an OpenEnv environment package that serves :class:`~simulator.envs.ExperimentEnv`.

    The layout follows the ``openenv init`` scaffold (openenv 0.12): ``openenv.yaml``,
    ``models.py`` (Action/Observation/State), ``client.py``, ``server/app.py``
    (``create_app``), ``server/<name>_environment.py`` (the ``Environment``),
    ``server/Dockerfile``, ``pyproject.toml`` - plus ``spec.json`` and, for models
    outside :mod:`simulator`, the model package under ``vendor/``.

    Args:
        spec: An :class:`~simulator.envs.ExperimentTaskSpec`, its dict form, or a
            YAML/JSON file. ``base_config`` may be a config path; it is inlined, and
            its ``bundle`` key is dropped.
        output_dir: Directory in which the package directory ``name`` is created.
        name: Package name (snake_case); class names derive from it as in
            ``openenv init`` (``sds_experiment_env`` -> ``SdsExperimentEnvironment``).
        simulator_source: Where uv installs the simulator from: ``git+URL[@rev]`` or
            a local path; ``None`` leaves the dependency unpinned.
        vendor_model: Copy the package defining ``experiment_type`` into ``vendor/``.
        max_concurrent_envs: Concurrent WebSocket sessions of the server.
        episode_timeout_s: Declared in ``openenv.yaml``.
        check: Run one short episode (reset, one experiment, answer) before writing.
        overwrite: Replace an existing, non-empty package directory.

    Returns:
        The path of the package directory.
    """
    name = _check_name(name)
    task = _load_spec(spec)
    spec_dict = task.to_dict()
    try:
        spec_json = json.dumps(spec_dict, indent=2, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"The spec is not JSON-serializable: {e}") from e
    if check:
        env = ExperimentEnv(json.loads(spec_json))
        env.reset(seed=0)
        env.step({"type": "experiment", "inputs": _example_inputs(task)})
        env.step({"type": "answer", "value": _example_answer(task)})

    path = _prepare_dir(output_dir, name, overwrite)
    vendored: List[str] = []
    if vendor_model:
        vendor_dir = os.path.join(path, "vendor")
        os.makedirs(vendor_dir, exist_ok=True)
        vendored = _vendor_model(str(task.base_config.get("experiment_type", "")), vendor_dir)
        if not vendored:
            os.rmdir(vendor_dir)

    source = _simulator_source(simulator_source)
    model = str(task.base_config.get("experiment_type", "?"))
    if task.target_is_categorical():
        target_rule = "one of " + ", ".join(map(str, task.hidden[task.target]["choices"]))
    else:
        target_rule = f"correct within {float(task.tolerance):.0%} relative error"
    top_vendor = sorted({p.split("/")[0] for p in vendored})
    vendor_tree = ("├── vendor/              # Model code (" + ", ".join(top_vendor) + "), copied at export\n"
                   if vendored else "")
    source_note = ""
    if source["uv_source"]:
        source_note = f" (`{simulator_source}`)"
    r = {
        "__ENV_NAME__": name,
        "__ENV_TITLE__": _title(name),
        "__PREFIX__": _env_prefix(name),
        "__MAX_CONCURRENT__": str(int(max_concurrent_envs)),
        "__EPISODE_TIMEOUT__": repr(float(episode_timeout_s)),
        "__OPENENV_REQ__": OPENENV_REQUIREMENT,
        "__SIMULATOR_DIST__": SIMULATOR_DISTRIBUTION,
        "__SIMULATOR_REQ__": source["requirement"],
        "__UV_SOURCES__": _uv_sources_block(source),
        "__DESCRIPTION__": task.description or "Run experiments on a simulated system with hidden parameters, "
                                               "then report the target quantity.",
        "__MODEL__": model,
        "__HIDDEN__": ", ".join(f"`{k}`" for k in task.hidden),
        "__CONTROLS__": ", ".join(f"`{k}` {json.dumps(v)}" for k, v in task.controls.items()) or "none",
        "__TARGET__": task.target,
        "__TARGET_RULE__": target_rule,
        "__BUDGET__": str(int(task.budget)),
        "__SOURCE_NOTE__": source_note,
        "__EXAMPLE_INPUTS__": repr(_example_inputs(task)),
        "__EXAMPLE_ANSWER__": repr(_example_answer(task)),
        "__VENDOR_TREE__": vendor_tree,
    }
    files = {
        "openenv.yaml": _OE_OPENENV_YAML,
        "models.py": _OE_MODELS,
        "client.py": _OE_CLIENT,
        "__init__.py": _OE_INIT,
        "pyproject.toml": _OE_PYPROJECT,
        "README.md": _OE_README,
        ".dockerignore": _OE_DOCKERIGNORE,
        "server/__init__.py": _OE_SERVER_INIT,
        "server/app.py": _OE_APP,
        f"server/{name}_environment.py": _OE_ENVIRONMENT,
        "server/requirements.txt": _OE_REQUIREMENTS,
        "server/Dockerfile": _OE_DOCKERFILE,
    }
    for rel, template in files.items():
        _write(os.path.join(path, *rel.split("/")), _fill(template, r))
    _write(os.path.join(path, "spec.json"), spec_json + "\n")
    return path


# ============================================================================ verifiers templates
_VF_TASKSET = '''\
# Generated by simulator.export.rl.export_verifiers_qa.

"""__TITLE__: single-turn questions about simulated scientific figures.

Tasks come from figure bundles of scientific-data-simulator runs (data/tasks.jsonl,
figures in data/figures/). The reward `correct` is simulator.export.tasks.score
applied to the model's final answer: a number within the question's tolerance,
an exact integer, or the right choice.
"""

import base64
import json
import mimetypes
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from simulator.export.tasks import score
from verifiers.v1.harnesses.null import NullHarness

DATA_DIR = Path(__file__).resolve().parent / "data"
TASKS_FILE = DATA_DIR / "tasks.jsonl"
SYSTEM_PROMPT = __SYSTEM_PROMPT__
_ATTACHED = "Use the attached figure."


def final_answer(reply: str) -> str:
    """The part of a reply that is graded: the last \\\\boxed{...}, else the last non-empty line."""
    boxed = re.findall(r"\\\\boxed\\{([^{}]*)\\}", reply or "")
    if boxed:
        return boxed[-1].strip()
    lines = [line.strip() for line in (reply or "").splitlines() if line.strip()]
    return lines[-1].replace("**", "").strip() if lines else ""


class __PREFIX__Config(vf.TasksetConfig):
    tasks_file: Path | None = None
    """JSONL file of tasks (default: the packaged data/tasks.jsonl)."""
    multimodal: bool = True
    """Attach the figure to the prompt (as a base64 data URL); false gives text-only prompts."""
    answerable_from: list[str] | None = None
    """Keep only questions answerable from these sources (e.g. ["figure"])."""


class __PREFIX__Data(vf.TaskData):
    question: str
    answer: float | int | str
    answer_type: str
    tolerance: float | None = None
    choices: list[str] | None = None
    figure: str | None = None
    """Path of the figure, relative to data/ (or absolute)."""
    table: str | None = None
    info: dict[str, Any] = {}
    """experiment_id, seed, model, fidelity, template, answerable_from of the source question."""

    def grading_item(self) -> dict[str, Any]:
        return {"answer": self.answer, "answer_type": self.answer_type,
                "tolerance": self.tolerance, "choices": self.choices}


class __PREFIX__Task(vf.Task[__PREFIX__Data, vf.State, vf.TaskConfig]):
    @property
    def key(self) -> str:
        return self.data.id or self.hash

    @vf.reward
    async def correct(self, trace) -> float:
        return score(self.data.grading_item(), final_answer(trace.last_reply))


def figure_path(figure: str | None, base: Path = DATA_DIR) -> Path | None:
    if not figure:
        return None
    path = Path(figure)
    return path if path.is_absolute() else base / path


def build_prompt(row: dict[str, Any], multimodal: bool, base: Path = DATA_DIR):
    figure = figure_path(row.get("figure"), base)
    if multimodal and figure is not None and figure.exists():
        mime = mimetypes.guess_type(figure.name)[0] or "image/png"
        url = f"data:{mime};base64," + base64.b64encode(figure.read_bytes()).decode()
        return [{"role": "user", "content": [
            {"type": "text", "text": row["prompt"]},
            {"type": "image_url", "image_url": {"url": url}},
        ]}]
    return row["prompt"].replace(_ATTACHED + "\\n\\n", "").replace(_ATTACHED, "")


class __PREFIX__Taskset(vf.Taskset[__PREFIX__Task, __PREFIX__Config]):
    def load(self) -> Iterator[__PREFIX__Task]:
        path = Path(self.config.tasks_file or TASKS_FILE)
        base = path.resolve().parent
        wanted = self.config.answerable_from
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for row in rows:
            if wanted and row.get("answerable_from") not in wanted:
                continue
            yield __PREFIX__Task(
                __PREFIX__Data(
                    id=row["id"],
                    name=row["id"],
                    prompt=build_prompt(row, self.config.multimodal, base),
                    system_prompt=SYSTEM_PROMPT,
                    question=row["question"],
                    answer=row["answer"],
                    answer_type=row["answer_type"],
                    tolerance=row.get("tolerance"),
                    choices=row.get("choices"),
                    figure=row.get("figure"),
                    table=row.get("table"),
                    info={k: row.get(k) for k in
                          ("experiment_id", "seed", "model", "fidelity", "template", "answerable_from")},
                ),
                self.config.task,
            )
'''

_VF_INIT = '''\
# Generated by simulator.export.rl.export_verifiers_qa.

"""__TITLE__ taskset (verifiers v1). Run with: vf-run __NAME__ -m <model>"""

from .taskset import (
    NullHarness,
    __PREFIX__Config,
    __PREFIX__Data,
    __PREFIX__Task,
    __PREFIX__Taskset,
    final_answer,
)

# The taskset and its default harness (plain chat, no tools; override with
# --env.agent.harness.id <id>).
__all__ = [
    "NullHarness",
    "__PREFIX__Config",
    "__PREFIX__Data",
    "__PREFIX__Task",
    "__PREFIX__Taskset",
    "final_answer",
]
'''

_VF_PYPROJECT = """\
# Generated by simulator.export.rl.export_verifiers_qa.

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "__DIST_NAME__"
version = "0.1.0"
description = "__TITLE__: figure question answering from scientific-data-simulator bundles (verifiers taskset)"
requires-python = ">=3.11"
dependencies = [
    "__VERIFIERS_REQ__",
    # Grading (simulator.export.tasks.score); not on PyPI, see [tool.uv.sources] and README
    "__SIMULATOR_DIST__",
]

[tool.setuptools]
packages = ["__NAME__"]

[tool.setuptools.package-data]
"__NAME__" = ["data/*.jsonl", "data/figures/*"]
__UV_SOURCES__"""

_VF_README = """\
# __TITLE__

A [verifiers](https://github.com/PrimeIntellect-ai/verifiers) taskset (v1 API) of
__N_TASKS__ single-turn questions about __N_FIGURES__ figure(s) generated with
[scientific-data-simulator](https://github.com/sandner-art/scientific-data-simulator)
(`simulator.export.rl.export_verifiers_qa`).

Each question has a verifiable answer computed from the exact simulated data:
numbers are correct within the question's tolerance, integers exactly, choices
by name. Answer types: __ANSWER_TYPES__. Sources: __SOURCES__.

## Install

The simulator is not on PyPI. uv reads `[tool.uv.sources]` in `pyproject.toml`__SOURCE_NOTE__;
with pip, install it first:

```bash
pip install "scientific-data-simulator @ git+https://github.com/sandner-art/scientific-data-simulator.git"
# or: pip install -e /path/to/scientific-data-simulator
pip install -e .
```

## Run

```bash
vf-run __NAME__ -m <model> -n 10
vf-run __NAME__ --env.taskset.multimodal false            # text-only prompts
vf-run __NAME__ --env.taskset.answerable-from '["figure"]' # only figure-reading questions
```

The package exports verifiers' `NullHarness` (plain chat, no tools) as its
default harness; pick another with `--env.agent.harness.id <id>`.

## Tasks

- Prompt: figure caption, question and answer format; by default the figure is
  attached as an image (base64 PNG data URL). With `multimodal=false` only the text is sent.
- Reward `correct` (0 or 1): `simulator.export.tasks.score` applied to the final
  answer: the last `\\boxed{...}` in the reply, else its last non-empty line.
- Task data also carries `question`, `answer`, `answer_type`, `tolerance`, `choices`,
  `figure` (path relative to `data/`), `table` (path of the source table, not copied)
  and `info` (experiment id, seed, model, fidelity, template, answerable_from).

Questions marked `inference` ask for model parameters or for latent series shown
only through noisy observations; they are hard by design.

## Files

```
__NAME__/
├── pyproject.toml
├── README.md
└── __NAME__/
    ├── __init__.py
    ├── taskset.py        # Data, Task (reward), Taskset (load)
    └── data/
        ├── tasks.jsonl   # tasks with answers
        └── figures/      # figures referenced by the tasks
```
"""

DEFAULT_QA_SYSTEM_PROMPT = (
    "You answer questions about scientific figures and data. Reason briefly if needed, "
    "then give only the final answer on the last line (a number, an integer, or one of "
    "the listed choices) without units."
)


def _qa_rows(tasks: Union[Sequence[Dict[str, Any]], str, Sequence[str]]) -> List[Dict[str, Any]]:
    if isinstance(tasks, (str, os.PathLike)):
        return collect_bundle_tasks(os.fspath(tasks))
    rows = list(tasks)
    if rows and all(isinstance(t, (str, os.PathLike)) for t in rows):
        return collect_bundle_tasks([os.fspath(t) for t in rows])
    return [dict(t) for t in rows]


def export_verifiers_qa(tasks: Union[Iterable[Dict[str, Any]], str, Sequence[str]], output_dir: str,
                        name: str = "sds_bundle_qa", *, system_prompt: Optional[str] = None,
                        copy_figures: bool = True,
                        simulator_source: Optional[str] = DEFAULT_SIMULATOR_SOURCE,
                        overwrite: bool = False) -> str:
    """Write a verifiers (v1) taskset package of single-turn bundle questions.

    Args:
        tasks: Tasks from :func:`~simulator.export.tasks.collect_bundle_tasks`
            (with answers), or run/design directories to collect them from.
        output_dir: Directory in which the package directory ``name`` is created.
        name: Package and taskset id (snake_case): ``vf-run <name>``.
        system_prompt: System prompt of every task (default: answer on the last line).
        copy_figures: Copy the figures into the package (``data/figures/``,
            deduplicated by content); otherwise the tasks keep the original paths.
        simulator_source: Where uv installs the simulator from (see :func:`export_openenv`).
        overwrite: Replace an existing, non-empty package directory.

    Layout: ``<name>/pyproject.toml``, ``<name>/README.md``, ``<name>/<name>/__init__.py``
    (exports the taskset and verifiers' ``NullHarness`` as default harness),
    ``<name>/<name>/taskset.py``, ``<name>/<name>/data/tasks.jsonl``,
    ``<name>/<name>/data/figures/``.

    Returns:
        The path of the package directory (the one with ``pyproject.toml``).
    """
    name = _check_name(name)
    rows = _qa_rows(tasks)
    if not rows:
        raise ValueError("No tasks to export (do the runs have bundles with Q&A?)")
    required = ("id", "prompt", "question", "answer", "answer_type")
    for row in rows:
        missing = [k for k in required if row.get(k) is None]
        if missing:
            raise ValueError(f"Task {row.get('id', '?')!r} lacks {missing} "
                             "(collect tasks with collect_bundle_tasks; answers are required)")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Task ids must be unique")

    path = _prepare_dir(output_dir, name, overwrite)
    pkg = os.path.join(path, name)
    data_dir = os.path.join(pkg, "data")
    fig_dir = os.path.join(data_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    copied: Dict[str, str] = {}
    out_rows: List[Dict[str, Any]] = []
    for row in rows:
        image = row.get("image")
        figure = None
        if image:
            if copy_figures and os.path.isfile(image):
                src = os.path.abspath(image)
                if src not in copied:
                    with open(src, "rb") as f:
                        digest = hashlib.sha256(f.read()).hexdigest()[:16]
                    fname = digest + (os.path.splitext(src)[1].lower() or ".png")
                    target = os.path.join(fig_dir, fname)
                    if not os.path.exists(target):
                        shutil.copy2(src, target)
                    copied[src] = "figures/" + fname
                figure = copied[src]
            else:
                figure = image
        out = {k: v for k, v in row.items() if k != "image"}
        out["figure"] = figure
        out_rows.append(out)
    with open(os.path.join(data_dir, "tasks.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for out in out_rows:
            f.write(json.dumps(out, allow_nan=False) + "\n")

    source = _simulator_source(simulator_source)
    prefix = _pascal(name)
    r = {
        "__NAME__": name,
        "__DIST_NAME__": name.replace("_", "-"),
        "__TITLE__": _title(name),
        "__PREFIX__": prefix,
        "__SYSTEM_PROMPT__": json.dumps(system_prompt if system_prompt is not None else DEFAULT_QA_SYSTEM_PROMPT),
        "__VERIFIERS_REQ__": VERIFIERS_REQUIREMENT,
        "__SIMULATOR_DIST__": SIMULATOR_DISTRIBUTION,
        "__UV_SOURCES__": _uv_sources_block(source),
        "__SOURCE_NOTE__": f" (`{simulator_source}`)" if source["uv_source"] else "",
        "__N_TASKS__": str(len(out_rows)),
        "__N_FIGURES__": str(len(set(copied.values())) if copy_figures else
                             len({r_["figure"] for r_ in out_rows if r_["figure"]})),
        "__ANSWER_TYPES__": ", ".join(sorted({str(r_["answer_type"]) for r_ in out_rows})),
        "__SOURCES__": ", ".join(sorted({str(r_.get("answerable_from")) for r_ in out_rows})),
    }
    _write(os.path.join(pkg, "taskset.py"), _fill(_VF_TASKSET, r))
    _write(os.path.join(pkg, "__init__.py"), _fill(_VF_INIT, r))
    _write(os.path.join(path, "pyproject.toml"), _fill(_VF_PYPROJECT, r))
    _write(os.path.join(path, "README.md"), _fill(_VF_README, r))
    return path


# ============================================================================ verifiers + OpenEnv
def verifiers_openenv_taskset_config(base_url: Optional[str] = None, env: Optional[str] = None,
                                     seeds: Iterable[int] = (0,), use_docker: bool = False,
                                     provider_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Configuration of verifiers' built-in ``openenv`` taskset for an exported environment.

    One task per seed: each task resets the environment with ``{"seed": s}``.
    Give ``base_url`` of a running server (``uv run --project <package> server``),
    or ``env``, an OpenEnv environment id (e.g. a Hugging Face Space after
    ``openenv push``). Validate with
    ``verifiers.v1.utils.loaders.taskset_config_type("openenv").model_validate(config)``.
    """
    if not base_url and not env:
        raise ValueError("pass base_url or env")
    resets = [{"seed": int(s)} for s in seeds]
    if not resets:
        raise ValueError("seeds must not be empty")
    return {"id": "openenv", "env": env, "base_url": base_url, "use_docker": bool(use_docker),
            "provider_kwargs": dict(provider_kwargs or {}), "resets": resets}


__all__ = ["export_openenv", "export_verifiers_qa", "verifiers_openenv_taskset_config",
           "DEFAULT_QA_SYSTEM_PROMPT", "DEFAULT_SIMULATOR_SOURCE"]
