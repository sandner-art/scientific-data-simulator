# simulator/envs.py
"""Interactive "run experiments on a hidden system" environments.

An episode hides one or more model parameters (drawn from priors with the
episode seed). The agent chooses inputs within bounds, runs a limited number of
noisy experiments, and finally submits an estimate of a target parameter (or
picks the hidden functional form). The answer is scored automatically.

This is the multi-turn, knowledge-heavy task shape that text-only procedural
generators lack. The class has no dependencies beyond the simulator; adapters
for OpenEnv and verifiers live in :mod:`simulator.export`.

Specification (YAML or dict)::

    base_config: {experiment_type: ..., n_steps: ..., observation: {...}}   # or a path
    hidden:                     # unknown to the agent; drawn per episode
      prey_growth_rate: [0.6, 1.6]
      functional_response: {choices: [linear, holling2]}
    controls:                   # the agent sets these per experiment (dotted keys allowed)
      initial_prey: [1, 50]
      initial_predators: [1, 20]
    target: prey_growth_rate    # what must be reported (a hidden parameter, or with
                                # target_type: expression a formula label such as law_expression)
    budget: 5                   # number of experiments
    tolerance: 0.05             # relative tolerance for numeric targets
    max_points: 60              # observed series are thinned to at most this many points
    description: "Estimate the prey growth rate of an ecosystem."

Actions (dicts)::

    {"type": "experiment", "inputs": {"initial_prey": 20, "initial_predators": 3}}
    {"type": "answer", "value": 1.07}

Every step returns ``{"observation": ..., "reward": float | None, "done": bool, "info": {...}}``.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .config import apply_overrides, load_config
from .engine import simulate
from .rng import derive_seeds, new_seed, validate_seed
from .utils import Role


@dataclass
class ExperimentTaskSpec:
    base_config: Dict[str, Any]
    hidden: Dict[str, Any]
    controls: Dict[str, Any]
    target: str
    budget: int = 5
    tolerance: float = 0.05
    max_points: int = 60
    description: str = ""
    measured: Optional[List[str]] = None  # default: all outputs with role "observation"
    # "auto": numeric/choice from the hidden prior; "expression": a formula scored on
    # ``domain`` (default: the range of the model's first coordinate output).
    target_type: str = "auto"
    domain: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, spec: Dict[str, Any]) -> "ExperimentTaskSpec":
        spec = dict(spec)
        base = spec.pop("base_config")
        if isinstance(base, str):
            base = load_config(base)
        unknown = set(spec) - {"hidden", "controls", "target", "budget", "tolerance", "max_points",
                               "description", "measured", "target_type", "domain"}
        if unknown:
            raise ValueError(f"Unknown environment spec keys: {sorted(unknown)}")
        task = cls(base_config=dict(base), **spec)
        task.validate()
        return task

    def validate(self) -> None:
        if not self.hidden:
            raise ValueError("spec.hidden must name at least one hidden parameter")
        if self.target_type not in ("auto", "expression"):
            raise ValueError("spec.target_type must be 'auto' or 'expression'")
        if self.target_type == "auto" and self.target not in self.hidden:
            raise ValueError(f"spec.target '{self.target}' must be one of the hidden parameters "
                             "(or a label output with target_type: expression)")
        overlap = set(self.hidden) & set(self.controls)
        if overlap:
            raise ValueError(f"Parameters cannot be both hidden and controlled: {sorted(overlap)}")
        for name, prior in self.hidden.items():
            _check_range(f"hidden.{name}", prior, allow_choices=True)
        for name, bounds in self.controls.items():
            _check_range(f"controls.{name}", bounds, allow_choices=True)
        if int(self.budget) < 1:
            raise ValueError("spec.budget must be >= 1")
        if not self.tolerance > 0:
            raise ValueError("spec.tolerance must be > 0")

    def target_is_categorical(self) -> bool:
        return self.target_type == "auto" and isinstance(self.hidden[self.target], dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"base_config": self.base_config, "hidden": self.hidden, "controls": self.controls,
                "target": self.target, "budget": self.budget, "tolerance": self.tolerance,
                "max_points": self.max_points, "description": self.description, "measured": self.measured,
                "target_type": self.target_type, "domain": self.domain}


def _check_range(name: str, spec: Any, allow_choices: bool) -> None:
    if isinstance(spec, dict):
        if not (allow_choices and isinstance(spec.get("choices"), list) and spec["choices"]):
            raise ValueError(f"{name}: use [low, high] or {{choices: [...]}}")
        return
    if not (isinstance(spec, (list, tuple)) and len(spec) == 2 and float(spec[0]) < float(spec[1])):
        raise ValueError(f"{name}: expected [low, high] with low < high, got {spec!r}")


def _thin(values: np.ndarray, max_points: int) -> np.ndarray:
    if len(values) <= max_points:
        return np.arange(len(values))
    return np.unique(np.round(np.linspace(0, len(values) - 1, max_points)).astype(int))


def _round(x: float) -> Optional[float]:
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.6g}")


class ExperimentEnv:
    """Gymnasium-style environment (``reset`` / ``step`` / ``state``) over a simulator model."""

    def __init__(self, spec: Union[ExperimentTaskSpec, Dict[str, Any]]):
        self.spec = spec if isinstance(spec, ExperimentTaskSpec) else ExperimentTaskSpec.from_dict(spec)
        self.spec.validate()
        self._truth: Dict[str, Any] = {}
        self._episode_seed: Optional[int] = None
        self._experiment_seeds: List[int] = []
        self._history: List[Dict[str, Any]] = []
        self._done = True
        self._steps = 0

    # ------------------------------------------------------------------ API
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self._episode_seed = new_seed() if seed is None else validate_seed(seed)
        rng = np.random.default_rng(self._episode_seed)
        self._truth = {}
        for name, prior in self.spec.hidden.items():
            if isinstance(prior, dict):
                self._truth[name] = prior["choices"][int(rng.integers(len(prior["choices"])))]
            else:
                self._truth[name] = float(rng.uniform(float(prior[0]), float(prior[1])))
        self._experiment_seeds = derive_seeds(self._episode_seed, int(self.spec.budget) + 1)
        self._history, self._done, self._steps = [], False, 0
        self._target_truth, self._domain = None, None
        if self.spec.target_type == "expression":
            # the target is a label of the mechanism (e.g. the law): read it from a noise-free run
            config = apply_overrides(self.spec.base_config, self._truth)
            config.pop("observation", None)
            config.pop("bundle", None)
            config["seed"] = self._experiment_seeds[-1]
            sim = simulate(config)
            if self.spec.target not in sim.results:
                raise ValueError(f"target '{self.spec.target}' is not an output of the model")
            self._target_truth = str(sim.results[self.spec.target]["data"])
            if self.spec.domain:
                self._domain = [float(v) for v in self.spec.domain]
            else:
                coordinate = next(i for i in sim.results.values()
                                  if i["descriptor"].role is Role.COORDINATE)["data"]
                self._domain = [float(np.min(coordinate)), float(np.max(coordinate))]
        return self._task_observation()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._done:
            raise RuntimeError("Episode is finished; call reset()")
        self._steps += 1
        kind = action.get("type") if isinstance(action, dict) else None
        if kind == "experiment":
            return self._experiment(action.get("inputs") or {})
        if kind == "answer":
            return self._answer(action.get("value"))
        return self._result({"error": "action must be {'type': 'experiment', 'inputs': {...}} or "
                                      "{'type': 'answer', 'value': ...}"}, None, False)

    @property
    def state(self) -> Dict[str, Any]:
        """Agent-visible bookkeeping. Seeds are excluded: with the (public) priors,
        the episode seed would regenerate the hidden values."""
        return {"step_count": self._steps, "experiments_used": len(self._history),
                "budget_left": int(self.spec.budget) - len(self._history),
                "history": [{"inputs": dict(h["inputs"])} for h in self._history], "done": self._done}

    @property
    def private_state(self) -> Dict[str, Any]:
        """Grader/logging view: the state plus the episode seed, experiment seeds and truth."""
        return {**self.state, "episode_seed": self._episode_seed, "truth": self.truth,
                "history": copy.deepcopy(self._history)}

    @property
    def truth(self) -> Dict[str, Any]:
        """The hidden values of the current episode (for graders, never for agents)."""
        truth = dict(self._truth)
        if self.spec.target_type == "expression" and self._target_truth is not None:
            truth[self.spec.target] = self._target_truth
        return truth

    # ------------------------------------------------------------------ internals
    def _task_observation(self) -> Dict[str, Any]:
        spec = self.spec
        if spec.target_type == "expression":
            goal = (f"Find a formula for the hidden law ('{spec.target}') that reproduces the noise-free "
                    f"relation on [{self._domain[0]:.4g}, {self._domain[1]:.4g}] within "
                    f"{spec.tolerance:.0%} relative RMS. Answer with a formula in x, e.g. 2.5*x**1.3.")
        elif spec.target_is_categorical():
            goal = (f"Determine which of {spec.hidden[spec.target]['choices']} the hidden "
                    f"'{spec.target}' is. Answer with one of these strings.")
        else:
            goal = (f"Estimate the hidden parameter '{spec.target}'. Answers within "
                    f"{spec.tolerance:.0%} (relative) count as correct.")
        text = "\n".join(filter(None, [
            spec.description,
            f"Model: {spec.base_config.get('experiment_type', '?').rsplit('.', 1)[-1]}.",
            goal,
            f"You may run up to {spec.budget} experiments. In each, set the inputs "
            f"{json.dumps(spec.controls)} (numbers within [low, high]).",
            "Action format: {\"type\": \"experiment\", \"inputs\": {...}} or {\"type\": \"answer\", \"value\": ...}.",
        ]))
        return {"task": text, "controls": copy.deepcopy(spec.controls), "budget": int(spec.budget),
                "target": spec.target, "done": False}

    def _validate_inputs(self, inputs: Dict[str, Any]) -> Optional[str]:
        unknown = set(inputs) - set(self.spec.controls)
        if unknown:
            return f"unknown inputs {sorted(unknown)}; allowed: {sorted(self.spec.controls)}"
        for name, value in inputs.items():
            bounds = self.spec.controls[name]
            if isinstance(bounds, dict):
                if value not in bounds["choices"]:
                    return f"{name} must be one of {bounds['choices']}"
            else:
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    return f"{name} must be a number"
                if not (float(bounds[0]) <= v <= float(bounds[1])) or not math.isfinite(v):
                    return f"{name}={value} is outside [{bounds[0]}, {bounds[1]}]"
        return None

    def _experiment(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if len(self._history) >= int(self.spec.budget):
            self._done = True
            return self._result({"error": "experiment budget exhausted; episode ends without an answer"},
                                0.0, True, {"truth": self.truth})
        problem = self._validate_inputs(inputs)
        if problem:
            return self._result({"error": problem}, None, False)
        index = len(self._history)
        config = apply_overrides(self.spec.base_config, {**self._truth, **inputs})
        config["seed"] = self._experiment_seeds[index]
        config.pop("bundle", None)
        try:
            sim = simulate(config)
        except ValueError:
            # The model refused these inputs. The message could name hidden values, and whether a
            # run fails can depend on them, so the agent learns only that it failed - and pays
            # for it like for any experiment, or failures would be a free oracle.
            self._history.append({"inputs": dict(inputs), "seed": config["seed"], "failed": True})
            return self._result({"error": "experiment failed for these inputs (no data)", "experiment": index + 1,
                                 "inputs": dict(inputs), "budget_left": int(self.spec.budget) - len(self._history)},
                                None, False)
        measured = self.spec.measured or [n for n, i in sim.results.items()
                                          if i["descriptor"].role is Role.OBSERVATION]
        data: Dict[str, Any] = {}
        for name in measured:
            info = sim.results[name]
            values = np.asarray(info["data"], dtype=float)
            x_name = info["descriptor"].x_axis
            idx = _thin(values, int(self.spec.max_points)) if values.ndim == 1 else None
            entry = {"units": info["descriptor"].units}
            if idx is not None:
                entry["values"] = [_round(v) for v in values[idx]]
                if x_name and x_name in sim.results:
                    entry[x_name] = [_round(v) for v in np.asarray(sim.results[x_name]["data"], float)[idx]]
            else:
                entry["values"] = _round(float(values))
            data[name] = entry
        self._history.append({"inputs": dict(inputs), "seed": config["seed"]})
        return self._result({"experiment": index + 1, "inputs": dict(inputs), "measurements": data,
                             "budget_left": int(self.spec.budget) - len(self._history)}, None, False)

    def _answer(self, value: Any) -> Dict[str, Any]:
        self._done = True
        if self.spec.target_type == "expression":
            from .expressions import compare, ExpressionError
            try:
                error = compare(str(value), self._target_truth, tuple(self._domain))["relative_rms"]
            except ExpressionError as e:
                error, value = float("inf"), f"{value} (invalid: {e})"
            correct = error <= float(self.spec.tolerance)
            info = {"truth": self.truth, "relative_rms_error": error, "correct": correct}
            return self._result({"answer": value, "correct": correct}, 1.0 if correct else 0.0, True, info)
        truth = self._truth[self.spec.target]
        if self.spec.target_is_categorical():
            correct = str(value).strip().lower() == str(truth).strip().lower()
            info = {"truth": self.truth, "correct": correct}
        else:
            try:
                estimate = float(value)
                rel_error = abs(estimate - truth) / max(abs(truth), 1e-12)
            except (TypeError, ValueError):
                estimate, rel_error = None, float("inf")
            correct = rel_error <= float(self.spec.tolerance)
            info = {"truth": self.truth, "estimate": estimate, "relative_error": rel_error, "correct": correct}
        return self._result({"answer": value, "correct": correct}, 1.0 if correct else 0.0, True, info)

    def _result(self, observation, reward, done, info=None) -> Dict[str, Any]:
        observation = dict(observation)
        observation["done"] = done
        return {"observation": observation, "reward": reward, "done": done, "info": info or {}}


def render_observation(observation: Dict[str, Any]) -> str:
    """Plain-text form of an observation, for language-model agents."""
    if "task" in observation:
        return observation["task"]
    return json.dumps({k: v for k, v in observation.items() if k != "done"}, indent=1)
