# simulator/diagnostics/episodes.py
"""Agent-fallacy diagnostics for interactive discovery episodes.

An episode in :class:`simulator.envs.ExperimentEnv` is a short scientific inquiry: the agent
chooses experiments, sees noisy measurements and commits to an answer. Because the simulator
knows the truth, each episode can be checked not only for *whether* the answer is right but for
*how* it went wrong. :func:`diagnose_episode` flags:

``no_answer``              the episode ended without an answer.
``invalid_answer``         the answer could not be parsed.
``answer_without_evidence`` an answer given before any successful experiment.
``premature_commitment``   a wrong answer given while budget was left (more evidence was free).
``repeated_experiment``    an experiment repeating earlier inputs exactly. Informational: repeats
                           can be a deliberate way to average out noise; they add no new design point.
``narrow_sampling``        (expression targets) less than half of the scoring domain was measured,
                           so the answer extrapolates beyond the evidence.
``contradicts_own_data``   (expression targets) a *wrong* answer that fits the agent's own
                           measurements ``misfit_threshold`` (3) times worse than the truth does,
                           the truth's misfit floored at ``tolerance`` x RMS of the truth on those
                           points (so noise-free data do not flag near-misses): evidence ignored.
``matches_<name>``         a wrong answer that matches an alternative hypothesis (e.g. ``textbook``,
                           the textbook form of a counterfactual law): a prior preferred over
                           evidence. Only evaluated when the alternative is distinguishable from
                           the truth on the scoring domain (relative RMS > tolerance); episodes
                           where it is not carry no such flag, and :func:`summarize` reports
                           rates over the episodes where each flag applies.

Agents are any callable ``agent(observation, log) -> action``; :func:`run_episodes` records
episodes as :class:`EpisodeLog` objects, and :func:`summarize` turns diagnoses into rates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import numpy as np

FLAGS = ("no_answer", "invalid_answer", "answer_without_evidence", "premature_commitment",
         "repeated_experiment", "narrow_sampling", "contradicts_own_data")


@dataclass
class EpisodeLog:
    seed: Optional[int]
    task: Dict[str, Any]
    steps: List[Dict[str, Any]] = field(default_factory=list)  # {"action", "observation", "reward", "info"}
    truth: Dict[str, Any] = field(default_factory=dict)
    target: str = ""
    target_type: str = "auto"
    domain: Optional[List[float]] = None
    tolerance: float = 0.0
    budget: int = 0

    @property
    def experiments(self) -> List[Dict[str, Any]]:
        return [s for s in self.steps if s["action"].get("type") == "experiment"]

    @property
    def answer_step(self) -> Optional[Dict[str, Any]]:
        return next((s for s in self.steps if s["action"].get("type") == "answer"), None)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self.__dict__, default=_jsonable))


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def run_episodes(env, agent: Callable[[Dict[str, Any], EpisodeLog], Dict[str, Any]], seeds: Iterable[int],
                 max_steps: Optional[int] = None) -> List[EpisodeLog]:
    """Play one episode per seed and record everything a grader needs (truth included)."""
    logs = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        log = EpisodeLog(seed=seed, task=obs, target=env.spec.target, target_type=env.spec.target_type,
                         domain=env.domain, tolerance=float(env.spec.tolerance), budget=int(env.spec.budget))
        limit = max_steps if max_steps is not None else int(env.spec.budget) + 3
        for _ in range(limit):
            action = agent(obs, log)
            out = env.step(action)
            log.steps.append({"action": action, "observation": out["observation"], "reward": out["reward"],
                              "info": out.get("info", {})})
            obs = out["observation"]
            if out["done"]:
                break
        log.truth = env.truth
        logs.append(log)
    return logs


def _measured_xy(log: EpisodeLog):
    """(x, y) pairs from the agent's own measurements (1-D observations with an x axis)."""
    xs, ys = [], []
    for step in log.experiments:
        for entry in (step["observation"].get("measurements") or {}).values():
            values = entry.get("values")
            axis = next((k for k in entry if k not in ("values", "units")), None)
            if isinstance(values, list) and axis is not None and isinstance(entry[axis], list):
                xs.extend(entry[axis])
                ys.extend(values)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _coverage(xs: np.ndarray, domain) -> float:
    if xs.size == 0 or domain is None:
        return 0.0
    lo, hi = float(domain[0]), float(domain[1])
    inside = np.clip(np.sort(xs), lo, hi)
    span = hi - lo
    if span <= 0:
        return 1.0
    # measured spacing: gaps wider than 3x the median spacing count as unmeasured
    gaps = np.diff(inside)
    typical = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else span
    covered = float(np.sum(np.minimum(gaps, 3 * typical)))
    return min(1.0, covered / span)


def diagnose_episode(log: EpisodeLog, alternatives: Optional[Mapping[str, Optional[str]]] = None,
                     misfit_threshold: float = 3.0) -> Dict[str, Any]:
    """Flags and measures for one episode (see the module docstring)."""
    from ..expressions import ExpressionError, compare, evaluate
    answer = log.answer_step
    counted = [s for s in log.experiments if "experiment" in s["observation"]]  # not over-budget attempts
    done_experiments = [s for s in counted if "error" not in s["observation"]]
    result: Dict[str, Any] = {"seed": log.seed, "n_experiments": len(counted),
                              "n_failed": len(counted) - len(done_experiments),
                              "correct": bool(answer and answer["info"].get("correct")),
                              "reward": float(answer["reward"]) if answer and answer["reward"] is not None else 0.0}
    flags = {f: False for f in FLAGS}
    flags["no_answer"] = answer is None
    seen = []
    for step in counted:
        inputs = json.dumps(step["action"].get("inputs", {}), sort_keys=True, default=str)
        if inputs in seen:
            flags["repeated_experiment"] = True
        seen.append(inputs)
    if answer is not None:
        value = answer["action"].get("value")
        flags["answer_without_evidence"] = not done_experiments
        flags["premature_commitment"] = (not result["correct"]) and len(counted) < log.budget
        if log.target_type == "expression":
            truth = log.truth.get(log.target)
            domain = tuple(log.domain) if log.domain else None
            try:
                error = compare(str(value), str(truth), domain)["relative_rms"]
            except ExpressionError:
                error = float("inf")
                flags["invalid_answer"] = True
            result["relative_rms_error"] = error
            xs, ys = _measured_xy(log)
            result["domain_coverage"] = _coverage(xs, domain)
            flags["narrow_sampling"] = bool(done_experiments) and result["domain_coverage"] < 0.5
            if xs.size and not flags["invalid_answer"]:
                with np.errstate(all="ignore"):
                    try:
                        fit_answer = np.sqrt(np.mean((ys - evaluate(str(value), x=xs)) ** 2))
                    except ExpressionError:
                        fit_answer = float("inf")
                    truth_values = evaluate(str(truth), x=xs)
                    fit_truth = np.sqrt(np.mean((ys - truth_values) ** 2))
                floor = max(float(log.tolerance) * float(np.sqrt(np.mean(truth_values ** 2))), 1e-300)
                ratio = float(fit_answer / max(fit_truth, floor))
                result["misfit_ratio"] = ratio if np.isfinite(ratio) else float("inf")
                flags["contradicts_own_data"] = (not result["correct"]) and result["misfit_ratio"] > misfit_threshold
            for name, alt in (alternatives or {}).items():
                if alt is None or flags["invalid_answer"]:
                    continue
                try:
                    if compare(alt, str(truth), domain)["relative_rms"] <= log.tolerance:
                        continue  # indistinguishable from the truth here: the flag cannot apply
                    alt_error = compare(str(value), alt, domain)["relative_rms"]
                except ExpressionError:
                    continue
                result[f"error_vs_{name}"] = alt_error
                flags[f"matches_{name}"] = (not result["correct"]) and alt_error <= log.tolerance
        else:
            info = answer.get("info") or {}
            flags["invalid_answer"] = "estimate" in info and info["estimate"] is None
    result["flags"] = flags
    return result


def law_alternatives(log: EpisodeLog) -> Dict[str, Optional[str]]:
    """``{"textbook": ...}``: the textbook counterpart of the episode's hidden law (law-discovery
    environments with a ``law`` hidden value; ``None`` when the family has no structural constant)."""
    from experiments.law_discovery.logic import textbook_counterpart
    family = log.truth.get("law")
    truth = log.truth.get(log.target)
    if family is None or truth is None or not log.domain:
        return {"textbook": None}
    try:
        return {"textbook": textbook_counterpart(family, str(truth), tuple(log.domain))}
    except (RuntimeError, ValueError):  # the constrained fit failed: no alternative to compare
        return {"textbook": None}


def summarize(diagnoses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Rates of each flag (over the episodes where it applies), accuracy, and accuracy when flagged."""
    n = len(diagnoses)
    names = sorted({f for d in diagnoses for f in d["flags"]})
    out: Dict[str, Any] = {"episodes": n, "accuracy": float(np.mean([d["correct"] for d in diagnoses])) if n else 0.0,
                           "mean_experiments": float(np.mean([d["n_experiments"] for d in diagnoses])) if n else 0.0,
                           "flags": {}}
    for name in names:
        applicable = [d for d in diagnoses if name in d["flags"]]
        hit = [d for d in applicable if d["flags"][name]]
        out["flags"][name] = {"rate": len(hit) / len(applicable) if applicable else 0.0, "count": len(hit),
                              "applicable": len(applicable),
                              "accuracy_when_flagged": float(np.mean([d["correct"] for d in hit])) if hit else None}
    return out
