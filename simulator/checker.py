# simulator/checker.py
"""Automatic checks for ExperimentLogic models - the gate for LLM-written code.

Plausible-looking simulation code can be silently wrong. Before a model (in
particular a generated one) is used to produce data, :func:`check_model` runs:

static (source code, AST)
  * ``lint``          - global ``np.random``/``random`` use, eval/exec, subprocess/os.system,
                        network modules, file deletion (fail); wall-clock reads, ``open`` (warn)
runtime (through :func:`simulator.engine.simulate`)
  * ``runs``          - constructs, steps and returns validated results
  * ``finite``        - latent/label/parameter outputs contain no NaN/inf
  * ``roles``         - outputs declare roles; at least one ``latent`` output
  * ``determinism``   - the same seed gives identical outputs
  * ``global_rng``    - the global ``np.random`` and ``random`` states are untouched
  * ``config_isolation`` - the model does not modify the config it receives
  * ``fidelity``      - a valid fidelity level is declared
  * ``units``         - output and parameter units parse (needs pint; else skipped)
  * ``convergence``   - halving ``dt`` (keeping the end time) converges; reports the
                        observed order, or that results do not depend on ``dt``
  * ``invariants``    - quantities from ``model.invariants(results)`` are conserved
  * ``sensitivity``   - each inferable parameter changes the output (no dead parameters)

Passing the gate makes a model *accepted*; it never makes it ``reference``.
Reference status needs verification against independent knowledge (closed
forms, high-accuracy solvers, invariants) in ``tests/test_verification.py``.

:func:`check_source` runs the whole check on source code in a separate Python
process with a time limit and a minimal environment (no API keys), after the
static lint. The report must carry a one-time token and contain every check.
This guards against careless or buggy code, the usual failure of generated
simulation code; it is not a security sandbox. Run code you do not trust only
inside an isolated container or VM.
"""
from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import os
import random
import secrets
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np

from .base import ExperimentLogic
from .engine import resolve_experiment_type, simulate
from .utils import FIDELITY_LEVELS, GROUND_TRUTH_ROLES, Role

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"

_ALLOWED_NP_RANDOM = {"default_rng", "Generator", "SeedSequence", "PCG64", "PCG64DXSM", "Philox", "SFC64",
                      "BitGenerator", "RandomState"}  # seeded generator objects, not global draws
_ALLOWED_STDLIB_RANDOM = {"Random"}  # random.Random(seed) is a seeded instance
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "breakpoint", "input"}
_FORBIDDEN_MODULES = {"subprocess", "socket", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp",
                      "ftplib", "smtplib", "ctypes", "multiprocessing", "pickle", "shutil", "importlib", "pty",
                      "signal", "webbrowser", "asyncio", "inspect", "gc", "__main__", "builtins"}
_FORBIDDEN_OS = {"system", "popen", "remove", "unlink", "rmdir", "removedirs", "kill", "killpg", "fork",
                 "forkpty", "startfile", "truncate", "rename", "renames", "replace", "chmod", "chown",
                 "putenv", "unsetenv"}
_FORBIDDEN_OS_PREFIXES = ("exec", "spawn", "posix_spawn")
_FORBIDDEN_METHODS = {"unlink", "rmdir", "rmtree"}  # e.g. pathlib.Path(...).unlink()
_WARN_METHODS = {"write_text", "write_bytes", "mkdir", "touch"}
_NONDETERMINISTIC = {("time", "time"), ("time", "perf_counter"), ("time", "monotonic"), ("datetime", "now"),
                     ("datetime", "today"), ("uuid", "uuid4"), ("os", "urandom"), ("random", "SystemRandom"),
                     ("secrets", "*")}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckReport:
    model: str
    results: List[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", **data: Any) -> CheckResult:
        result = CheckResult(name, status, detail, data)
        self.results.append(result)
        return result

    @property
    def failures(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def warnings(self) -> List[CheckResult]:
        return [r for r in self.results if r.status == WARN]

    @property
    def accepted(self) -> bool:
        return not self.failures

    def get(self, name: str) -> Optional[CheckResult]:
        return next((r for r in self.results if r.name == name), None)

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "accepted": self.accepted,
                "summary": {s: sum(r.status == s for r in self.results) for s in (PASS, WARN, FAIL, SKIP)},
                "results": [asdict(r) for r in self.results]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckReport":
        report = cls(d["model"])
        report.results = [CheckResult(**r) for r in d["results"]]
        return report

    def to_markdown(self) -> str:
        icon = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", SKIP: "skip"}
        lines = [f"# Model check: `{self.model}`", "",
                 f"**Verdict: {'accepted' if self.accepted else 'rejected'}** "
                 f"({len(self.failures)} failed, {len(self.warnings)} warnings)", "",
                 "| check | result | detail |", "|---|---|---|"]
        lines += [f"| {r.name} | {icon[r.status]} | {r.detail.replace('|', '/')} |" for r in self.results]
        lines += ["", "Passing these checks does not make a model `reference`; that requires verification "
                      "against independent knowledge (closed forms, solvers, invariants)."]
        return "\n".join(lines) + "\n"

    def feedback(self) -> str:
        """Failures and warnings as plain text (e.g. to send back to an LLM)."""
        return "\n".join(f"- [{r.status}] {r.name}: {r.detail}" for r in self.results if r.status in (FAIL, WARN))


# ---------------------------------------------------------------------------- static lint
def lint_source(source: str) -> List[CheckResult]:
    """AST scan of model source code for forbidden or nondeterministic constructs.

    This is a gate against careless code (the usual failure of generated
    simulation code), not a security sandbox: run untrusted code only inside an
    isolated container.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [CheckResult("lint", FAIL, f"syntax error: {e}")]
    problems: List[str] = []
    warnings: List[str] = []
    aliases: Dict[str, str] = {}  # local name -> dotted path ("np" -> "numpy", "system" -> "os.system")

    def resolve(node) -> Optional[str]:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(aliases.get(node.id, node.id))
        return ".".join(reversed(parts))

    def check_path(path: str, line: int, called: bool) -> None:
        parts = path.split(".")
        root, last = parts[0], parts[-1]
        if root in _FORBIDDEN_MODULES:
            problems.append(f"line {line}: uses '{path}'")
        if root == "sys" and len(parts) >= 2 and parts[1] in ("_getframe", "settrace", "setprofile", "modules"):
            problems.append(f"line {line}: uses {path} (introspection of the checking process)")
        if root == "os" and len(parts) >= 2 and (parts[1] in _FORBIDDEN_OS or parts[1].startswith(_FORBIDDEN_OS_PREFIXES)):
            problems.append(f"line {line}: calls {path}()")
        if parts[:2] == ["numpy", "random"] and len(parts) >= 3 and parts[2] not in _ALLOWED_NP_RANDOM and called:
            problems.append(f"line {line}: uses global RNG {path}(); use self.rng")
        if root == "random" and len(parts) == 2 and parts[1] not in _ALLOWED_STDLIB_RANDOM and called:
            if parts[1] == "SystemRandom":
                warnings.append(f"line {line}: uses random.SystemRandom (nondeterministic)")
            else:
                problems.append(f"line {line}: uses the stdlib global RNG {path}(); use self.rng")
        if (root, last) in _NONDETERMINISTIC or (root, "*") in _NONDETERMINISTIC or parts[-2:] == ["datetime", "now"]:
            warnings.append(f"line {line}: reads {path}() (nondeterministic)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                if a.name.split(".")[0] in _FORBIDDEN_MODULES:
                    problems.append(f"line {node.lineno}: imports '{a.name}'")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in _FORBIDDEN_MODULES:
                problems.append(f"line {node.lineno}: imports from '{node.module}'")
            for a in node.names:
                full = f"{node.module}.{a.name}"
                aliases[a.asname or a.name] = full
                if node.module == "numpy.random" and a.name not in _ALLOWED_NP_RANDOM:
                    problems.append(f"line {node.lineno}: imports global RNG function '{full}'; use self.rng")
                if node.module == "random" and a.name not in _ALLOWED_STDLIB_RANDOM:
                    problems.append(f"line {node.lineno}: imports '{full}' from the stdlib global RNG; use self.rng")
                if node.module == "os" and (a.name in _FORBIDDEN_OS or a.name.startswith(_FORBIDDEN_OS_PREFIXES)):
                    problems.append(f"line {node.lineno}: imports {full}")
    # aliases created by assignment: r = np.random ; s = os.system
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Attribute, ast.Name)):
            path = resolve(node.value)
            if path and path.split(".")[0] in ("numpy", "os", "random", "subprocess", "shutil"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = path

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and aliases.get(node.value.id, node.value.id) == "sys" \
                and node.attr in ("_getframe", "settrace", "setprofile", "modules"):
            problems.append(f"line {node.lineno}: uses sys.{node.attr} (introspection of the checking process)")
        if isinstance(node, ast.Attribute) and node.attr in ("f_back", "f_locals", "f_globals", "gi_frame",
                                                              "tb_frame", "__globals__", "__subclasses__"):
            problems.append(f"line {node.lineno}: accesses .{node.attr} (frame/object introspection)")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _FORBIDDEN_CALLS:
                problems.append(f"line {node.lineno}: calls {func.id}()")
            elif func.id == "open":
                warnings.append(f"line {node.lineno}: opens a file")
            elif func.id in ("getattr", "setattr", "delattr") and node.args:
                target = resolve(node.args[0]) or ""
                name = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
                if target.split(".")[0] in ("os", "numpy", "random", "sys", "builtins") or target in _FORBIDDEN_MODULES:
                    if name is None or not isinstance(name, str):
                        problems.append(f"line {node.lineno}: dynamic {func.id}() on '{target}'")
                    else:
                        check_path(f"{target}.{name}", node.lineno, called=True)
            if func.id in aliases:
                check_path(aliases[func.id], node.lineno, called=True)
        elif isinstance(func, ast.Attribute):
            path = resolve(func)
            if path:
                check_path(path, node.lineno, called=True)
            if func.attr in _FORBIDDEN_METHODS:
                problems.append(f"line {node.lineno}: calls .{func.attr}() (deletes files)")
            elif func.attr in _WARN_METHODS:
                warnings.append(f"line {node.lineno}: calls .{func.attr}() (writes to disk)")
    problems = list(dict.fromkeys(problems))
    warnings = list(dict.fromkeys(warnings))
    if problems:
        return [CheckResult("lint", FAIL, "; ".join(problems))]
    if warnings:
        return [CheckResult("lint", WARN, "; ".join(warnings))]
    return [CheckResult("lint", PASS, "no forbidden constructs")]


# ---------------------------------------------------------------------------- helpers
def _resolve(model: Union[str, Type[ExperimentLogic]]) -> Type[ExperimentLogic]:
    if isinstance(model, str):
        return resolve_experiment_type({"experiment_type": model})
    if not (isinstance(model, type) and issubclass(model, ExperimentLogic)):
        raise TypeError("model must be an ExperimentLogic subclass or a dotted path")
    return model


def _arrays(results, roles=None) -> Dict[str, np.ndarray]:
    out = {}
    for name, info in results.items():
        role = info["descriptor"].role
        if roles is not None and role not in roles:
            continue
        data = info["data"]
        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
            out[name] = data.astype(float)
        elif isinstance(data, (int, float, np.number)) and not isinstance(data, bool):
            out[name] = np.array([float(data)])
    return out


def _identical(a, b) -> bool:
    if set(a) != set(b):
        return False
    for name in a:
        da, db = a[name]["data"], b[name]["data"]
        if isinstance(da, np.ndarray) or isinstance(db, np.ndarray):
            if not (isinstance(da, np.ndarray) and isinstance(db, np.ndarray)) or da.shape != db.shape:
                return False
            if not np.array_equal(da, db, equal_nan=np.issubdtype(da.dtype, np.floating)):
                return False
        elif hasattr(da, "equals"):
            if not da.equals(db):
                return False
        elif da != db and not (isinstance(da, float) and math.isnan(da) and isinstance(db, float) and math.isnan(db)):
            return False
    return True


def _max_change(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> float:
    change = 0.0
    for name in a:
        if name in b and a[name].shape == b[name].shape:
            scale = max(float(np.nanmax(np.abs(a[name]))) if a[name].size else 0.0, 1e-300)
            change = max(change, float(np.nanmax(np.abs(a[name] - b[name]))) / scale)
    return change


# ---------------------------------------------------------------------------- main check
EXPECTED_CHECKS = ("lint", "runs", "global_rng", "config_isolation", "finite", "roles", "determinism",
                   "stochasticity", "fidelity", "units", "convergence", "invariants", "sensitivity")


def _module_source(model: Union[str, Type[ExperimentLogic]]) -> Optional[str]:
    """Source of the model's module, found *without importing it* for dotted paths."""
    import importlib.util
    if isinstance(model, str):
        module_name = model.rpartition(".")[0]
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            return None
        if spec is None or not spec.origin or not os.path.exists(spec.origin):
            return None
        with open(spec.origin, encoding="utf-8") as f:
            return f.read()
    try:
        return inspect.getsource(inspect.getmodule(model))
    except (OSError, TypeError):
        return None


def check_model(model: Union[str, Type[ExperimentLogic]], config: Dict[str, Any],
                refinement: Optional[str] = "dt", levels: int = 3, min_order: float = 0.8,
                invariant_tolerance: float = 1e-6, sensitivity_parameters: Optional[List[str]] = None,
                source: Optional[str] = None) -> CheckReport:
    """Run the automatic checks on ``model`` with ``config`` (see module docs).

    The source is linted *before* the module is imported (for dotted paths);
    code that fails the lint is never executed. The caller's global random
    states are restored afterwards.

    Args:
        refinement: Config key of the time step to halve for the convergence
            check (skipped if absent from ``config``).
        levels: Number of resolutions for the convergence check (>= 3).
        min_order: Minimum observed convergence order to pass.
        sensitivity_parameters: Parameters that must influence the output
            (default: the model's ``inferable_parameters``).
        source: Source code to lint (default: the model's module source).
    """
    if source is None:
        source = _module_source(model)
    lint = lint_source(source) if source is not None else [CheckResult("lint", SKIP, "source not available")]
    if any(r.status == FAIL for r in lint):
        name = model if isinstance(model, str) else f"{model.__module__}.{model.__qualname__}"
        report = CheckReport(name)
        report.results.extend(lint)
        report.add("runs", SKIP, "not executed: the code failed the static lint")
        return report

    np_saved, py_saved = np.random.get_state(), random.getstate()
    try:
        return _check_model(_resolve(model), config, refinement, levels, min_order, invariant_tolerance,
                            sensitivity_parameters, lint)
    finally:
        np.random.set_state(np_saved)
        random.setstate(py_saved)


def _check_model(cls, config, refinement, levels, min_order, invariant_tolerance, sensitivity_parameters,
                 lint) -> CheckReport:
    report = CheckReport(f"{cls.__module__}.{cls.__qualname__}")
    config = copy.deepcopy(config)
    config["experiment_type"] = report.model
    for key in ("bundle", "generate_plots", "save_csv", "storage"):
        config.pop(key, None)
    config.setdefault("seed", 12345)
    clean = {k: v for k, v in config.items() if k != "observation"}  # mechanism only

    report.results.extend(lint)

    # -- runs (with global RNG and config isolation probes)
    np.random.seed(2024)
    random.seed(2024)
    np_state, py_state = np.random.get_state()[1].copy(), random.getstate()
    try:
        first = simulate(clean, cls)
    except Exception as e:
        report.add("runs", FAIL, f"{type(e).__name__}: {e}")
        return report
    report.add("runs", PASS, f"{len(first.results)} outputs")
    rng_touched = not np.array_equal(np.random.get_state()[1], np_state) or random.getstate() != py_state
    report.add("global_rng", FAIL if rng_touched else PASS,
               "the global np.random/random state changed during the run; draw from self.rng"
               if rng_touched else "global RNG state untouched")

    probe_config = copy.deepcopy(first.model_config)
    before = copy.deepcopy(probe_config)
    try:
        probe = cls(probe_config)
        probe.rng = np.random.default_rng(0)
        state = probe.initialize(probe_config)
        for step in range(int(probe_config.get("n_steps", 1))):
            state = probe.run_step(state, step)
        probe.get_results()
        mutated = probe_config != before
        report.add("config_isolation", WARN if mutated else PASS,
                   "the model modifies the config it receives (the engine protects the stored config, "
                   "but this usually signals a bug)" if mutated else "config untouched")
    except Exception as e:  # pragma: no cover - same code path as 'runs'
        report.add("config_isolation", WARN, f"probe failed: {e}")

    # -- outputs
    truth = _arrays(first.results, set(GROUND_TRUTH_ROLES))
    bad = [n for n, a in truth.items() if not np.all(np.isfinite(a))]
    report.add("finite", FAIL if bad else PASS,
               f"NaN/inf in ground-truth outputs: {bad}" if bad else "ground-truth outputs are finite")
    missing_roles = [n for n, i in first.results.items() if i["descriptor"].role is None]
    latent = [n for n, i in first.results.items() if i["descriptor"].role is Role.LATENT]
    if not latent:
        report.add("roles", WARN, "no output has role 'latent' (observation operators need one)")
    elif missing_roles:
        report.add("roles", WARN, f"outputs without a role: {missing_roles}")
    else:
        report.add("roles", PASS, f"latent outputs: {latent}")

    # -- determinism and stochasticity
    second = simulate(clean, cls)
    report.add("determinism", PASS if _identical(first.results, second.results) else FAIL,
               "same seed, identical outputs" if _identical(first.results, second.results)
               else "the same seed gave different outputs (hidden randomness or state)")
    other = simulate(dict(clean, seed=int(clean["seed"]) + 1), cls)
    stochastic = not _identical(first.results, other.results)
    report.add("stochasticity", PASS, "stochastic model (output depends on the seed)" if stochastic
               else "deterministic model (output independent of the seed)", stochastic=stochastic)

    # -- fidelity
    fidelity = first.ground_truth.get("fidelity") or {}
    if fidelity.get("level") not in FIDELITY_LEVELS:
        report.add("fidelity", FAIL, f"invalid fidelity level {fidelity.get('level')!r}")
    elif fidelity.get("level") == "reference":
        report.add("fidelity", WARN, "declares 'reference': make sure verification tests back this up")
    else:
        report.add("fidelity", PASS, f"declares '{fidelity.get('level')}'")

    # -- units
    try:
        from .units import available, check_output_units, parse_units
        if not available():
            raise ImportError
        unit_problems = {n: i["problem"] for n, i in check_output_units(first.results).items() if not i["ok"]}
        from .units import declared_parameter_units
        for key, units in declared_parameter_units(cls, first.config).items():
            try:
                parse_units(units)
            except Exception as e:
                unit_problems[f"parameter {key}"] = f"cannot parse '{units}' ({type(e).__name__})"
        report.add("units", WARN if unit_problems else PASS,
                   "; ".join(f"{k}: {v}" for k, v in unit_problems.items()) if unit_problems else "units parse")
    except ImportError:
        report.add("units", SKIP, "pint not installed")

    # -- convergence under refinement of the time step
    _check_convergence(report, cls, clean, first, refinement, max(3, levels), min_order, stochastic)

    # -- invariants
    invariants = getattr(cls, "invariants", None)
    if callable(invariants):
        try:
            values = first.logic.invariants(first.results) or {}
        except Exception as e:
            report.add("invariants", FAIL, f"invariants() failed: {e}")
            values = {}
        drifts = {}
        for name, series in values.items():
            series = np.asarray(series, dtype=float)
            scale = max(float(np.max(np.abs(series))), 1e-300)
            drifts[name] = float(np.max(np.abs(series - series[0])) / scale)
        if values:
            worst = max(drifts.values())
            report.add("invariants", PASS if worst <= invariant_tolerance else WARN,
                       ", ".join(f"{n}: relative drift {d:.2e}" for n, d in drifts.items()), drifts=drifts)
    else:
        report.add("invariants", SKIP, "model declares no invariants()")

    # -- parameter sensitivity
    params = sensitivity_parameters if sensitivity_parameters is not None else list(
        getattr(cls, "inferable_parameters", ()) or ())
    if not params:
        report.add("sensitivity", SKIP, "no inferable_parameters declared")
    else:
        base_latent = _arrays(first.results, {Role.LATENT, Role.LABEL})
        dead, changes = [], {}
        untested = []
        for p in params:
            value = first.model_config.get(p, getattr(first.logic, p, None))  # defaults live on the model
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                untested.append(p)
                continue
            bumped = simulate(dict(clean, **{p: value * 1.1 if value != 0 else 0.1}), cls)
            change = _max_change(base_latent, _arrays(bumped.results, {Role.LATENT, Role.LABEL}))
            changes[p] = change
            if change < 1e-12:
                dead.append(p)
        detail = (f"output does not depend on {dead}" if dead
                  else ", ".join(f"{p}: {c:.2e}" for p, c in changes.items()))
        if untested:
            detail += f"; not tested (no numeric value found): {untested}"
        report.add("sensitivity", FAIL if dead else (WARN if untested else PASS), detail, changes=changes)
    return report


def _check_convergence(report, cls, clean, first, key, levels, min_order, stochastic) -> None:
    if not key or key not in first.model_config or "n_steps" not in first.model_config:
        report.add("convergence", SKIP, f"no '{key}'/'n_steps' in config")
        return
    if stochastic:
        report.add("convergence", SKIP, "stochastic model: refinement changes the random path")
        return
    dt = float(first.model_config[key])
    n = int(first.model_config["n_steps"])
    runs = []
    for level in range(levels):
        factor = 2 ** level
        sim = simulate(dict(clean, **{key: dt / factor, "n_steps": n * factor}), cls)
        latent = {name: arr[::factor] for name, arr in _arrays(sim.results, {Role.LATENT}).items()
                  if arr.ndim == 1 and len(arr) == n * factor + 1}
        runs.append(latent)
    names = sorted(set.intersection(*(set(r) for r in runs))) if runs else []
    if not names:
        report.add("convergence", SKIP, "no time-series latent outputs of length n_steps + 1")
        return
    diffs = []
    for a, b in zip(runs, runs[1:]):
        diffs.append(max(float(np.max(np.abs(a[nm] - b[nm]))) for nm in names))
    scale = max(float(np.max(np.abs(runs[-1][nm]))) for nm in names) or 1.0
    if diffs[-1] <= 1e-12 * scale:
        report.add("convergence", PASS, f"results do not depend on {key} (exact or adaptive solution)", diffs=diffs)
        return
    orders = [math.log2(d1 / d2) if d2 > 0 else float("inf") for d1, d2 in zip(diffs, diffs[1:])]
    observed = orders[-1]
    ok = observed >= min_order
    report.add("convergence", PASS if ok else FAIL,
               f"observed order {observed:.2f} when halving {key} (differences {', '.join(f'{d:.2e}' for d in diffs)})"
               + ("" if ok else f"; expected >= {min_order}: the solution does not converge"),
               orders=orders, diffs=diffs)


# ---------------------------------------------------------------------------- isolated checks of source code
_RUNNER = textwrap.dedent('''
    import json, sys

    def _run():
        token = sys.stdin.readline().strip()   # one-time token, read before any model code is imported
        sys.path.insert(0, {model_dir!r})
        for p in {extra_path!r}:
            sys.path.insert(0, p)
        from simulator.checker import check_model
        with open({source_path!r}, encoding="utf-8") as f:
            source = f.read()
        with open({config_path!r}, encoding="utf-8") as f:
            config = json.load(f)
        report = check_model({module!r} + "." + {class_name!r}, config, source=source, **{options!r})
        sys.stdout.write("\\n@@REPORT@@" + token + json.dumps(report.to_dict(), default=float) + "\\n")

    _run()
''')


def _child_environment(tmp: str) -> Dict[str, str]:
    """Minimal environment for the checking process: no API keys or other secrets."""
    keep = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "LANG", "LC_ALL", "VIRTUAL_ENV")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update({"HOME": tmp, "USERPROFILE": tmp, "TMP": tmp, "TEMP": tmp, "TMPDIR": tmp,
                "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def _validated_report(stdout: str, token: str) -> Optional[CheckReport]:
    """The report printed with the right token, if it is complete; else None."""
    marker = "@@REPORT@@" + token
    lines = [l for l in stdout.splitlines() if l.startswith(marker)]
    if len(lines) != 1:
        return None
    try:
        report = CheckReport.from_dict(json.loads(lines[0][len(marker):]))
    except (ValueError, KeyError, TypeError):
        return None
    names = {r.name for r in report.results}
    runs = report.get("runs")
    if runs is None or "lint" not in names:
        return None
    if runs.status == PASS and not set(EXPECTED_CHECKS) <= names:
        return None
    return report


def check_source(source: str, class_name: str, config: Dict[str, Any], timeout: float = 120.0,
                 **options: Any) -> CheckReport:
    """Check model source code in a separate Python process (time-limited).

    The code is linted first; code that fails the lint is never executed.
    """
    report = CheckReport(f"<generated>.{class_name}")
    lint = lint_source(source)
    if any(r.status == FAIL for r in lint):
        report.results.extend(lint)
        report.add("runs", SKIP, "not executed: the code failed the static lint")
        return report
    simulator_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory(prefix="sds_check_") as tmp:
        module = "generated_model"
        source_path = os.path.join(tmp, f"{module}.py")
        config_path = os.path.join(tmp, "config.json")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, default=str)
        runner = _RUNNER.format(model_dir=tmp, extra_path=[simulator_root], source_path=source_path,
                                config_path=config_path, module=module, class_name=class_name, options=options)
        token = secrets.token_hex(16)
        try:
            proc = subprocess.run([sys.executable, "-E", "-c", runner], input=token + "\n", capture_output=True,
                                  text=True, timeout=timeout, cwd=tmp, env=_child_environment(tmp))
        except subprocess.TimeoutExpired:
            report.results.extend(lint)
            report.add("runs", FAIL, f"timed out after {timeout:.0f} s")
            return report
    report_out = _validated_report(proc.stdout, token)
    if report_out is None:
        report.results.extend(lint)
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        report.add("runs", FAIL, "the check process did not return a valid, complete report: " + " | ".join(tail))
        return report
    return report_out


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m simulator.checker module.Class config.yaml [--markdown out.md] [--json out.json]``"""
    import argparse
    from .config import load_config
    parser = argparse.ArgumentParser(description="Automatic checks for an ExperimentLogic model.")
    parser.add_argument("model", help="dotted path, e.g. experiments.oscillator.logic.OscillatorExperiment")
    parser.add_argument("config", help="YAML config to check the model with")
    parser.add_argument("--markdown")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    report = check_model(args.model, load_config(args.config))
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, default=float)
    print(report.to_markdown())
    return 0 if report.accepted else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
