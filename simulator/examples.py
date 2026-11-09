# simulator/examples.py
"""Example registry: discover examples from their manifests and run them the same way.

Every folder under ``examples/`` with an ``example.yaml`` is an example. The manifest says what
the example is (title, summary, fields, level, runtime), how to run it (entry module, the
argument that sets the output folder, the arguments of its quick mode) and what a user may change
(parameters with type, default, range or choices, help text). Tools - the ``sds`` command line,
a UI, CI - read manifests instead of knowing examples, so a new example appears everywhere once
its manifest exists.

Manifest fields::

    title: Law discovery from noisy data          # required
    summary: One sentence.                        # required
    entry: examples.example_10_law_discovery.run_experiment   # required: module run with -m
    fields: [physics, mathematics]                # physics, mathematics, geometry, social,
                                                  # economics, methods, ai
    level: research                               # student | research
    runtime: {full: 30 s, quick: 10 s}
    extras: [parquet]                             # pip extras it needs (beyond the core)
    output_arg: --output_dir                      # null: the run's folder becomes the working dir
    quick_args: [--replicates, "1"]
    params:
      - {name: suite_seed, arg: --suite-seed, type: int, default: null, help: ...}
      - {name: export, arg: --export, type: flag, default: false, help: ...}
      - {name: model, arg: --model, type: choice, choices: [a, b], default: a}
    outputs:                                      # what to show after a run (globs)
      - {path: report.md, kind: markdown}
      - {path: "**/bundle/figure.png", kind: image}

:func:`run_example` runs an example in a subprocess (crashes and hangs stay contained; output is
streamed to ``run.log``) and writes ``run.json`` with the exact command, parameters and versions,
so every run - from the command line or a UI - can be repeated as a script.
"""
from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from glob import glob
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

FIELDS = ("physics", "mathematics", "geometry", "social", "economics", "methods", "ai")
LEVELS = ("student", "research")
PARAM_TYPES = ("int", "float", "str", "flag", "choice")
OUTPUT_KINDS = ("markdown", "image", "html", "table", "json", "text", "folder")


def default_examples_dir() -> Optional[str]:
    """``$SDS_EXAMPLES``, else ``examples/`` next to the package (a source checkout)."""
    env = os.environ.get("SDS_EXAMPLES")
    if env:
        return os.path.abspath(env)
    candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    return candidate if os.path.isdir(candidate) else None


def project_root(examples_dir: str) -> str:
    return os.path.dirname(os.path.abspath(examples_dir))


@dataclass
class Param:
    name: str
    arg: str
    type: str
    default: Any = None
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[List[Any]] = None

    def coerce(self, value: Any) -> Any:
        """Convert a value (e.g. a string from ``--set name=value``) and check it."""
        if value is None:
            return None
        if self.type == "flag":
            if isinstance(value, str):
                low = value.strip().lower()
                if low not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
                    raise ValueError(f"{self.name}: expected true/false, got {value!r}")
                return low in ("1", "true", "yes", "on")
            return bool(value)
        if self.type in ("int", "float") and isinstance(value, bool):
            raise ValueError(f"{self.name}: expected a number, got {value!r}")
        if self.type == "int":
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{self.name}: expected an integer, got {value!r}") from None
            if not number.is_integer():
                raise ValueError(f"{self.name}: expected an integer, got {value!r}")
            v = int(number)
        elif self.type == "float":
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{self.name}: expected a number, got {value!r}") from None
        elif self.type == "choice":
            matches = [c for c in self.choices or [] if str(c) == str(value)]
            if not matches:
                raise ValueError(f"{self.name}: {value!r} is not one of {self.choices}")
            return matches[0]
        else:
            return str(value)
        if self.min is not None and v < self.min or self.max is not None and v > self.max:
            raise ValueError(f"{self.name}: {v} is outside [{self.min}, {self.max}]")
        return v

    def to_args(self, value: Any) -> List[str]:
        if value is None:
            return []
        if self.type == "flag":
            return [self.arg] if value else []
        return [self.arg, str(value)]


@dataclass
class Example:
    name: str
    path: str
    title: str
    summary: str
    entry: str
    fields: List[str] = field(default_factory=list)
    level: str = "student"
    runtime: Dict[str, str] = field(default_factory=dict)
    extras: List[str] = field(default_factory=list)
    output_arg: Optional[str] = "--output_dir"
    quick_args: List[str] = field(default_factory=list)
    params: List[Param] = field(default_factory=list)
    outputs: List[Dict[str, str]] = field(default_factory=list)

    def param(self, name: str) -> Param:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"{self.name} has no parameter '{name}' (has: {[p.name for p in self.params]})")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ManifestError(ValueError):
    pass


def load_manifest(folder: str) -> Example:
    """Read and validate ``<folder>/example.yaml``."""
    path = os.path.join(folder, "example.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ManifestError(f"{path}: not valid YAML ({e})") from None
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: the manifest must be a mapping of keys to values")
    name = os.path.basename(os.path.normpath(folder))
    problems = []
    for key, kind in (("fields", list), ("extras", list), ("quick_args", list), ("params", list),
                      ("outputs", list), ("runtime", dict)):
        if data.get(key) is not None and not isinstance(data[key], kind):
            problems.append(f"'{key}' must be a {'list' if kind is list else 'mapping'}")
            data[key] = None
    if data.get("output_arg") is not None and not (isinstance(data["output_arg"], str)
                                                    and data["output_arg"].startswith("-")):
        problems.append("'output_arg' must be an option such as --output_dir, or null")
    for key in ("title", "summary", "entry"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            problems.append(f"'{key}' is required")
    known = {"title", "summary", "entry", "fields", "level", "runtime", "extras", "output_arg", "quick_args",
             "params", "outputs"}
    problems += [f"unknown key '{k}'" for k in data if k not in known]
    fields_ = data.get("fields") or []
    problems += [f"unknown field '{f}' (use {FIELDS})" for f in fields_ if f not in FIELDS]
    if data.get("level", "student") not in LEVELS:
        problems.append(f"level must be one of {LEVELS}")
    params = []
    for i, raw in enumerate(data.get("params") or []):
        if not isinstance(raw, dict) or not raw.get("name") or not raw.get("arg"):
            problems.append(f"params[{i}] needs 'name' and 'arg'")
            continue
        if not str(raw["arg"]).startswith("-"):
            problems.append(f"params[{i}]: 'arg' must be an option such as --n")
            continue
        unknown = set(raw) - {"name", "arg", "type", "default", "help", "min", "max", "choices"}
        if unknown:
            problems.append(f"params[{i}]: unknown key(s) {sorted(unknown)}")
            continue
        p = Param(**{"type": "str", **raw})
        if p.type not in PARAM_TYPES:
            problems.append(f"params[{i}]: type must be one of {PARAM_TYPES}")
        elif p.type == "choice" and not p.choices:
            problems.append(f"params[{i}]: a choice needs 'choices'")
        else:
            try:
                p.default = p.coerce(p.default)
            except (TypeError, ValueError) as e:
                problems.append(f"params[{i}] default: {e}")
        params.append(p)
    names = [p.name for p in params]
    if len(set(names)) != len(names):
        problems.append("parameter names must be unique")
    outputs = data.get("outputs") or []
    for i, o in enumerate(outputs):
        if not isinstance(o, dict) or not o.get("path") or o.get("kind", "text") not in OUTPUT_KINDS:
            problems.append(f"outputs[{i}] needs 'path' and a kind in {OUTPUT_KINDS}")
    module_file = os.path.join(project_root(os.path.dirname(folder)), *str(data.get("entry", "")).split(".")) + ".py"
    if data.get("entry") and not os.path.exists(module_file):
        problems.append(f"entry module not found: {data.get('entry')}")
    if problems:
        raise ManifestError(f"{path}: " + "; ".join(problems))
    return Example(name=name, path=os.path.abspath(folder), title=data["title"].strip(),
                   summary=data["summary"].strip(), entry=data["entry"], fields=list(fields_),
                   level=data.get("level", "student"), runtime=dict(data.get("runtime") or {}),
                   extras=list(data.get("extras") or []), output_arg=data.get("output_arg", "--output_dir"),
                   quick_args=[str(a) for a in data.get("quick_args") or []], params=params,
                   outputs=[{"path": o["path"], "kind": o.get("kind", "text")} for o in outputs])


def _natural(name: str):
    """Sort key: example_2 before example_10."""
    import re
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def discover(examples_dir: Optional[str] = None, strict: bool = False) -> Dict[str, Example]:
    """All examples with a valid manifest, by folder name. ``strict``: raise on an invalid
    manifest instead of skipping it (and report folders without one via :func:`check_all`)."""
    examples_dir = examples_dir or default_examples_dir()
    if not examples_dir or not os.path.isdir(examples_dir):
        raise FileNotFoundError("examples folder not found: run from a source checkout or set SDS_EXAMPLES")
    found: Dict[str, Example] = {}
    for name in sorted(os.listdir(examples_dir), key=_natural):
        folder = os.path.join(examples_dir, name)
        if not os.path.isfile(os.path.join(folder, "example.yaml")):
            continue
        try:
            found[name] = load_manifest(folder)
        except ManifestError:
            if strict:
                raise
    return found


def check_all(examples_dir: Optional[str] = None) -> List[str]:
    """Problems with the examples folder: invalid manifests and example folders without one."""
    examples_dir = examples_dir or default_examples_dir()
    if not examples_dir or not os.path.isdir(examples_dir):
        raise FileNotFoundError("examples folder not found: run from a source checkout or set SDS_EXAMPLES")
    problems = []
    for name in sorted(os.listdir(examples_dir)):
        folder = os.path.join(examples_dir, name)
        if not os.path.isdir(folder) or name.startswith((".", "_")):
            continue
        if not os.path.isfile(os.path.join(folder, "example.yaml")):
            if os.path.isfile(os.path.join(folder, "run_experiment.py")):
                problems.append(f"{name}: no example.yaml")
            continue
        try:
            load_manifest(folder)
        except ManifestError as e:
            problems.append(str(e))
    return problems


@dataclass
class ExampleRun:
    name: str
    command: List[str]
    output_dir: str
    returncode: Optional[int]
    seconds: float
    log_path: str
    outputs: List[Dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    cwd: str = ""

    @property
    def command_line(self) -> str:
        """The command, quoted for this system's shell (cmd on Windows, POSIX shells elsewhere)."""
        if os.name == "nt":
            return subprocess.list2cmdline(self.command)
        return " ".join(shlex.quote(c) for c in self.command)


def _quick_settings(example: Example) -> Dict[str, Any]:
    """Parameter values implied by the quick arguments (``--n 3`` -> ``{"n": 3}``)."""
    by_arg = {p.arg: p for p in example.params}
    out, i, args = {}, 0, example.quick_args
    while i < len(args):
        p = by_arg.get(args[i])
        if p is not None and p.type != "flag" and i + 1 < len(args):
            out[p.name] = p.coerce(args[i + 1])
            i += 2
            continue
        if p is not None and p.type == "flag":
            out[p.name] = True
        i += 1
    return out


def effective_values(example: Example, values: Mapping[str, Any], quick: bool) -> Dict[str, Any]:
    """What each parameter will be: the user's value, else the quick-mode value, else the default."""
    quick_values = _quick_settings(example) if quick else {}
    return {p.name: p.coerce(values[p.name]) if p.name in values else quick_values.get(p.name, p.default)
            for p in example.params}


def build_command(example: Example, values: Mapping[str, Any], quick: bool, output_dir: str,
                  python: str = sys.executable) -> List[str]:
    """The command line: user values win over quick-mode values, which win over defaults."""
    cmd = [python, "-m", example.entry]
    if example.output_arg:
        cmd += [example.output_arg, output_dir]
    effective = effective_values(example, values, quick)
    for p in example.params:
        cmd += p.to_args(effective[p.name])
    if quick:  # quick arguments that are not parameters (e.g. --quick itself)
        by_arg = {p.arg: p for p in example.params}
        i, args = 0, example.quick_args
        while i < len(args):
            p = by_arg.get(args[i])
            if p is not None:
                i += 1 if p.type == "flag" else 2
                continue
            cmd.append(args[i])
            i += 1
    return cmd


def find_outputs(example: Example, output_dir: str) -> List[Dict[str, str]]:
    found, seen = [], set()
    for spec in example.outputs:
        for path in sorted(glob(os.path.join(output_dir, spec["path"]), recursive=True)):
            if path not in seen:  # a file matched by two patterns is listed once, under the first
                seen.add(path)
                found.append({"path": path, "kind": spec["kind"]})
    return found


def _kill_tree(proc: "subprocess.Popen") -> None:
    """Stop a process and everything it started (workers, helper processes)."""
    try:
        import psutil
        try:
            children = psutil.Process(proc.pid).children(recursive=True)
        except psutil.Error:
            children = []
        for child in children:
            try:
                child.kill()
            except psutil.Error:
                pass
    except ImportError:  # pragma: no cover - psutil is a core dependency
        pass
    proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


RUN_LOG = "run.log"
RUN_RECORD = "run.json"
RUNNER_FILES = frozenset({RUN_LOG, RUN_RECORD})


def prepare_output_dir(path: str) -> str:
    """Create ``path`` or clear an example's earlier results from it, and return the absolute path.

    Examples call this instead of deleting the whole folder. When ``sds examples run`` or the UI
    starts an example, the folder is the run folder and already holds the runner's ``run.log``
    (open for writing) and ``run.json``: those are kept. Deleting them fails on Windows (the file
    is in use) and silently loses the log elsewhere."""
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        if name in RUNNER_FILES:
            continue
        full = os.path.join(path, name)
        if os.path.isdir(full) and not os.path.islink(full):
            shutil.rmtree(full)
        else:
            os.remove(full)
    return path


def run_example(example: Example, values: Optional[Mapping[str, Any]] = None, quick: bool = False,
                output_dir: Optional[str] = None, timeout: Optional[float] = None,
                on_line: Optional[Callable[[str], None]] = None, python: str = sys.executable) -> ExampleRun:
    """Run ``example`` in a subprocess and record the run.

    ``values`` override parameter defaults (by name; strings are converted). ``output_dir``
    defaults to ``sds_runs/<example>/<timestamp>``. Each output line goes to ``run.log`` and
    to ``on_line`` (for a UI). ``run.json`` records the command, parameters, versions, time and
    return code.
    """
    values = dict(values or {})
    for name in values:
        example.param(name)  # unknown names fail early
    import uuid
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]  # unique even within a second
    output_dir = os.path.abspath(output_dir or os.path.join("sds_runs", example.name, stamp))
    os.makedirs(output_dir, exist_ok=True)
    root = project_root(os.path.dirname(example.path))
    cmd = build_command(example, values, quick, output_dir, python)
    env = dict(os.environ)
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"  # Windows: a piped child would otherwise write cp1252 and fail on non-ASCII text
    env["PYTHONIOENCODING"] = "utf-8"
    cwd = root if example.output_arg else output_dir
    log_path = os.path.join(output_dir, RUN_LOG)
    start = time.perf_counter()
    returncode: Optional[int] = None
    state = {"open": True}
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")

        def pump():  # a reader thread, so a silent hang cannot block the timeout
            assert proc.stdout is not None
            for line in proc.stdout:
                if not state["open"]:
                    continue  # keep draining so the child never blocks on a full pipe
                log.write(line)
                log.flush()
                if on_line is not None:
                    try:
                        on_line(line.rstrip("\n"))
                    except Exception:  # a failing callback (e.g. a console encoding) must not stall the run
                        pass

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            returncode = None
        seconds = time.perf_counter() - start
        reader.join(timeout=5)
        if returncode is None:
            log.write(f"\n[sds] stopped after {timeout:g} s\n")
        state["open"] = False
    run = ExampleRun(example.name, cmd, output_dir, returncode, round(seconds, 2), log_path,
                     find_outputs(example, output_dir), cwd=cwd)
    from . import __version__ as sds_version
    record = {"example": example.name, "title": example.title, "quick": quick,
              "parameters": effective_values(example, values, quick),
              "command": cmd, "command_line": run.command_line, "cwd": cwd, "pythonpath": root,
              "returncode": returncode,
              "seconds": run.seconds, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "simulator_version": sds_version, "python": platform.python_version(),
              "platform": platform.platform(), "outputs": [os.path.relpath(o["path"], output_dir).replace(os.sep, "/")
                          for o in run.outputs]}
    with open(os.path.join(output_dir, RUN_RECORD), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    return run
