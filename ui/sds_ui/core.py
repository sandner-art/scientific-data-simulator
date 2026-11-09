# sds_ui/core.py
"""Everything the UI needs from the simulator, as plain functions (no Streamlit here).

The UI uses only the simulator's public functions - the example manifests and runner
(:mod:`simulator.examples`), the doctor (:mod:`simulator.doctor`) and the version - so whatever
it does can also be done from the command line, and a run started here is recorded exactly
like one started with ``sds examples run``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

from simulator import __version__ as SIMULATOR_VERSION
from simulator.doctor import EXTRAS
from simulator.examples import Example, ExampleRun, Param, discover, effective_values, run_example

UI_SETTINGS = ("SDS_EXAMPLES", "SDS_RUNS")


def runs_dir() -> str:
    """Where runs started from the UI go (``$SDS_RUNS``, default ``sds_runs``)."""
    return os.path.abspath(os.environ.get("SDS_RUNS") or "sds_runs")


def load_examples(examples_dir: Optional[str] = None) -> Dict[str, Example]:
    return discover(examples_dir or os.environ.get("SDS_EXAMPLES") or None)


def missing_extras(example: Example) -> List[str]:
    """Extras the example needs that are not installed (checked by importing their modules)."""
    missing = []
    for extra in example.extras:
        modules = EXTRAS.get(extra, ((),))[0]
        if any(importlib.util.find_spec(m) is None for m in modules):
            missing.append(extra)
    return missing


def filter_examples(examples: Mapping[str, Example], text: str = "", fields: Optional[List[str]] = None,
                    level: str = "any") -> List[Example]:
    text = text.strip().lower()
    out = []
    for e in examples.values():
        if text and text not in f"{e.name} {e.title} {e.summary}".lower():
            continue
        if fields and not set(fields) & set(e.fields):
            continue
        if level != "any" and e.level != level:
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------- parameter forms
@dataclass
class WidgetSpec:
    """How to show one parameter: ``kind`` is number, text, checkbox or select."""
    name: str
    label: str
    kind: str
    default: Any
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: List[Any] = field(default_factory=list)
    optional: bool = False  # no default: an empty field keeps the example's own behaviour


def widget_spec(p: Param) -> WidgetSpec:
    label = p.name.replace("_", " ")
    if p.type == "flag":
        return WidgetSpec(p.name, label, "checkbox", bool(p.default), p.help)
    if p.type == "choice":
        return WidgetSpec(p.name, label, "select", p.default, p.help, options=list(p.choices or []))
    if p.type in ("int", "float") and p.default is not None:
        cast = int if p.type == "int" else float  # the widget's number type follows the parameter's
        return WidgetSpec(p.name, label, "number", cast(p.default), p.help,
                          None if p.min is None else cast(p.min), None if p.max is None else cast(p.max),
                          1 if p.type == "int" else None)
    return WidgetSpec(p.name, label, "text", "" if p.default is None else str(p.default), p.help,
                      optional=p.default is None)


def form_values(example: Example, raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Values to pass to the runner: only those that differ from the defaults, converted and
    checked by the manifest's own rules (raises ValueError with the parameter's name)."""
    values = {}
    for p in example.params:
        if p.name not in raw:
            continue
        value = raw[p.name]
        if isinstance(value, str) and value.strip() == "" and p.default is None:
            continue
        value = p.coerce(value)
        if value != p.default:
            values[p.name] = value
    return values


# ---------------------------------------------------------------------------- running
@dataclass
class LiveRun:
    """An example running in a background thread; the page polls ``lines`` and ``done``."""
    example: Example
    values: Dict[str, Any]
    quick: bool
    output_dir: str
    lines: List[str] = field(default_factory=list)
    result: Optional[ExampleRun] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None

    @property
    def done(self) -> bool:
        return self.thread is not None and not self.thread.is_alive()


def run_folder_name() -> str:
    """Time stamp plus a random suffix: two runs started in the same second (two browser tabs)
    never share a folder."""
    import uuid
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def start_run(example: Example, values: Mapping[str, Any], quick: bool, timeout: Optional[float] = None,
              output_dir: Optional[str] = None) -> LiveRun:
    output_dir = output_dir or os.path.join(runs_dir(), example.name, run_folder_name())
    live = LiveRun(example, dict(values), quick, os.path.abspath(output_dir))

    def target():
        try:
            live.result = run_example(example, live.values, quick=quick, output_dir=live.output_dir,
                                      timeout=timeout, on_line=live.lines.append)
        except Exception as e:  # shown on the page, never lost
            live.error = f"{type(e).__name__}: {e}"

    live.thread = threading.Thread(target=target, daemon=True)
    live.thread.start()
    return live


# ---------------------------------------------------------------------------- past runs
@dataclass
class RunRecord:
    folder: str
    example: str
    title: str
    finished: str
    quick: bool
    returncode: Optional[int]
    seconds: float
    parameters: Dict[str, Any]
    command_line: str
    outputs: List[str]
    cwd: str = ""
    pythonpath: str = ""

    @property
    def status(self) -> str:
        if self.returncode == 0:
            return "ok"
        return "timed out" if self.returncode is None else f"failed ({self.returncode})"


def list_runs(folder: Optional[str] = None) -> List[RunRecord]:
    """Runs recorded under ``folder`` (any depth), newest first. Unreadable or malformed
    ``run.json`` files are skipped, and runs still in progress (a ``run.log`` without
    ``run.json``) are not searched."""
    folder = folder or runs_dir()
    records = []
    if not os.path.isdir(folder):
        return records
    for root, dirs, files in os.walk(folder):
        if "run.json" not in files:
            if "run.log" in files:
                dirs[:] = []  # a run in progress: its outputs are not runs
            continue
        dirs[:] = []  # a run's own outputs are not searched for more runs
        try:
            with open(os.path.join(root, "run.json"), encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            outputs = data.get("outputs") or []
            records.append(RunRecord(
                root, str(data.get("example", "?")), str(data.get("title", "")), str(data.get("finished") or ""),
                bool(data.get("quick")), data.get("returncode") if isinstance(data.get("returncode"), int) else None,
                float(data.get("seconds") or 0),
                data.get("parameters") if isinstance(data.get("parameters"), dict) else {},
                str(data.get("command_line", "")),
                [os.path.join(root, str(o)) for o in outputs] if isinstance(outputs, list) else [],
                cwd=str(data.get("cwd") or ""), pythonpath=str(data.get("pythonpath") or "")))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(records, key=lambda r: r.finished, reverse=True)


def repeat_command(cwd: str, pythonpath: str, command_line: str, windows: Optional[bool] = None) -> str:
    """Shell lines that repeat a run from any folder, for this system's shell."""
    windows = os.name == "nt" if windows is None else windows
    if windows:
        lines = [f'cd /d "{cwd}"'] if cwd else []
        if pythonpath:
            lines.append(f'set "PYTHONPATH={pythonpath}"')
        return "\n".join(lines + [command_line])
    import shlex
    prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
    env = f"PYTHONPATH={shlex.quote(pythonpath)} " if pythonpath else ""
    return prefix + env + command_line


def outputs_of(record: RunRecord, examples: Mapping[str, Example]) -> List[Dict[str, str]]:
    """The outputs of a past run, with the kinds its example's manifest gives them."""
    from simulator.examples import find_outputs
    example = examples.get(record.example)
    if example is not None:
        return find_outputs(example, record.folder)
    return [{"path": p, "kind": guess_kind(p)} for p in record.outputs]


def figures_in(folder: str, limit: int = 12) -> List[Dict[str, str]]:
    """Figures anywhere in a run folder (PNG, SVG, JPEG), for runs whose manifest lists none."""
    from glob import glob
    found: List[str] = []
    for pattern in ("*.png", "*.svg", "*.jpg", "*.jpeg"):
        found += glob(os.path.join(folder, "**", pattern), recursive=True)
    return [{"path": p, "kind": "image"} for p in sorted(found)[:limit]]


def citation() -> Dict[str, str]:
    """Title, authors, version and DOI from ``CITATION.cff`` in a source checkout (empty if absent)."""
    import simulator
    import yaml
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(simulator.__file__))), "CITATION.cff")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    authors = ", ".join(" ".join(filter(None, [a.get("given-names"), a.get("family-names")]))
                        for a in data.get("authors") or [] if isinstance(a, dict))
    return {"title": str(data.get("title", "")), "authors": authors, "version": str(data.get("version", "")),
            "doi": str(data.get("doi", "") or ""), "url": str(data.get("repository-code", "") or "")}


def guess_kind(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".md": "markdown", ".png": "image", ".jpg": "image", ".jpeg": "image", ".svg": "image",
            ".html": "html", ".csv": "table", ".json": "json"}.get(ext, "text")


def tail(path: str, n: int = 200, max_bytes: int = 256 * 1024) -> str:
    """The last ``n`` lines, reading at most ``max_bytes`` from the end of the file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
        return "\n".join(text.splitlines()[-n:])
    except OSError:
        return ""
