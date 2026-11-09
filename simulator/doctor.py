# simulator/doctor.py
"""Check that the simulator is installed and working, and say how to fix what is not.

    python -m simulator.doctor            # or: sds doctor
    python -m simulator.doctor --json
    python -m simulator.doctor --plots    # also try a real static plot export (slower)

Checks: Python version; core packages; optional extras (each with the pip command that adds it);
a Chrome install for static plotly export (kaleido 1.x needs one); a writable output folder; the
examples folder and its manifests; and a short test simulation. Exit status 1 if anything the
core needs is missing or the test simulation fails; optional parts only warn.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from typing import List, Optional

OK, WARN, FAIL = "ok", "warn", "fail"
MIN_PYTHON = (3, 9)
CORE = ("numpy", "scipy", "pandas", "yaml", "dotenv", "pydantic", "psutil", "matplotlib", "plotly")
#: extra -> (modules it provides, what it is for)
EXTRAS = {
    "static-plots": (("kaleido",), "PNG/SVG/PDF export of plotly figures"),
    "parquet": (("pyarrow",), "Parquet storage and the Hugging Face export"),
    "hdf5": (("h5py",), "HDF5 storage and The Well export"),
    "units": (("pint",), "physical units in configs"),
    "validate": (("mlcroissant", "datasets"), "validating exported datasets"),
}
CHROME_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge")
CHROME_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _version(module: str) -> Optional[str]:
    try:
        mod = importlib.import_module(module)
    except Exception:
        return None
    return str(getattr(mod, "__version__", "installed"))


def _find_chrome() -> Optional[str]:
    for env in ("BROWSER_PATH", "CHROME_PATH"):
        if os.environ.get(env) and os.path.exists(os.environ[env]):
            return os.environ[env]
    for name in CHROME_NAMES:
        path = shutil.which(name)
        if path:
            return path
    local = os.environ.get("LOCALAPPDATA")
    paths = list(CHROME_PATHS) + ([os.path.join(local, "Google", "Chrome", "Application", "chrome.exe")]
                                  if local else [])
    for path in paths:
        if os.path.exists(path):
            return path
    try:  # where kaleido looks, including the copy plotly_get_chrome downloads
        from choreographer.browsers.chromium import Chromium
        path = Chromium.find_browser(skip_local=False)
        if path:
            return str(path)
    except Exception:  # older or newer choreographer: fall back to "not found"
        pass
    return None


def run_checks(output_dir: str = "experiments_output", plots: bool = False) -> List[Check]:
    checks: List[Check] = []
    py = sys.version_info
    checks.append(Check("python", OK if py[:2] >= MIN_PYTHON else FAIL,
                        f"{platform.python_version()} ({sys.executable})",
                        "" if py[:2] >= MIN_PYTHON else "install Python 3.9 or newer (python.org, uv or your package manager)"))
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    checks.append(Check("environment", OK if in_venv else WARN,
                        "virtual environment" if in_venv else "not in a virtual environment",
                        "" if in_venv else "use the install script, or: python -m venv .venv"))
    from . import __version__
    checks.append(Check("simulator", OK, f"version {__version__}, {platform.system()} {platform.machine()}"))

    missing = [m for m in CORE if _version(m) is None]
    checks.append(Check("core packages", FAIL if missing else OK,
                        "missing: " + ", ".join(missing) if missing else ", ".join(f"{m} {_version(m)}" for m in CORE[:4]) + ", ...",
                        'pip install -e .' if missing else ""))
    for extra, (modules, purpose) in EXTRAS.items():
        absent = [m for m in modules if _version(m) is None]
        checks.append(Check(f"extra: {extra}", WARN if absent else OK,
                            f"{purpose}: " + ("not installed" if absent else "installed"),
                            f'pip install -e ".[{extra}]"' if absent else ""))
    if _version("kaleido") is not None:
        chrome = _find_chrome()
        checks.append(Check("chrome for static plots", OK if chrome else WARN,
                            chrome or "no Chrome/Chromium found (interactive HTML plots still work)",
                            "" if chrome else "install Chrome, or run: plotly_get_chrome"))

    created = not os.path.exists(output_dir)
    try:
        os.makedirs(output_dir, exist_ok=True)
        probe = os.path.join(output_dir, ".sds_write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        checks.append(Check("output folder", OK, os.path.abspath(output_dir) + " is writable"))
        if created:  # leave no trace of the check
            os.rmdir(output_dir)
    except OSError as e:
        checks.append(Check("output folder", FAIL, f"{output_dir}: {e}", "choose a writable folder"))

    from .examples import check_all, default_examples_dir, discover
    folder = default_examples_dir()
    if folder is None:
        checks.append(Check("examples", WARN, "examples folder not found (installed without the source tree)",
                            "clone the repository, or set SDS_EXAMPLES to its examples folder"))
    else:
        problems = check_all(folder)
        found = discover(folder)
        checks.append(Check("examples", WARN if problems else OK,
                            f"{len(found)} examples in {folder}" + (f"; {len(problems)} problem(s): {problems[0]}"
                                                                     if problems else ""),
                            "sds examples check" if problems else ""))

    with tempfile.TemporaryDirectory() as tmp:
        try:
            from .cli import _ensure_models_importable
            _ensure_models_importable()
            from .engine import simulate
            sim = simulate({"experiment_type": "experiments.example_random_walk.logic.RandomWalkExperiment",
                            "n_steps": 50, "step_size": 1.0, "seed": 1, "generate_plots": False})
            again = simulate({"experiment_type": "experiments.example_random_walk.logic.RandomWalkExperiment",
                              "n_steps": 50, "step_size": 1.0, "seed": 1, "generate_plots": False})
            same = bool((sim.results["position"]["data"] == again.results["position"]["data"]).all())
            checks.append(Check("test simulation", OK if same else FAIL,
                                "random walk, 51 points, reproducible from its seed" if same else
                                "the same seed gave different results",
                                "" if same else "report this as a bug, with the output of: sds doctor --json"))
        except Exception as e:  # report, do not crash
            checks.append(Check("test simulation", FAIL, f"{type(e).__name__}: {e}",
                                "reinstall: pip install -e . (from the repository folder)"))
        if plots and _version("kaleido") is not None:
            try:
                import plotly.graph_objects as go
                go.Figure(go.Scatter(y=[1, 3, 2])).write_image(os.path.join(tmp, "probe.png"))
                checks.append(Check("static plot export", OK, "PNG written"))
            except Exception as e:
                checks.append(Check("static plot export", WARN, f"{type(e).__name__}: {e}",
                                    "install Chrome, or run: plotly_get_chrome"))
    return checks


def report(checks: List[Check]) -> str:
    marks = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}
    width = max(len(c.name) for c in checks)
    lines = []
    for c in checks:
        lines.append(f"{marks[c.status]} {c.name:<{width}}  {c.detail}")
        if c.fix:
            lines.append(f"{'':>7}{'':<{width}}  -> {c.fix}")
    failed = sum(c.status == FAIL for c in checks)
    warned = sum(c.status == WARN for c in checks)
    lines.append("")
    lines.append("Ready." if not failed else f"{failed} problem(s) to fix.")
    if warned and not failed:
        lines[-1] += f" {warned} optional part(s) missing; the core works without them."
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sds doctor", description="Check the installation.")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--plots", action="store_true", help="also try a static plot export")
    parser.add_argument("--output_dir", default="experiments_output")
    args = parser.parse_args(argv)
    checks = run_checks(args.output_dir, plots=args.plots)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        print(report(checks))
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
