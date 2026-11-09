# tests/test_ui_boundary.py
"""The UI is a separate package: the simulator must never depend on it, and both move in lockstep."""
import os
import re

import pytest

from simulator import cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _version(pyproject):
    with open(pyproject, encoding="utf-8") as f:
        return re.search(r'^version\s*=\s*"([^"]+)"', f.read(), flags=re.M).group(1)


def test_simulator_does_not_import_the_ui():
    offenders = []
    for package in ("simulator", "experiments"):
        for folder, _, files in os.walk(os.path.join(ROOT, package)):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                if re.search(r"^\s*(import|from)\s+(sds_ui|streamlit)\b", text, flags=re.M):
                    offenders.append(os.path.relpath(path, ROOT))
                if "sds_ui" in text and path != os.path.join(ROOT, "simulator", "cli.py"):
                    offenders.append(os.path.relpath(path, ROOT))
    assert offenders == []
    with open(os.path.join(ROOT, "simulator", "cli.py"), encoding="utf-8") as f:
        assert 'importlib.import_module("sds_ui.cli")' in f.read()  # optional, by name only


def test_versions_move_together():
    root = _version(os.path.join(ROOT, "pyproject.toml"))
    assert _version(os.path.join(ROOT, "ui", "pyproject.toml")) == root
    with open(os.path.join(ROOT, "ui", "sds_ui", "__init__.py"), encoding="utf-8") as f:
        assert f'__version__ = "{root}"' in f.read()
    with open(os.path.join(ROOT, "ui", "pyproject.toml"), encoding="utf-8") as f:
        assert f'"scientific-data-simulator>={root}"' in f.read()


def test_sds_ui_routing_and_missing_ui(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(cli, "_cmd_ui", lambda args: seen.append(args) or 0)
    assert cli.main(["ui", "--port", "8600", "--runs-dir", "x"]) == 0
    assert cli.main(["--examples-dir", "ex", "ui"]) == 0
    assert seen == [["--port", "8600", "--runs-dir", "x"], ["--examples-dir", "ex"]]
    with pytest.raises(SystemExit):  # no abbreviations: "--examples" is not "--examples-dir"
        cli.main(["--examples", "ex", "ui"])
    monkeypatch.undo()

    import importlib
    real = importlib.import_module

    def no_ui(name, *a, **k):
        if name.startswith("sds_ui"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(importlib, "import_module", no_ui)
    assert cli.main(["ui"]) == 2
    assert "pip install -e ./ui" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["install.sh", "update.sh", "start-ui.sh"])
def test_shell_scripts_know_the_ui_profile(name):
    with open(os.path.join(ROOT, "scripts", name), encoding="utf-8") as f:
        text = f.read()
    if name != "start-ui.sh":
        assert re.search(r"^\s*ui\) EXTRAS=", text, flags=re.M)
    assert "ui" in text
