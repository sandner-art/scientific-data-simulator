# simulator/provenance.py
"""Provenance helpers: installed package versions (via importlib.metadata)."""
from __future__ import annotations

import re
from importlib import metadata
from typing import Dict, Iterable, Optional

DISTRIBUTION_NAME = "scientific-data-simulator"

# Used when the package itself is not installed (e.g. running from a checkout).
_FALLBACK_PACKAGES = ("numpy", "scipy", "pandas", "pyyaml", "python-dotenv", "pydantic",
                      "psutil", "matplotlib", "plotly", "kaleido", "pyarrow", "h5py")


def simulator_version() -> Optional[str]:
    try:
        return metadata.version(DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return None


def _dependency_names() -> Iterable[str]:
    try:
        requirements = metadata.requires(DISTRIBUTION_NAME) or []
    except metadata.PackageNotFoundError:
        return _FALLBACK_PACKAGES
    names = []
    for req in requirements:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
        if match:
            names.append(match.group(1))
    return sorted(set(names) | set(_FALLBACK_PACKAGES))


def collect_software_versions() -> Dict[str, str]:
    """Versions of the simulator and of every (optional) dependency that is installed."""
    versions: Dict[str, str] = {}
    own = simulator_version()
    if own:
        versions[DISTRIBUTION_NAME] = own
    for name in _dependency_names():
        try:
            versions[name.lower()] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def config_sha256(config: Dict) -> str:
    """Hash of the canonical JSON form of a configuration."""
    import hashlib
    import json
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args, cwd: str) -> Optional[str]:
    import subprocess
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def code_provenance(cls: type) -> Dict[str, object]:
    """Identify the exact code of an ExperimentLogic class: module, source
    hash and, when the source lives in a git checkout, commit and dirty flag."""
    import inspect
    import os
    info: Dict[str, object] = {"module": cls.__module__, "class": cls.__qualname__}
    try:
        source_file = inspect.getsourcefile(cls)
    except TypeError:
        source_file = None
    if source_file and os.path.exists(source_file):
        info["source_sha256"] = file_sha256(source_file)
        directory = os.path.dirname(os.path.abspath(source_file))
        commit = _git(["rev-parse", "HEAD"], directory)
        if commit:
            info["git_commit"] = commit
            status = _git(["status", "--porcelain", "--untracked-files=no"], directory)
            info["git_dirty"] = bool(status)
    return info
