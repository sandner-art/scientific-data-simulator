# simulator/__init__.py
"""Scientific Data Simulator: reproducible simulated data with recorded ground truth."""
import os as _os
import re as _re


def _find_version() -> str:
    # a source checkout (editable install): pyproject.toml is the truth, installed metadata may lag
    pyproject = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "pyproject.toml")
    try:
        with open(pyproject, encoding="utf-8") as f:
            text = f.read()
        if 'name = "scientific-data-simulator"' in text:
            match = _re.search(r'^version\s*=\s*"([^"]+)"', text, flags=_re.M)
            if match:
                return match.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("scientific-data-simulator")
        except PackageNotFoundError:
            return "unknown"
    except ImportError:  # pragma: no cover
        return "unknown"


__version__ = _find_version()
