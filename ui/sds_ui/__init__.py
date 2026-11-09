# sds_ui/__init__.py
"""Streamlit UI for Scientific Data Simulator: examples, runs, results, installation check.

A separate package that depends on the simulator's public functions; the simulator never imports
it. Start it with ``sds ui`` or ``python -m sds_ui``.
"""
import os
import subprocess
import sys
from typing import List, Optional

__version__ = "0.7.0.dev0"

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")


def launch_command(port: int = 8501, address: str = "127.0.0.1") -> List[str]:
    """The streamlit command. It listens on this computer only (127.0.0.1) unless you choose
    otherwise: the UI runs example code with the parameters you enter."""
    return [sys.executable, "-m", "streamlit", "run", APP, "--server.address", address,
            "--server.port", str(port), "--browser.gatherUsageStats", "false",
            "--client.toolbarMode", "minimal"]


def launch(port: int = 8501, address: str = "127.0.0.1", runs_dir: Optional[str] = None,
           examples_dir: Optional[str] = None) -> int:
    env = dict(os.environ)
    if runs_dir:
        env["SDS_RUNS"] = os.path.abspath(runs_dir)
    if examples_dir:
        env["SDS_EXAMPLES"] = os.path.abspath(examples_dir)
    if address not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: listening on {address}. Anyone who can reach this port can run the examples, "
              "fill the disk with results, and (example 12) import Python modules by name. Use only on "
              "a network you trust.")
    print(f"Starting the UI on http://{address}:{port} (Ctrl+C stops it) ...")
    try:
        return subprocess.call(launch_command(port, address), env=env)
    except KeyboardInterrupt:
        return 0
