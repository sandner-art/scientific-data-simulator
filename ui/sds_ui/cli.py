# sds_ui/cli.py
"""The ``sds-ui`` command (``sds ui`` calls the same)."""
import argparse
from typing import List, Optional

from sds_ui import launch


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sds-ui", description="Start the simulator UI in your browser.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="127.0.0.1",
                        help="127.0.0.1 = this computer only (default). Another address lets anyone who can "
                             "reach it run examples on this computer")
    parser.add_argument("--runs-dir", default=None, help="where runs are saved (default: sds_runs)")
    parser.add_argument("--examples-dir", default=None)
    args = parser.parse_args(argv)
    return launch(args.port, args.address, args.runs_dir, args.examples_dir)
