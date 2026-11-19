"""Run the pre-registered Gaia DR4 analysis (10.5281/zenodo.24370371) on fetched data.

Reads the folder ``tools/gaia_dr4_fetch.py`` wrote, checks the record's hash, runs H1-H5 and the
exploratory analyses with the registered method, and writes ``report.json``, ``report.md``,
``injections.csv`` and ``controls.csv`` to ``--out``. H1 (20 noise replicates per control) and the
injections dominate the time; use ``--jobs`` (default: all cores).

    python tools/gaia_dr4_analyze.py --record preregistration_gaia_dr4.json --data gaia_dr4_data --out gaia_dr4_report
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None):
    import pandas as pd
    from scidatasim.experiments.astrometry import dr4, dr4_analysis
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--record", required=True)
    parser.add_argument("--data", default="gaia_dr4_data")
    parser.add_argument("--out", default="gaia_dr4_report")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--replicates", type=int, default=20, help="H1 noise replicates per control (registered: 20)")
    args = parser.parse_args(argv)
    record = dr4.load_record(args.record)
    report = dr4_analysis.analyze(record, args.data, jobs=args.jobs, replicates=args.replicates)
    if args.replicates != 20:
        report.setdefault("deviations", []).append(f"H1 replicates {args.replicates} instead of the registered 20")
    os.makedirs(args.out, exist_ok=True)
    for part, prefix in ((report, ""), (report.get("fallback_rescaled"), "rescaled_")):
        if part:
            pd.DataFrame(part.pop("_injections")).to_csv(os.path.join(args.out, f"{prefix}injections.csv"), index=False)
            pd.DataFrame(part.pop("_controls")).to_csv(os.path.join(args.out, f"{prefix}controls.csv"), index=False)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=float)
    text = dr4_analysis.markdown(report)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
