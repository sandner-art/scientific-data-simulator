"""Calibrate the leak audit on law-discovery releases: false alarms and leaks caught.

Builds quick releases with example 13 (fixed seeds, so the numbers are reproducible), then

* audits each release as built: a release that leaks nothing should pass, so every failure
  is a false alarm;
* plants each kind of leak, one at a time, next to the private answers of every release and
  audits that: the share of planted answers the audit catches.

Run from the repository root::

    python tools/check_audit_calibration.py --releases 40          # builds 40 quick releases
    python tools/check_audit_calibration.py --releases 40 --reuse  # audits releases built before

To compare two versions of the audit on the same releases, build once, then run with
``--reuse`` under each version (for example in two checkouts). The script uses only
``audit_export``, ``_load_private`` and ``_nontrivial``, which every version since 0.6 has.

Writes ``audit_calibration.json`` and ``audit_calibration.md`` to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scidatasim.audit import _load_private, _nontrivial, audit_export  # noqa: E402

#: How each planted leak stores the answers, and whether the audit is meant to catch it.
LEAKS = {
    "float64 copy (JSON)": dict(file="answers.json", transform=lambda v: v, meant=True),
    "float64, 2 ulp off (npy)": dict(file="answers.npy", transform=lambda v: float(
        np.nextafter(np.nextafter(v, np.inf), np.inf)), meant=True),
    "float32 copy (npy)": dict(file="answers.npy", dtype=np.float32, transform=lambda v: v, meant=True),
    "printed, 6 digits (text)": dict(file="notes.md", fmt="{:.6g}", meant=True),
    "printed, 4 digits (text)": dict(file="notes.md", fmt="{:.4g}", meant=True),
    "recomputed, 1e-9 off (npy)": dict(file="answers.npy", transform=lambda v: v * (1 + 1e-9), meant=True),
    "recomputed, 3e-7 off (npy)": dict(file="answers.npy", transform=lambda v: v * (1 + 3e-7), meant=True),
    "printed, 3 digits (text)": dict(file="notes.md", fmt="{:.3g}", meant=False),
    "unit change, x1000 (npy)": dict(file="answers.npy", transform=lambda v: v * 1000, meant=False),
}
FLAG_ATTACKS = ("answer_lookup", "answer_near_match")


def build(release_dir: str, seed: int) -> int:
    cmd = [sys.executable, "-W", "ignore", "-m", "examples.example_13_law_release.run_experiment", "--quick",
           "--suite-seed", str(seed), "--base-seed", str(seed), "--output_dir", release_dir]
    with open(release_dir + ".log", "w") as log:
        return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def answers(private: str, per_release: int, rng: np.random.Generator):
    """Non-trivial numeric answers to plant (at most ``per_release``)."""
    tasks = [t for t in _load_private(private, include_arrays=False).tasks
             if t["answer_type"] == "number" and isinstance(t.get("answer"), (int, float))
             and not isinstance(t.get("answer"), bool) and _nontrivial(float(t["answer"]))]
    pick = rng.choice(len(tasks), size=min(per_release, len(tasks)), replace=False)
    return [float(tasks[i]["answer"]) for i in sorted(pick)]


def plant(folder: str, kind: str, values) -> None:
    spec = LEAKS[kind]
    path = os.path.join(folder, spec["file"])
    if spec["file"].endswith(".md"):
        with open(path, "w", encoding="utf-8") as f:
            f.write("Results\n\n" + "\n".join(f"- value {i}: {spec['fmt'].format(v)}" for i, v in enumerate(values)))
    elif spec["file"].endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"values": [spec["transform"](v) for v in values]}, f)
    else:
        np.save(path, np.array([spec["transform"](v) for v in values], dtype=spec.get("dtype", np.float64)))


def audit_release(release: str, per_release: int, seed: int):
    public, private = os.path.join(release, "public"), os.path.join(release, "private", "runs")
    clean = audit_export(public, private)
    row = {"release": os.path.basename(release), "clean_ok": clean.ok,
           "clean_findings": [(f.attack, f.severity) for f in clean.findings if f.attack in FLAG_ATTACKS],
           "planted": {}}
    values = answers(private, per_release, np.random.default_rng(seed))
    for kind in LEAKS:
        with tempfile.TemporaryDirectory() as tmp:
            plant(tmp, kind, values)
            report = audit_export(tmp, private, check_arrays=False)
        flagged = [f for f in report.findings if f.attack in FLAG_ATTACKS]
        row["planted"][kind] = {
            "planted": len(values),
            "caught_as_copy": report.exposure.get("answer_lookup", {}).get("exposed", 0),
            "fails_release": not report.ok,
            "flagged": sorted({f"{f.attack}:{f.severity}" for f in flagged}),
        }
    return row


def summarise(rows):
    n = len(rows)
    out = {"releases": n,
           "false_alarms": sum(not r["clean_ok"] for r in rows),
           "clean_warnings": sum(bool(r["clean_findings"]) and r["clean_ok"] for r in rows),
           "leaks": {}}
    for kind, spec in LEAKS.items():
        cells = [r["planted"][kind] for r in rows]
        planted = sum(c["planted"] for c in cells)
        out["leaks"][kind] = {
            "meant_to_be_caught": spec["meant"],
            "answers_caught_as_copy": sum(c["caught_as_copy"] for c in cells) / max(planted, 1),
            "releases_failed": sum(c["fails_release"] for c in cells) / max(n, 1),
            "releases_flagged": sum(bool(c["flagged"]) for c in cells) / max(n, 1),
        }
    return out


def markdown(summary, version: str) -> str:
    lines = [f"# Leak audit calibration ({version})", "",
             f"{summary['releases']} quick law-discovery releases (example 13, seeds 1..{summary['releases']}).", "",
             f"- Clean releases refused (false alarms): {summary['false_alarms']}",
             f"- Clean releases passed with a near-match note: {summary['clean_warnings']}", "",
             "| Planted leak | Meant to be caught | Answers caught as copies | Releases failed | Releases flagged |",
             "|---|---|---|---|---|"]
    for kind, s in summary["leaks"].items():
        lines.append(f"| {kind} | {'yes' if s['meant_to_be_caught'] else 'no (blind spot)'} | "
                     f"{s['answers_caught_as_copy']:.0%} | {s['releases_failed']:.0%} | {s['releases_flagged']:.0%} |")
    lines += ["", "*Failed*: a high-severity finding, the release is refused. *Flagged*: any answer_lookup or "
              "answer_near_match finding, at any severity."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--releases", type=int, default=20, help="number of quick releases (seeds 1..N)")
    parser.add_argument("--workdir", default=os.path.join("experiments_output", "audit_calibration"))
    parser.add_argument("--reuse", action="store_true", help="audit releases built before; build none")
    parser.add_argument("--per-release", type=int, default=3, help="answers planted per release")
    parser.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default=None, help="folder for the report (default: the workdir)")
    args = parser.parse_args(argv)
    os.makedirs(args.workdir, exist_ok=True)
    releases = [os.path.abspath(os.path.join(args.workdir, f"release_{s:03d}")) for s in range(1, args.releases + 1)]
    if not args.reuse:
        start = time.time()
        with ThreadPoolExecutor(args.jobs) as pool:
            codes = list(pool.map(build, releases, range(1, args.releases + 1)))
        print(f"built {len(releases)} releases in {time.time() - start:.0f} s; "
              f"{sum(c != 0 for c in codes)} refused by the audit at build time")
    missing = [r for r in releases if not os.path.isdir(os.path.join(r, "private", "runs"))]
    if missing:
        print(f"missing releases (build them first): {', '.join(map(os.path.basename, missing[:5]))}")
        return 1
    rows = [audit_release(r, args.per_release, seed) for seed, r in enumerate(releases, start=1)]
    from scidatasim import __version__
    summary = summarise(rows)
    out = args.out or args.workdir
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "audit_calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"version": __version__, "summary": summary, "releases": rows}, f, indent=2)
    text = markdown(summary, __version__)
    with open(os.path.join(out, "audit_calibration.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
