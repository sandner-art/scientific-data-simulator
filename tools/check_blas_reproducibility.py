"""Which results depend on the linear-algebra kernel? Same code, same seed, different processors.

numpy's OpenBLAS picks a kernel for the processor at run time (Haswell, Zen, SkylakeX ...); kernels
order floating-point operations differently, so a BLAS or LAPACK call can return different last
bits on different machines. This tool runs the same models and estimators under several kernels
(``OPENBLAS_CORETYPE``, one subprocess each) and reports, per output, whether it is bit-identical,
equal to ``--rtol`` (1e-6), differs below ``--fail-rtol`` (1e-3), or is UNSTABLE above it.
Different last bits are expected where BLAS is used; a large difference means an unstable or
unconverged computation - a bug to fix, not a platform to blame. (Found this way: the w0wa
supernova fit stopped early in flat valleys and gave Om = 0.07 or 0.15 depending on the kernel;
with several starts, tight tolerances and a simplex polish the kernels agree to 3e-4 relative,
far inside the statistical error, and chi2 to 1e-7.)

Only OpenBLAS builds of numpy honour ``OPENBLAS_CORETYPE`` (the PyPI wheels do); with MKL use
``MKL_CBWR`` instead. Kernels the processor cannot run are skipped.

    python tools/check_blas_reproducibility.py                       # about 2 minutes
    python tools/check_blas_reproducibility.py --kernels Haswell SkylakeX --models linear cosmology

Writes ``blas_reproducibility.json`` and ``.md`` to ``--out``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KERNELS = ("Haswell", "Zen", "SkylakeX", "Nehalem", "Sandybridge", "Prescott")
GAUSS = lambda target, s: {"targets": [target], "operators": [{"type": "gaussian", "sigma": s}]}  # noqa: E731
MODELS = {
    "linear": dict(experiment_type="scidatasim.experiments.linear_function.logic.LinearFunctionExperiment",
                   n_points=200, m=2, c=1, x_min=0, x_max=10, observation=GAUSS("y", 0.3)),
    "law_discovery": dict(experiment_type="scidatasim.experiments.law_discovery.logic.HiddenLawExperiment"),
    "heat": dict(experiment_type="scidatasim.experiments.heat.logic.HeatExperiment", alpha=1e-3, length=1.0, nx=32,
                 dt=0.1, n_steps=200, integrator="ftcs"),
    "astrometry": dict(experiment_type="scidatasim.experiments.astrometry.logic.AstrometricBinaryExperiment",
                       companion=True),
    "transits": dict(experiment_type="scidatasim.experiments.transits.logic.TransitExperiment", planet=True,
                     observation=GAUSS("flux", 0.001)),
    "cosmology": dict(experiment_type="scidatasim.experiments.cosmology.logic.SupernovaHubbleExperiment",
                      observation=GAUSS("m_b", 0.15)),
    "resonances": dict(experiment_type="scidatasim.experiments.resonances.logic.ResonanceExperiment", signal=True,
                       observation={"targets": ["counts"], "operators": [{"type": "poisson"}]}),
}


def _estimate(name, sim):
    """The pack's standard estimator on its own output (None for models without one)."""
    r = sim.results
    if name == "astrometry":
        from scidatasim.experiments.astrometry.fit import detect
        d = detect(*sim.logic.inputs_for_baseline(r))
        return [d["fap"], d["period_best"], d["parallax"]]
    if name == "cosmology":
        from scidatasim.experiments.cosmology.model import fit
        f = fit(r["z"]["data"], r["m_b_observed"]["data"], 0.15, "w0wa")
        return list(f["params"].values()) + [f["chi2"]]
    if name == "resonances":
        from scidatasim.experiments.resonances.logic import bump_hunt
        b = bump_hunt(sim.logic.edges, r["counts_observed"]["data"], sim.logic.resolution)
        return [b["z_global"], b["mass"]]
    if name == "transits":
        from scidatasim.experiments.transits.model import transit_search
        t = transit_search(r["time"]["data"], r["flux_observed"]["data"], 1e-3)
        return [t["period"], t["snr"]]
    return None


def probe(models, seed):
    """In this process: data checksums and estimates of the models."""
    from scidatasim.engine import simulate
    from scidatasim.provenance import numerical_environment
    out = {"numerics": numerical_environment(), "results": {}}
    for name in models:
        sim = simulate(dict(MODELS[name], seed=seed))
        arrays = [np.ascontiguousarray(v["data"], float) for _, v in sorted(sim.results.items())
                  if isinstance(v.get("data"), np.ndarray) and v["data"].dtype.kind in "fiu"]
        out["results"][f"{name}: data"] = {"sha256": hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest(),
                                           "values": [float(a.sum()) for a in arrays]}
        est = _estimate(name, sim)
        if est is not None:
            out["results"][f"{name}: estimate"] = {"sha256": hashlib.sha256(json.dumps(est).encode()).hexdigest(),
                                                   "values": [float(v) for v in est]}
    return out


def compare(runs, rtol, fail_rtol=1e-3):
    keys = list(next(iter(runs.values()))["results"])
    rows = []
    for key in keys:
        entries = {k: r["results"][key] for k, r in runs.items()}
        hashes = {e["sha256"] for e in entries.values()}
        values = np.array([e["values"] for e in entries.values()], float)
        spread = np.max(np.abs(values - values[0]) / np.maximum(np.abs(values[0]), 1e-300))
        status = ("bit-identical" if len(hashes) == 1 else "equal to rtol" if spread <= rtol
                  else "differs" if spread <= fail_rtol else "UNSTABLE")
        rows.append({"output": key, "status": status, "max_relative_difference": float(spread)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kernels", nargs="*", default=list(KERNELS))
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rtol", type=float, default=1e-6, help="'equal to rtol' below this relative difference")
    parser.add_argument("--fail-rtol", type=float, default=1e-3,
                        help="'UNSTABLE' (exit 1) above this: an estimate that moves with the last bits")
    parser.add_argument("--out", default=".")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.probe:
        print(json.dumps(probe(args.models, args.seed)))
        return 0
    runs = {}
    for kernel in args.kernels:
        env = dict(os.environ, OPENBLAS_CORETYPE=kernel, OPENBLAS_NUM_THREADS="1")
        done = subprocess.run([sys.executable, os.path.abspath(__file__), "--probe", "--seed", str(args.seed),
                               "--models", *args.models], capture_output=True, text=True, env=env)
        if done.returncode != 0:
            print(f"{kernel}: skipped ({done.stderr.strip().splitlines()[-1] if done.stderr.strip() else 'failed'})")
            continue
        runs[kernel] = json.loads(done.stdout.strip().splitlines()[-1])
    seen = {k: [p.get("architecture") for p in r["numerics"].get("runtime", [])] for k, r in runs.items()}
    rows = compare(runs, args.rtol, args.fail_rtol) if len(runs) > 1 else []
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "blas_reproducibility.json"), "w") as f:
        json.dump({"kernels": seen, "rows": rows, "rtol": args.rtol, "seed": args.seed}, f, indent=1)
    lines = ["# Results across BLAS kernels", "", f"Kernels run (as reported at run time): {seen}", "",
             "| output | status | max relative difference |", "|---|---|---|"]
    lines += [f"| {r['output']} | {r['status']} | {r['max_relative_difference']:.2e} |" for r in rows]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "blas_reproducibility.md"), "w") as f:
        f.write(text)
    print(text)
    return 1 if any(r["status"] == "UNSTABLE" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
