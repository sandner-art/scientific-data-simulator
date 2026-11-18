"""Law-discovery environment: the right law against a polynomial, fixed 2 % against tolerance: auto.

Plays example 10's environment (``env.yaml``) at several noise levels. In every episode the agent
runs its three experiments over the whole measurable range [0.5, 5], then answers twice:

* the classical law fit of example 10 (``fit_best_law``): it should pass;
* a least-squares polynomial (degree by BIC): it should fail.

Each answer is scored with the fixed 2 % of 0.7.0rc2 and with ``tolerance: auto`` (the larger of
2 % and the error a fit of the true law reaches from the agent's measurements).

Run from the repository root::

    python tools/check_env_tolerance.py                      # 40 episodes per noise level
    python tools/check_env_tolerance.py --episodes 10 --snr 40 20

Writes ``env_tolerance.json`` and ``env_tolerance.md`` to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FIXED = 0.02


def play(snr_db: float, episodes: int):
    from examples.example_10_law_discovery.run_experiment import fit_best_law
    from scidatasim.config import yaml_load
    from scidatasim.diagnostics.shortcuts import polynomial_formula
    from scidatasim.envs import ExperimentEnv
    from scidatasim.expressions import compare
    with open(os.path.join(ROOT, "examples", "example_10_law_discovery", "env.yaml"), encoding="utf-8") as f:
        spec = yaml_load(f)
    spec["tolerance"] = "auto"
    spec["base_config"]["observation"]["operators"][0]["snr_db"] = snr_db
    env = ExperimentEnv(spec)
    rows = []
    for seed in range(episodes):
        env.reset(seed=seed)
        xs, ys = [], []
        for _ in range(int(spec["budget"])):
            m = env.step({"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}})["observation"]
            xs += m["measurements"]["y_observed"]["x"]
            ys += m["measurements"]["y_observed"]["values"]
        x, y = np.array(xs, float), np.array(ys, float)
        tolerance = env.formula_tolerance()
        truth, domain = env.truth["law_expression"], tuple(env.domain)
        _, law = fit_best_law(x, y)
        law_error = compare(law or "0", truth, domain)["relative_rms"]
        poly_error = compare(polynomial_formula(x, y) or "0", truth, domain)["relative_rms"]
        rows.append({"seed": seed, "law": env.truth["law"], "tolerance": tolerance,
                     "law_error": law_error, "poly_error": poly_error})
    return rows


def summarise(rows):
    tol = np.array([r["tolerance"] for r in rows])
    law, poly = np.array([r["law_error"] for r in rows]), np.array([r["poly_error"] for r in rows])
    return {"episodes": len(rows), "median_tolerance": float(np.median(tol)), "max_tolerance": float(np.max(tol)),
            "law_fixed": float(np.mean(law <= FIXED)), "law_auto": float(np.mean(law <= tol)),
            "poly_fixed": float(np.mean(poly <= FIXED)), "poly_auto": float(np.mean(poly <= tol)),
            "poly_auto_laws": sorted({r["law"] for r, p, t in zip(rows, poly, tol) if p <= t})}


def markdown(summary, version) -> str:
    lines = [f"# Environment tolerance: fixed 2 % against auto ({version})", "",
             "| Noise | Episodes | Tolerance, median (max) | Law fit, 2 % | Law fit, auto | Polynomial, 2 % | "
             "Polynomial, auto | Laws a polynomial passes (auto) |", "|---|---|---|---|---|---|---|---|"]
    for snr, s in summary.items():
        lines.append(f"| {snr} dB | {s['episodes']} | {s['median_tolerance']:.1%} ({s['max_tolerance']:.1%}) | "
                     f"{s['law_fixed']:.0%} | {s['law_auto']:.0%} | {s['poly_fixed']:.0%} | {s['poly_auto']:.0%} | "
                     f"{', '.join(s['poly_auto_laws']) or '-'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--snr", type=float, nargs="+", default=[40, 30, 20, 15], help="noise levels (dB)")
    parser.add_argument("--out", default=os.path.join("experiments_output", "env_tolerance"))
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    from scidatasim import __version__
    runs = {f"{snr:g}": play(snr, args.episodes) for snr in args.snr}
    summary = {snr: summarise(rows) for snr, rows in runs.items()}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "env_tolerance.json"), "w", encoding="utf-8") as f:
        json.dump({"version": __version__, "summary": summary, "episodes": runs}, f, indent=2)
    text = markdown(summary, __version__)
    with open(os.path.join(args.out, "env_tolerance.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
