"""Does a plain polynomial pass the law-discovery formula question? Old rule against current rule.

Runs the hidden-law model over its whole prior (every family, textbook and counterfactual,
noise-free / 40 dB / 20 dB, ``--draws`` laws each) and submits two answers to every run's formula
question:

* a least-squares polynomial in x, degree 1-8 chosen by BIC (``diagnostics.polynomial_formula``):
  it knows nothing about laws, so it should fail;
* the classical law fit of example 10 (``fit_best_law``), on the first ``--law-fit-draws`` noisy
  draws: it should pass.

Each answer is scored under the current rule (the shipped question: data range and half of it
beyond, tolerance set by the run's noise) and under the rule of 0.7.0rc1 (the data range,
1 % relative RMS). This reproduces the tables of the 0.7.0rc1 verification note.

Run from the repository root (about 4 minutes on 2 cores with the defaults)::

    python tools/check_formula_scoring.py
    python tools/check_formula_scoring.py --draws 5   # a quick look

Writes ``formula_scoring.json`` and ``formula_scoring.md`` to ``--out``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NOISE = ("none", 40, 20)


def _base():
    from scidatasim.config import yaml_load
    with open(os.path.join(ROOT, "examples", "example_10_law_discovery", "config.yaml"), encoding="utf-8") as f:
        base = yaml_load(f)
    base.pop("bundle", None)
    base["generate_plots"] = False
    return base


def family_rows(job):
    family, draws, law_fit_draws = job
    warnings.filterwarnings("ignore")
    from examples.example_10_law_discovery.run_experiment import fit_best_law
    from scidatasim.bundle import check_answer
    from scidatasim.diagnostics.shortcuts import polynomial_formula
    from scidatasim.engine import simulate
    base, rows = _base(), []
    for counterfactual in (False, True):
        for noise in NOISE:
            for draw in range(draws):
                cfg = copy.deepcopy(base)
                cfg.update(law=family, counterfactual=counterfactual, seed=1000 + draw)
                if noise == "none":
                    cfg.pop("observation", None)
                else:
                    cfg["observation"]["operators"][0]["snr_db"] = noise
                try:
                    sim = simulate(cfg)
                except ValueError:
                    continue  # a draw the model rejects (e.g. non-finite values)
                x = np.asarray(sim.results["x"]["data"], float)
                y = np.asarray((sim.results.get("y_observed") or sim.results["y"])["data"], float)
                ok = np.isfinite(y)
                x, y = x[ok], y[ok]
                items = sim.logic.questions(sim.results, np.random.default_rng(draw))
                current = next(i for i in items if i["template"] == "law_formula")
                old = dict(current, domain=[float(np.min(x)), float(np.max(x))], tolerance=0.01)
                poly = polynomial_formula(x, y) or "0"
                row = {"family": family, "counterfactual": counterfactual, "noise": str(noise),
                       "tolerance": current["tolerance"],
                       "poly_old": bool(check_answer(old, poly)), "poly_current": bool(check_answer(current, poly))}
                if noise != "none" and draw < law_fit_draws:
                    _, formula = fit_best_law(x, y)
                    row["fit_old"] = bool(formula) and bool(check_answer(old, formula))
                    row["fit_current"] = bool(formula) and bool(check_answer(current, formula))
                rows.append(row)
    return rows


def _rate(rows, key):
    values = [r[key] for r in rows if key in r]
    return (float(np.mean(values)), len(values)) if values else (float("nan"), 0)


def summarise(rows):
    out = {}
    for noise in map(str, NOISE):
        cell = [r for r in rows if r["noise"] == noise]
        out[noise] = {key: _rate(cell, key) for key in ("poly_old", "poly_current", "fit_old", "fit_current")}
        out[noise]["poly_current_by_family"] = {
            f: _rate([r for r in cell if r["family"] == f], "poly_current")[0]
            for f in sorted({r["family"] for r in cell})}
        out[noise]["median_tolerance"] = float(np.median([r["tolerance"] for r in cell])) if cell else float("nan")
    return out


def markdown(summary, n_runs, draws, law_fit_draws, version) -> str:
    def pct(v):
        rate, n = v
        return "-" if not n else f"{rate:.0%}"
    lines = [f"# Formula scoring: old rule against current rule ({version})", "",
             f"{n_runs} runs: every family x textbook/counterfactual x noise x {draws} draws; the law fit on the "
             f"first {law_fit_draws} noisy draws.", "",
             "| Noise | Polynomial, rc1 rule | Polynomial, current | Law fit, rc1 rule | Law fit, current | "
             "Median tolerance |", "|---|---|---|---|---|---|"]
    for noise, s in summary.items():
        lines.append(f"| {noise if noise == 'none' else noise + ' dB'} | {pct(s['poly_old'])} | "
                     f"{pct(s['poly_current'])} | {pct(s['fit_old'])} | {pct(s['fit_current'])} | "
                     f"{s['median_tolerance']:.1%} |")
    lines += ["", "Polynomial passes under the current rule, by family:", ""]
    families = sorted(next(iter(summary.values()))["poly_current_by_family"])
    lines += ["| Family | " + " | ".join(n if n == "none" else f"{n} dB" for n in summary) + " |",
              "|---|" + "---|" * len(summary)]
    for f in families:
        lines.append(f"| {f} | " + " | ".join(f"{summary[n]['poly_current_by_family'].get(f, float('nan')):.0%}"
                                              for n in summary) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=40, help="laws per family, variant and noise level")
    parser.add_argument("--law-fit-draws", type=int, default=10, help="noisy draws also answered by the law fit")
    parser.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default=os.path.join("experiments_output", "formula_scoring"))
    args = parser.parse_args(argv)
    from scidatasim import __version__
    from scidatasim.experiments.law_discovery.logic import LAWS
    start = time.time()
    jobs = [(f, args.draws, args.law_fit_draws) for f in sorted(LAWS)]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = [r for part in pool.map(family_rows, jobs) for r in part]
    summary = summarise(rows)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "formula_scoring.json"), "w", encoding="utf-8") as f:
        json.dump({"version": __version__, "draws": args.draws, "summary": summary, "runs": rows}, f, indent=2)
    text = markdown(summary, len(rows), args.draws, args.law_fit_draws, __version__)
    with open(os.path.join(args.out, "formula_scoring.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text + f"\n{len(rows)} runs in {time.time() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
