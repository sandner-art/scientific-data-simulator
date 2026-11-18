"""How should an extrapolation question be scored? Blind answers against the right method.

The law-discovery extrapolation question asks for y at a point beyond the data. Under the 0.7.0rc2
rule (5 % of the true value, at least 1 % of the largest observed |y|; a quarter of the range
beyond the data) the answer 0 passed about a third of them: laws that have decayed below the floor.
This study chose the rule shipped since (``achievable``, half the range beyond). This study measures, over the model's
whole prior, which items separate a solver that found the law from answers that ignore it, under
several scoring rules and extrapolation distances.

Grid: every family x textbook/counterfactual x noise-free / 40 dB / 20 dB x ``--draws`` laws; the
point lies 10 %, 25 % (0.7.0rc2) or 50 % (since) of the observed range beyond the data.

Answers:

* blind: ``zero``; ``constant`` (one number, chosen leave-one-out on the other items);
* naive, from the data: ``last_value`` (mean of the last 3 observations), ``linear_tail`` (a line
  through the last quarter);
* ``polynomial``: a least-squares polynomial, degree by BIC;
* ``law_fit``: example 10's classical fit, every family, best by AIC;
* ``true_family``: a fit of the true family (what a solver who identified the law reaches).

Rules (tolerance on |answer - truth|):

* ``rc2``: max(5 % of |y*|, 1 % of the largest |y| on the data range) - 0.7.0rc2;
* ``relative``: 5 % of |y*|;
* ``achievable``: max(5 % of |y*|, the error a fit of the true family reaches at the point in 90 %
  of replicates of the run's noise) - the pointwise analogue of the formula question's
  ``formula_tolerance: auto`` - shipped since (``extrapolation_tolerance: auto``);
* ``rc2, labelled``: ``rc2``, scored only on items the answer 0 does not pass (the
  ``zero_reachable`` label).

An item *discriminates* when the true-family fit passes and every blind and naive answer fails.

Run from the repository root (about 10 minutes on 2 cores with the defaults)::

    python tools/study_extrapolation_scoring.py
    python tools/study_extrapolation_scoring.py --draws 5

Writes ``extrapolation_study.json`` (every item) and ``extrapolation_study.md`` to ``--out``.
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
EXTENSIONS = (0.1, 0.25, 0.5)  # share of the observed range beyond the data
SHIPPED = 0.25  # the tables compare rules at the 0.7.0rc2 distance; by-distance tables show 0.5
SOLVERS = ("zero", "constant", "last_value", "linear_tail", "polynomial", "law_fit", "true_family")
BLIND = ("zero", "constant", "last_value", "linear_tail")
RULES = ("rc2", "relative", "achievable", "rc2, labelled")
REPLICATES, QUANTILE = 30, 0.9


def _base():
    from scidatasim.config import yaml_load
    with open(os.path.join(ROOT, "examples", "example_10_law_discovery", "config.yaml"), encoding="utf-8") as f:
        base = yaml_load(f)
    base.pop("bundle", None)
    base["generate_plots"] = False
    return base


def _predict(expression, points):
    from scidatasim.expressions import evaluate
    if not expression:
        return [float("nan")] * len(points)
    try:
        return [float(evaluate(expression, x=p)) for p in points]
    except Exception:
        return [float("nan")] * len(points)


def family_rows(job):
    family, draws = job
    warnings.filterwarnings("ignore")
    from examples.example_10_law_discovery.run_experiment import fit_best_law
    from scidatasim.diagnostics.shortcuts import polynomial_formula
    from scidatasim.engine import simulate
    from scidatasim.experiments.law_discovery.logic import fit_law, law_function
    base, rows = _base(), []
    names, f = law_function(family)
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
                    continue
                truth = [sim.logic.values[n] for n in names]
                x = np.asarray(sim.results["x"]["data"], float)
                clean = np.asarray(sim.results["y"]["data"], float)
                y = np.asarray((sim.results.get("y_observed") or sim.results["y"])["data"], float)
                ok = np.isfinite(x) & np.isfinite(y)
                x, y, clean = x[ok], y[ok], clean[ok]
                lo, hi = float(x.min()), float(x.max())
                points = [hi + e * (hi - lo) for e in EXTENSIONS]
                y_star = [float(f(np.array([p]), *truth)[0]) for p in points]
                scale = float(np.max(np.abs(clean[np.isfinite(clean)]))) if np.any(np.isfinite(clean)) else 1.0
                # answers
                tail = x >= lo + 0.75 * (hi - lo)
                slope, intercept = np.polyfit(x[tail], y[tail], 1)
                fitted = fit_law(family, x, y)
                true_family = ([float(f(np.array([p]), *[fitted[n] for n in names])[0]) for p in points]
                               if fitted else [float("nan")] * len(points))
                answers = {"zero": [0.0] * len(points),
                           "last_value": [float(np.mean(y[np.argsort(x)[-3:]]))] * len(points),
                           "linear_tail": [float(slope * p + intercept) for p in points],
                           "polynomial": _predict(polynomial_formula(x, y), points),
                           "law_fit": _predict(fit_best_law(x, y)[1], points),
                           "true_family": true_family}
                # achievable error at the points: true-family fits to replicates of the run's noise
                residuals = y - clean
                rng = np.random.default_rng(draw)
                if np.any(residuals):
                    errs = []
                    for _ in range(REPLICATES):
                        rep = fit_law(family, x, clean + rng.choice(residuals, size=x.size, replace=True),
                                      dict(zip(names, truth)))
                        errs.append([abs(float(f(np.array([p]), *[rep[n] for n in names])[0]) - ys)
                                     if rep else np.inf for p, ys in zip(points, y_star)])
                    achievable = np.quantile(np.array(errs, float), QUANTILE, axis=0).tolist()
                else:
                    achievable = [0.0] * len(points)
                for k, ext in enumerate(EXTENSIONS):
                    ys = y_star[k]
                    tol = {"rc2": max(0.05 * abs(ys), 0.01 * scale), "relative": 0.05 * abs(ys),
                           "achievable": max(0.05 * abs(ys), achievable[k])}
                    rows.append({"family": family, "counterfactual": counterfactual, "noise": str(noise),
                                 "draw": draw, "extension": ext, "x_star": points[k], "y_star": ys,
                                 "scale": scale, "tolerance": tol,
                                 "answers": {s: answers[s][k] for s in answers}})
    return rows


def _passes(answer, truth, tolerance):
    return bool(np.isfinite(answer) and abs(answer - truth) <= tolerance)


def score(rows):
    """Adds per-rule pass flags (incl. the leave-one-out constant) to every row, in place."""
    for ext in EXTENSIONS:
        for rule in ("rc2", "relative", "achievable"):
            group = [r for r in rows if r["extension"] == ext]
            tasks = [{"answer": r["y_star"], "answer_type": "number", "tolerance": r["tolerance"][rule]}
                     for r in group]
            # leave-one-out constant, per item (as blind_baseline's constant_accuracy, which reports only the mean)
            values = np.array([t["answer"] for t in tasks])
            candidates = np.unique(np.concatenate([[0.0], np.quantile(values, np.linspace(0, 1, 101))]))
            hits = np.array([[abs(c - t["answer"]) <= t["tolerance"] for t in tasks] for c in candidates], float)
            totals = hits.sum(axis=1)
            for i, r in enumerate(group):
                r.setdefault("constant_pass", {})[rule] = bool(hits[int(np.argmax(totals - hits[:, i])), i])
    for r in rows:
        r["passes"] = {}
        for rule in ("rc2", "relative", "achievable"):
            p = {s: _passes(a, r["y_star"], r["tolerance"][rule]) for s, a in r["answers"].items()}
            p["constant"] = r["constant_pass"][rule]
            r["passes"][rule] = p
        r["zero_reachable"] = r["passes"]["rc2"]["zero"]


def _cell(rows, rule):
    """Pass rates per solver and the share of discriminating items, for one rule."""
    if rule == "rc2, labelled":
        rows, rule = [r for r in rows if not r["zero_reachable"]], "rc2"
    if not rows:
        return None
    rates = {s: float(np.mean([r["passes"][rule][s] for r in rows])) for s in SOLVERS}
    rates["discriminating"] = float(np.mean([r["passes"][rule]["true_family"]
                                             and not any(r["passes"][rule][b] for b in BLIND) for r in rows]))
    rates["n"] = len(rows)
    return rates


def summarise(rows):
    out = {"by_rule": {}, "by_family": {}, "by_noise": {}, "by_extension": {}}
    shipped = [r for r in rows if r["extension"] == SHIPPED]
    for rule in RULES:
        out["by_rule"][rule] = _cell(shipped, rule)
        out["by_family"][rule] = {f: _cell([r for r in shipped if r["family"] == f], rule)
                                  for f in sorted({r["family"] for r in rows})}
        out["by_noise"][rule] = {n: _cell([r for r in shipped if r["noise"] == n], rule) for n in map(str, NOISE)}
        out["by_extension"][rule] = {str(e): _cell([r for r in rows if r["extension"] == e], rule)
                                     for e in EXTENSIONS}
    out["zero_reachable_share"] = {str(e): float(np.mean([r["zero_reachable"] for r in rows if r["extension"] == e]))
                                   for e in EXTENSIONS}
    return out


def markdown(summary, n_runs, draws, version) -> str:
    def row(name, c, cols):
        return f"| {name} | " + " | ".join("-" if c is None else f"{c[k]:.0%}" for k in cols) + \
               f" | {'' if c is None else c['n']} |"
    cols = list(SOLVERS) + ["discriminating"]
    head = "| " + " | ".join(cols) + " | n |"
    lines = [f"# Extrapolation scoring study ({version})", "",
             f"{n_runs} runs (every family x textbook/counterfactual x noise x {draws} draws); point "
             f"{SHIPPED:.0%} of the range beyond the data unless stated. Cells: share of items each answer passes; "
             "*discriminating*: the true-family fit passes and every blind and naive answer fails.", "",
             "## Rules (0.7.0rc2 distance, 25 %)", "", "| rule " + head, "|---" * (len(cols) + 2) + "|"]
    lines += [row(rule, c, cols) for rule, c in summary["by_rule"].items()]
    for rule in RULES:
        lines += ["", f"## By family: {rule}", "", "| family " + head, "|---" * (len(cols) + 2) + "|"]
        lines += [row(f, c, cols) for f, c in summary["by_family"][rule].items()]
    for rule in ("rc2", "achievable"):
        lines += ["", f"## By noise: {rule}", "", "| noise " + head, "|---" * (len(cols) + 2) + "|"]
        lines += [row(n if n == "none" else f"{n} dB", c, cols) for n, c in summary["by_noise"][rule].items()]
    for rule in ("rc2", "relative", "achievable"):
        lines += ["", f"## By distance beyond the data: {rule}", "", "| distance " + head,
                  "|---" * (len(cols) + 2) + "|"]
        lines += [row(f"{float(e):.0%}", c, cols) for e, c in summary["by_extension"][rule].items()]
    lines += ["", "Share of items the answer 0 passes under the rc2 rule, by distance: "
              + ", ".join(f"{float(e):.0%}: {v:.0%}" for e, v in summary["zero_reachable_share"].items()) + "."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draws", type=int, default=40, help="laws per family, variant and noise level")
    parser.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default=os.path.join("experiments_output", "extrapolation_study"))
    args = parser.parse_args(argv)
    from scidatasim import __version__
    from scidatasim.experiments.law_discovery.logic import LAWS
    start = time.time()
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = [r for part in pool.map(family_rows, [(f, args.draws) for f in sorted(LAWS)]) for r in part]
    score(rows)
    summary = summarise(rows)
    n_runs = len(rows) // len(EXTENSIONS)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "extrapolation_study.json"), "w", encoding="utf-8") as f:
        json.dump({"version": __version__, "draws": args.draws, "summary": summary, "items": rows}, f, indent=1,
                  default=float)
    text = markdown(summary, n_runs, args.draws, __version__)
    with open(os.path.join(args.out, "extrapolation_study.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text + f"\n{n_runs} runs in {time.time() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
