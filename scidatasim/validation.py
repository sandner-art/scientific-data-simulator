"""Validation reports: does an estimator behave the same on simulated data and on the real data
the simulation mimics, and does it reproduce the published answer?

A model earns more trust than "verified against closed forms" when a standard estimator, applied
both to its output and to real data, (1) reproduces a published result on the real data and
(2) recovers the planted truth on simulated data of the same design without bias and with honest
error bars. The report states both, with the real file's provenance and the reference, and gives a
verdict per check rather than a single "validated" flag:

* ``reproduces_published``: |real - published| <= ``k_published`` x the published error;
* ``unbiased``: |mean(estimate - truth)| <= ``k_bias`` x the spread of the estimates / sqrt(n);
* ``coverage``: share of simulations whose 1-sigma interval holds the truth, within the binomial
  95 % range around 68.3 %.
"""
from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy import stats

ONE_SIGMA = 0.6826894921370859


def validation_report(name: str, estimator: str, parameter: str, real: Dict[str, Any], published: Dict[str, Any],
                      truth: float, estimates: Sequence[float], errors: Sequence[float],
                      k_published: float = 1.0, k_bias: float = 3.0, notes: Optional[str] = None) -> Dict[str, Any]:
    """``real``: {"estimate", "error", "provenance": {...}}; ``published``: {"value", "error",
    "reference"}; ``truth``: the value planted in the simulations; ``estimates``/``errors``: the
    estimator's results on them."""
    est, err = np.asarray(estimates, float), np.asarray(errors, float)
    n = est.size
    bias = float(np.mean(est - truth)) if n else float("nan")
    spread = float(np.std(est, ddof=1)) if n > 1 else float("nan")
    covered = int(np.sum(np.abs(est - truth) <= err))
    lo, hi = stats.binom.ppf([0.025, 0.975], n, ONE_SIGMA) / max(n, 1)
    pull_sd = float(np.std((est - truth) / err, ddof=1)) if n > 1 else float("nan")
    diff = float(real["estimate"] - published["value"])
    checks = {
        "reproduces_published": bool(abs(diff) <= k_published * published["error"]),
        "unbiased": bool(n > 1 and abs(bias) <= k_bias * spread / np.sqrt(n)),
        "coverage": bool(n > 0 and lo <= covered / n <= hi),
    }
    return {
        "name": name, "estimator": estimator, "parameter": parameter,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "real": real, "published": published,
        "real_minus_published": diff, "in_published_errors": diff / published["error"],
        "simulation": {"n": n, "truth": float(truth), "bias": bias, "spread": spread,
                       "mean_reported_error": float(np.mean(err)) if n else float("nan"),
                       "coverage_1sigma": covered / n if n else float("nan"),
                       "coverage_expected": [float(lo), float(hi)], "pull_sd": pull_sd},
        "checks": checks, "passed": all(checks.values()), "notes": notes,
    }


def markdown(report: Dict[str, Any]) -> str:
    r, p, s = report["real"], report["published"], report["simulation"]
    yes = {True: "yes", False: "NO"}
    lines = [f"# Validation: {report['name']}", "",
             f"Estimator: {report['estimator']} ({report['parameter']})", "",
             "| check | result |", "|---|---|",
             f"| reproduces the published value | {yes[report['checks']['reproduces_published']]}: "
             f"{r['estimate']:.4g} ± {r['error']:.2g} vs {p['value']:.4g} ± {p['error']:.2g} "
             f"({report['in_published_errors']:+.2f} published σ) |",
             f"| unbiased on simulations | {yes[report['checks']['unbiased']]}: bias {s['bias']:+.3g} "
             f"(spread {s['spread']:.3g}, n = {s['n']}) |",
             f"| honest error bars | {yes[report['checks']['coverage']]}: 1σ coverage {s['coverage_1sigma']:.3f} "
             f"(expected {s['coverage_expected'][0]:.3f}-{s['coverage_expected'][1]:.3f}); pull sd {s['pull_sd']:.3f} |",
             "", f"Reference: {p.get('reference', '')}", ""]
    prov = r.get("provenance") or {}
    if prov:
        lines += ["Real data:", ""] + [f"- {k}: {v}" for k, v in prov.items()] + [""]
    if report.get("notes"):
        lines += [report["notes"], ""]
    return "\n".join(lines)


def save(report: Dict[str, Any], path_without_extension: str) -> None:
    with open(path_without_extension + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=float)
    with open(path_without_extension + ".md", "w", encoding="utf-8") as f:
        f.write(markdown(report))
