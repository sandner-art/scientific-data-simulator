"""Astrometric binaries: is the detection baseline calibrated, and what does it find?

Runs the astrometry model (``AstrometricBinaryExperiment``) and applies the baseline of
``scidatasim.experiments.astrometry.fit`` to what a solver is given (times, scan angles, parallax
factors, reported errors, observed w). Two samples, both with fixed seeds:

**Single stars** (``--nulls``, default 2,000): the share whose highest periodogram peak has FAP
below 0.1, 0.01 and 0.001, against those nominal rates, with a 95 % binomial interval. The FAP is
an approximation (independent frequencies = span x frequency range); this measures it.

**Binaries** (``--binaries``, default 400):

1. Detection rate by the orbit's non-centrality (the Delta chi2 the noise-free orbit adds over the
   best five-parameter fit), against the rate predicted from a non-central chi2 with 4 degrees of
   freedom at the true period. Non-centrality, unlike a0 / sigma, carries over to other sampling
   and noise, so this is the curve a prediction for DR4 is stated in.
2. Detection with weights that include the source excess noise of a five-parameter fit (as AGIS
   does) against the measurement errors alone.
3. Period: the periodogram peak and the final Keplerian period within 5 % of the truth, and how
   often a wrong period has a lower chi2 than the true one (then the data, not the search, are at
   fault: the private label ``period_ambiguous``).
4. Parallax: how often the five-parameter parallax and the pipeline's parallax (the Keplerian
   fit's when an orbit is detected) pass the parallax question, by period.

Run from the repository root::

    python tools/check_astrometry_detection.py                  # about 15 minutes
    python tools/check_astrometry_detection.py --nulls 500 --binaries 100

Writes ``astrometry_detection.json`` and ``astrometry_detection.md`` to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ALPHAS = (0.1, 0.01, 0.001)
LAMBDA_BINS = [(0, 10), (10, 20), (20, 35), (35, 50), (50, 80), (80, 150), (150, np.inf)]
PERIOD_BINS = [(20, 200), (200, 300), (300, 450), (450, 1000), (1000, 2000)]
NULL_SEED0, BINARY_SEED0 = 100_000, 200_000


def _model(seed, companion):
    from scidatasim.experiments.astrometry.logic import AstrometricBinaryExperiment
    model = AstrometricBinaryExperiment({"seed": seed, "companion": companion})
    model.initialize({"seed": seed})
    results = model.get_results()
    return model, results


def null_run(seed: int):
    from scidatasim.experiments.astrometry.fit import detect
    model, results = _model(seed, False)
    ep, w = model.inputs_for_baseline(results)
    return {"seed": seed, "fap": detect(ep, w, keplerian=False)["fap"]}


def binary_run(seed: int):
    from scidatasim.experiments.astrometry.fit import _keplerian_solution, detect, excess_noise
    from scidatasim.experiments.astrometry.model import Epochs
    model, results = _model(seed, True)
    ep, w = model.inputs_for_baseline(results)
    orbit = model.orbit
    found = detect(ep, w)
    eps = excess_noise(ep, w)
    agis = detect(Epochs(ep.t, ep.theta, ep.f_al, np.sqrt(ep.sigma ** 2 + eps ** 2)), w, keplerian=False)
    items = {q["template"]: q for q in model.questions(results, np.random.default_rng(seed))}
    plx = items["astrometric_parallax"]

    def close(p):
        return bool(abs(p - orbit.period) <= 0.05 * orbit.period)

    row = {"seed": seed, "period": orbit.period, "eccentricity": orbit.eccentricity,
           "signal": model.elements["a0"] / model.sigma_al, "noncentrality": model.noncentrality(),
           "predicted_detection": model.detection_probability(), "detected": found["detected"],
           "detected_agis": agis["detected"], "fap": found["fap"], "excess_noise": eps,
           "undetectable": bool(items["astrometric_companion"]["signal_undetectable"]),
           "period_ambiguous": bool(items["astrometric_period"].get("period_ambiguous", False)),
           "plx_five_pass": bool(abs(found["single_star"]["parallax"] - plx["answer"]) <= plx["tolerance"]),
           "plx_pipeline_pass": bool(abs(found["parallax"] - plx["answer"]) <= plx["tolerance"])}
    if found["detected"]:
        truth = _keplerian_solution(ep, w, orbit.period, orbit.eccentricity, orbit.t_periastron)
        row.update(peak_ok=close(found["period"]), best_ok=close(found["period_best"]),
                   wrong_period_lower_chi2=bool(not close(found["period_best"])
                                                and found["keplerian"]["chi2"] < truth["chi2"]))
    return row


def binomial_interval(k, n):
    from scipy.stats import beta
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def rate(rows, key):
    rows = [r for r in rows if key in r]
    return (sum(bool(r[key]) for r in rows) / len(rows)) if rows else None


def summarise(nulls, bins):
    out = {"nulls": len(nulls), "binaries": len(bins), "false_alarms": [], "by_noncentrality": [],
           "by_period": [], "labels": {}}
    faps = np.array([r["fap"] for r in nulls])
    for alpha in ALPHAS:
        k = int((faps < alpha).sum())
        lo, hi = binomial_interval(k, len(faps))
        out["false_alarms"].append({"FAP below": alpha, "observed": k / len(faps), "95 % interval": f"{lo:.4f}-{hi:.4f}",
                                    "count": k})
    for lo, hi in LAMBDA_BINS:
        sel = [r for r in bins if lo <= r["noncentrality"] < hi]
        out["by_noncentrality"].append({
            "non-centrality": f"{lo}-{hi:g}", "n": len(sel), "detected": rate(sel, "detected"),
            "predicted": float(np.mean([r["predicted_detection"] for r in sel])) if sel else None,
            "AGIS weights": rate(sel, "detected_agis"), "peak within 5 %": rate(sel, "peak_ok"),
            "final period within 5 %": rate(sel, "best_ok")})
    for lo, hi in PERIOD_BINS:
        sel = [r for r in bins if lo <= r["period"] < hi]
        out["by_period"].append({"period (d)": f"{lo}-{hi}", "n": len(sel), "detected": rate(sel, "detected"),
                                 "final period within 5 %": rate(sel, "best_ok"),
                                 "parallax passes, five-parameter": rate(sel, "plx_five_pass"),
                                 "parallax passes, pipeline": rate(sel, "plx_pipeline_pass")})
    det = [r for r in bins if r["detected"]]
    wrong = [r for r in det if not r["best_ok"]]
    out["period_errors"] = {"detected": len(det), "final period off by > 5 %": len(wrong),
                            "of those, wrong period has lower chi2 than the truth": sum(
                                r["wrong_period_lower_chi2"] for r in wrong)}
    for name in ("undetectable", "period_ambiguous"):
        for flag in (False, True):
            sel = [r for r in bins if r[name] == flag]
            out["labels"][f"{name}={flag}"] = {"n": len(sel), "detected": rate(sel, "detected"),
                                               "final period within 5 %": rate(sel, "best_ok")}
    return out


def markdown(s):
    def fmt(v):
        return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    def table(rows):
        cols = list(rows[0])
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        return lines + ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]

    lines = [f"# Astrometric binaries: detection baseline ({s['nulls']} single stars, {s['binaries']} binaries)", "",
             "## False alarms on single stars", ""] + table(s["false_alarms"])
    lines += ["", "## Binaries by non-centrality (predicted: non-central chi2, 4 dof, at the true period)", ""]
    lines += table(s["by_noncentrality"])
    lines += ["", "## Binaries by period", ""] + table(s["by_period"])
    lines += ["", "## Period errors", ""] + [f"- {k}: {v}" for k, v in s["period_errors"].items()]
    lines += ["", "## Private labels", ""] + [f"- {k}: n = {v['n']}, detected {fmt(v['detected'])}, final period "
                                             f"within 5 % {fmt(v['final period within 5 %'])}"
                                             for k, v in s["labels"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nulls", type=int, default=2000)
    parser.add_argument("--binaries", type=int, default=400)
    parser.add_argument("--out", default=".")
    args = parser.parse_args(argv)
    nulls = [null_run(NULL_SEED0 + i) for i in range(args.nulls)]
    bins = [binary_run(BINARY_SEED0 + i) for i in range(args.binaries)]
    summary = summarise(nulls, bins)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "astrometry_detection.json"), "w") as f:
        json.dump({"summary": summary, "nulls": nulls, "binaries": bins}, f, indent=1, default=float)
    text = markdown(summary)
    with open(os.path.join(args.out, "astrometry_detection.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
