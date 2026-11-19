"""Astrometric binaries: is the detection baseline calibrated, and what does it find?

Runs the astrometry model (``AstrometricBinaryExperiment``) many times and applies the baseline of
``scidatasim.experiments.astrometry.fit`` to what a solver is given (times, scan angles, parallax
factors, reported errors, observed w). It reports:

1. **False alarms on single stars.** The share of companion-free runs whose highest periodogram
   peak has FAP below 1e-2 and 1e-3, against those nominal rates. The FAP is an approximation
   (independent frequencies = span x frequency range); this measures it.
2. **Detections by signal.** Detection rate and period recovery (within 5 %) by a0 / sigma_al, and
   separately for companions labelled ``signal_undetectable``.
3. **Weights.** Detection rate when the weights include the source excess noise of a
   five-parameter fit, as AGIS does, against the measurement errors alone. The excess noise
   absorbs the orbit, so AGIS-style weights should cost detections.
4. **Parallax.** How often the five-parameter parallax passes the parallax question for binaries,
   by period: an orbit near one year imitates parallax.

Run from the repository root::

    python tools/check_astrometry_detection.py               # 400 runs (under a minute)
    python tools/check_astrometry_detection.py --runs 100    # quicker

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

SIGNAL_BINS = [(0.3, 0.7), (0.7, 1.5), (1.5, 3.0), (3.0, 10.0)]
PERIOD_BINS = [(20, 200), (200, 300), (300, 450), (450, 2000)]


def one_run(seed: int):
    from scidatasim.experiments.astrometry.fit import detect, excess_noise, fit_single_star
    from scidatasim.experiments.astrometry.logic import AstrometricBinaryExperiment
    from scidatasim.experiments.astrometry.model import Epochs
    model = AstrometricBinaryExperiment({"seed": seed})
    model.initialize({"seed": seed})
    results = model.get_results()
    ep, w = model.inputs_for_baseline(results)
    found = detect(ep, w, keplerian=False)
    eps = excess_noise(ep, w)
    agis = detect(Epochs(ep.t, ep.theta, ep.f_al, np.sqrt(ep.sigma ** 2 + eps ** 2)), w, keplerian=False)
    items = {q["template"]: q for q in model.questions(results, np.random.default_rng(seed))}
    plx = fit_single_star(ep, w)["parallax"]
    row = {"seed": seed, "companion": model.orbit is not None, "fap": found["fap"], "detected": found["detected"],
           "period_found": found["period"], "fap_agis": agis["fap"], "detected_agis": agis["detected"],
           "excess_noise": eps, "parallax_pass": bool(abs(plx - items["astrometric_parallax"]["answer"])
                                                     <= items["astrometric_parallax"]["tolerance"]),
           "undetectable": bool(items["astrometric_companion"]["signal_undetectable"])}
    if model.orbit is not None:
        row.update(period=model.orbit.period, signal=model.elements["a0"] / model.sigma_al,
                   period_ok=bool(abs(found["period"] - model.orbit.period) <= 0.05 * model.orbit.period))
    return row


def rate(rows, key):
    return (sum(bool(r[key]) for r in rows) / len(rows)) if rows else None


def summarise(rows):
    null = [r for r in rows if not r["companion"]]
    bins = [r for r in rows if r["companion"]]
    out = {"runs": len(rows), "single_stars": len(null), "binaries": len(bins),
           "false_alarms": {"fap<1e-2": float(np.mean([r["fap"] < 1e-2 for r in null])) if null else None,
                            "fap<1e-3": float(np.mean([r["fap"] < 1e-3 for r in null])) if null else None,
                            "fap<1e-3, AGIS weights": float(np.mean([r["fap_agis"] < 1e-3 for r in null]))
                            if null else None},
           "by_signal": [], "labels": {}, "by_period": []}
    for lo, hi in SIGNAL_BINS:
        sel = [r for r in bins if lo <= r["signal"] < hi]
        out["by_signal"].append({"a0/sigma": f"{lo}-{hi}", "n": len(sel), "detected": rate(sel, "detected"),
                                 "detected, AGIS weights": rate(sel, "detected_agis"),
                                 "period within 5 %": rate([r for r in sel if r["detected"]], "period_ok"),
                                 "labelled undetectable": rate(sel, "undetectable")})
    for flag in (False, True):
        sel = [r for r in bins if r["undetectable"] == flag]
        out["labels"]["undetectable" if flag else "detectable"] = {"n": len(sel), "detected": rate(sel, "detected")}
    for lo, hi in PERIOD_BINS:
        sel = [r for r in bins if lo <= r["period"] < hi]
        out["by_period"].append({"period (d)": f"{lo}-{hi}", "n": len(sel),
                                 "five-parameter parallax passes": rate(sel, "parallax_pass")})
    out["single_star_parallax_passes"] = rate(null, "parallax_pass")
    return out


def markdown(s):
    def fmt(v):
        return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = [f"# Astrometric binaries: detection baseline ({s['runs']} runs: {s['single_stars']} single, "
             f"{s['binaries']} binaries)", "", "## False alarms on single stars (nominal rate = threshold)", ""]
    lines += [f"- {k}: {fmt(v)}" for k, v in s["false_alarms"].items()]
    for title, key in (("Detections by signal", "by_signal"), ("Five-parameter parallax by period", "by_period")):
        rows = s[key]
        cols = list(rows[0])
        lines += ["", f"## {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    lines += ["", "## Private label `signal_undetectable`", ""]
    lines += [f"- {k}: n = {v['n']}, detected {fmt(v['detected'])}" for k, v in s["labels"].items()]
    lines += ["", f"Single stars: five-parameter parallax passes {fmt(s['single_star_parallax_passes'])} "
                  "(the tolerance is the 90 % error of the true model)."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=int, default=400)
    parser.add_argument("--out", default=".")
    args = parser.parse_args(argv)
    rows = [one_run(seed) for seed in range(args.runs)]
    summary = summarise(rows)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "astrometry_detection.json"), "w") as f:
        json.dump({"summary": summary, "runs": rows}, f, indent=1, default=float)
    text = markdown(summary)
    with open(os.path.join(args.out, "astrometry_detection.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
