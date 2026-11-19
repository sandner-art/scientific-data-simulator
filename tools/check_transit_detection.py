"""Transits: how the standard search (detrend + BLS + SNR > 7.1) does on planted truth.

Runs ``TransitExperiment`` with white noise (1000 ppm per 2-min point, a TESS sector) and reports:

1. **False alarms**: share of stars without a planet where the search claims one - for quiet stars
   and for spotted stars (rotational variability, amplitude 0.3 %), which the running-median
   detrending only partly removes.
2. **Detections by expected SNR**, against the private label ``signal_undetectable`` (SNR < 7.1).
3. **The questions answered by the baseline**: period within tolerance (and how often it found
   P/2 or 2P), and the depth question with the box depth - a box fit to a limb-darkened transit
   is shallower than its largest decrement, which the depth tolerance only partly allows for.

    python tools/check_transit_detection.py                  # about 15 minutes
    python tools/check_transit_detection.py --nulls 20 --planets 40

Writes ``transit_detection.json`` and ``.md`` to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EXP = "scidatasim.experiments.transits.logic.TransitExperiment"
NOISE = {"targets": ["flux"], "operators": [{"type": "gaussian", "sigma": 0.001}]}
SNR_BINS = [(0, 4), (4, 7.1), (7.1, 10), (10, 15), (15, 30), (30, np.inf)]


def run(seed, planet, amplitude=0.0):
    from scidatasim.engine import simulate
    from scidatasim.experiments.transits.model import transit_search
    sim = simulate(dict(experiment_type=EXP, seed=seed, planet=planet, sigma_ppm=1000.0, observation=NOISE,
                        variability={"amplitude": amplitude}))
    found = transit_search(sim.results["time"]["data"], sim.results["flux_observed"]["data"], 1e-3)
    items = {q["template"]: q for q in sim.logic.questions(sim.results, np.random.default_rng(seed))}
    row = {"seed": seed, "planet": planet, "amplitude": amplitude, "detected": found["detected"], "snr": found["snr"],
           "found_period": found["period"], "found_depth": found["depth_ppm"]}
    if planet:
        p, d = items["transit_period"], items["transit_depth"]
        true_p = p["answer"]
        row.update(expected_snr=sim.logic.expected_snr(), period=true_p,
                   undetectable=bool(items["transit_present"]["signal_undetectable"]),
                   single=bool(p["period_ambiguous"]),
                   period_pass=bool(found["detected"] and abs(found["period"] - true_p) <= p["tolerance"]),
                   alias=bool(found["detected"] and any(abs(found["period"] - k * true_p) <= 0.01 * k * true_p
                                                        for k in (0.5, 2.0))),
                   depth_pass=bool(found["detected"] and abs(found["depth_ppm"] - d["answer"]) <= d["tolerance"]),
                   depth_ratio=found["depth_ppm"] / d["answer"] if d["answer"] else None)
    return row


def rate(rows, key):
    rows = [r for r in rows if r.get(key) is not None]
    return float(np.mean([bool(r[key]) for r in rows])) if rows else None


def summarise(quiet, spotted, planets):
    out = {"false_alarms": {"quiet": {"n": len(quiet), "rate": rate(quiet, "detected")},
                            "spotted (0.3 %)": {"n": len(spotted), "rate": rate(spotted, "detected")}},
           "by_expected_snr": [], "labels": {}}
    for lo, hi in SNR_BINS:
        sel = [r for r in planets if lo <= r["expected_snr"] < hi]
        det = [r for r in sel if r["detected"]]
        out["by_expected_snr"].append({"expected SNR": f"{lo}-{hi:g}", "n": len(sel), "detected": rate(sel, "detected"),
                                       "period passes": rate(det, "period_pass"), "P/2 or 2P": rate(det, "alias"),
                                       "depth passes (box)": rate(det, "depth_pass"),
                                       "box / true depth": float(np.median([r["depth_ratio"] for r in det]))
                                       if det else None})
    for name in ("undetectable", "single"):
        for flag in (False, True):
            sel = [r for r in planets if r[name] == flag]
            out["labels"][f"{name}={flag}"] = {"n": len(sel), "detected": rate(sel, "detected"),
                                               "period passes": rate(sel, "period_pass")}
    return out


def markdown(s):
    def fmt(v):
        return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = ["# Transits: detrend + BLS + SNR > 7.1 on planted truth", "", "## False alarms (no planet)", ""]
    lines += [f"- {k}: {fmt(v['rate'])} of {v['n']}" for k, v in s["false_alarms"].items()]
    rows = s["by_expected_snr"]
    cols = list(rows[0])
    lines += ["", "## Planets by expected SNR", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    lines += ["", "## Private labels", ""] + [f"- {k}: n = {v['n']}, detected {fmt(v['detected'])}, period passes "
                                             f"{fmt(v['period passes'])}" for k, v in s["labels"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nulls", type=int, default=60)
    parser.add_argument("--planets", type=int, default=150)
    parser.add_argument("--out", default=".")
    args = parser.parse_args(argv)
    quiet = [run(10_000 + i, False) for i in range(args.nulls)]
    spotted = [run(20_000 + i, False, 0.003) for i in range(args.nulls)]
    planets = [run(30_000 + i, True) for i in range(args.planets)]
    summary = summarise(quiet, spotted, planets)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "transit_detection.json"), "w") as f:
        json.dump({"summary": summary, "runs": quiet + spotted + planets}, f, indent=1, default=float)
    text = markdown(summary)
    with open(os.path.join(args.out, "transit_detection.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
