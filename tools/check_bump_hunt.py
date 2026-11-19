"""Resonances: is the bump hunt's global significance calibrated, and what does a trigger do?

Runs ``ResonanceExperiment`` (Poisson counts) and the baseline ``bump_hunt`` (background fit,
Gaussian signal scan, Gross-Vitells global p-value) and reports:

1. **Calibration on spectra without a new resonance**: the share with global p below 0.1, 0.05 and
   0.01 against those nominal rates, and with local p below the same (the look-elsewhere effect:
   local p-values are far too optimistic).
2. **The trigger**: spectra without a signal but with a trigger turn-on at 20 GeV, searched with
   and without the efficiency in the background model.
3. **Signals by Asimov Z**: detection (global Z > 3), mass within the question's tolerance, against
   the private label ``signal_undetectable`` (Asimov Z < 3).

    python tools/check_bump_hunt.py                   # about 3 minutes
    python tools/check_bump_hunt.py --nulls 100 --signals 100

Writes ``bump_hunt.json`` and ``.md`` to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EXP = "scidatasim.experiments.resonances.logic.ResonanceExperiment"
OBS = {"targets": ["counts"], "operators": [{"type": "poisson"}]}
Z_BINS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6.01)]
ALPHAS = (0.1, 0.05, 0.01)


def run(seed, signal, trigger=None, model_trigger=True):
    from scipy import stats
    from scidatasim.engine import simulate
    from scidatasim.experiments.resonances.logic import bump_hunt
    cfg = dict(experiment_type=EXP, seed=seed, signal=signal, observation=OBS)
    if trigger:
        cfg["trigger"] = trigger
    sim = simulate(cfg)
    L = sim.logic
    found = bump_hunt(L.edges, sim.results["counts_observed"]["data"], L.resolution,
                      efficiency=L.eff if (trigger and model_trigger) else None)
    row = {"seed": seed, "signal": signal, "detected": found["detected"], "p_global": found["p_global"],
           "p_local": float(stats.norm.sf(found["z_local"])), "z_global": found["z_global"], "found_mass": found["mass"]}
    if signal:
        items = {q["template"]: q for q in L.questions(sim.results, None)}
        m = items["resonance_mass"]
        row.update(asimov=L.asimov(), undetectable=bool(items["new_resonance"]["signal_undetectable"]),
                   mass_pass=bool(found["detected"] and abs(found["mass"] - m["answer"]) <= m["tolerance"]))
    return row


def rate(rows, key):
    rows = [r for r in rows if key in r]
    return float(np.mean([bool(r[key]) for r in rows])) if rows else None


def summarise(nulls, trig_ignored, trig_modelled, signals):
    out = {"calibration": [], "trigger": {}, "by_asimov_z": [], "labels": {}}
    for a in ALPHAS:
        out["calibration"].append({"alpha": a, "global p < alpha": float(np.mean([r["p_global"] < a for r in nulls])),
                                   "local p < alpha": float(np.mean([r["p_local"] < a for r in nulls]))})
    out["trigger"] = {"efficiency ignored: detected": rate(trig_ignored, "detected"),
                      "efficiency modelled: detected": rate(trig_modelled, "detected"),
                      "found masses (ignored, median)": float(np.median([r["found_mass"] for r in trig_ignored]))}
    out["false_detections"] = rate(nulls, "detected")
    for lo, hi in Z_BINS:
        sel = [r for r in signals if lo <= r["asimov"] < hi]
        out["by_asimov_z"].append({"Asimov Z": f"{lo}-{hi:g}", "n": len(sel), "detected": rate(sel, "detected"),
                                   "mass passes": rate(sel, "mass_pass")})
    for flag in (False, True):
        sel = [r for r in signals if r["undetectable"] == flag]
        out["labels"][f"undetectable={flag}"] = {"n": len(sel), "detected": rate(sel, "detected")}
    return out


def markdown(s):
    def fmt(v):
        return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    def table(rows):
        cols = list(rows[0])
        return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + \
            ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    lines = ["# Bump hunt on planted truth", "", "## Calibration (no new resonance)", ""] + table(s["calibration"])
    lines += ["", f"Detections (global Z > 3) without a signal: {fmt(s['false_detections'])}", "", "## Trigger turn-on, no signal", ""]
    lines += [f"- {k}: {fmt(v)}" for k, v in s["trigger"].items()]
    lines += ["", "## Signals by Asimov Z", ""] + table(s["by_asimov_z"])
    lines += ["", "## Private label", ""] + [f"- {k}: n = {v['n']}, detected {fmt(v['detected'])}"
                                            for k, v in s["labels"].items()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nulls", type=int, default=400)
    parser.add_argument("--signals", type=int, default=300)
    parser.add_argument("--out", default=".")
    args = parser.parse_args(argv)
    trigger = {"threshold": 20.0, "width": 1.5}
    nulls = [run(1000 + i, False) for i in range(args.nulls)]
    n_trig = max(20, args.nulls // 10)
    trig_ignored = [run(2000 + i, False, trigger, model_trigger=False) for i in range(n_trig)]
    trig_modelled = [run(2000 + i, False, trigger, model_trigger=True) for i in range(n_trig)]
    signals = [run(3000 + i, True) for i in range(args.signals)]
    summary = summarise(nulls, trig_ignored, trig_modelled, signals)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "bump_hunt.json"), "w") as f:
        json.dump({"summary": summary}, f, indent=1, default=float)
    text = markdown(summary)
    with open(os.path.join(args.out, "bump_hunt.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
