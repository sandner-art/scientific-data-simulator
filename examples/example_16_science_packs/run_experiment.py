"""Example 16: the science packs scored on planted truth, with the standard baselines.

For each pack - transits, resonances, supernova cosmology, astrometric binaries - simulate a batch
of items (half with a signal, some with a planted instrument artefact), answer every question with
the field's standard method from ``scidatasim.baselines`` or the pack, and score the answers
against the private keys: overall, and split by the private labels (``signal_undetectable``,
``period_ambiguous``, ``instrument_artifact``). A baseline should fail where the data cannot settle
the question and where it ignores the instrument; this report shows where it does.

    python -m examples.example_16_science_packs.run_experiment            # about 2 minutes
    python -m examples.example_16_science_packs.run_experiment --quick    # about 20 s
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scidatasim.bundle import check_answer  # noqa: E402
from scidatasim.engine import simulate  # noqa: E402
from scidatasim.examples import prepare_output_dir  # noqa: E402

LABELS = ("signal_undetectable", "period_ambiguous", "instrument_artifact")


def transits(seed):
    from scidatasim.experiments.transits.model import transit_search
    sim = simulate({"experiment_type": "scidatasim.experiments.transits.logic.TransitExperiment", "seed": seed,
                    "span": 13.7, "gap": None, "period_range": [0.5, 6.0],
                    "observation": {"targets": ["flux"], "operators": [{"type": "gaussian", "sigma": 0.001}]}})
    found = transit_search(sim.results["time"]["data"], sim.results["flux_observed"]["data"], 1e-3, max_period=6.0)
    yes = found["detected"]
    answers = {"transit_present": "yes" if yes else "no", "transit_period": found["period"] if yes else 0.0,
               "transit_depth": found["depth_ppm"] if yes else 0.0}
    return sim.logic.questions(sim.results, None), answers


def resonances(seed):
    from scidatasim.experiments.resonances.logic import bump_hunt
    trigger = {"threshold": 20.0, "width": 1.5} if seed % 3 == 0 else None
    cfg = {"experiment_type": "scidatasim.experiments.resonances.logic.ResonanceExperiment", "seed": seed,
           "observation": {"targets": ["counts"], "operators": [{"type": "poisson"}]}}
    if trigger:
        cfg["trigger"] = trigger
    sim = simulate(cfg)
    L = sim.logic
    found = bump_hunt(L.edges, sim.results["counts_observed"]["data"], L.resolution,
                      efficiency=L.eff if (trigger and seed % 2 == 0) else None)  # half the trigger items ignore it
    yes = found["detected"]
    return L.questions(sim.results, None), {"new_resonance": "yes" if yes else "no",
                                            "resonance_mass": found["mass"] if yes else 0.0}


def cosmology(seed):
    from scidatasim.experiments.cosmology.logic import compare_models
    cfg = {"experiment_type": "scidatasim.experiments.cosmology.logic.SupernovaHubbleExperiment", "seed": seed,
           "observation": {"targets": ["m_b"], "operators": [{"type": "gaussian", "sigma": 0.15}]}}
    if seed % 3 == 0:
        cfg["calibration"] = {"offset": 0.04}
    sim = simulate(cfg)
    found = compare_models(sim.results["z"]["data"], sim.results["m_b_observed"]["data"], 0.15)
    return sim.logic.questions(sim.results, None), {
        "cosmological_constant": "evolving" if found["evolving"] else "cosmological constant",
        "omega_m": found["om"], "w0": found["w0"] if found["evolving"] else -1.0}


def astrometry(seed):
    from scidatasim.experiments.astrometry.fit import detect
    sim = simulate({"experiment_type": "scidatasim.experiments.astrometry.logic.AstrometricBinaryExperiment",
                    "seed": seed})
    ep, w = sim.logic.inputs_for_baseline(sim.results)
    found = detect(ep, w)
    yes = found["detected"]
    return sim.logic.questions(sim.results, None), {"astrometric_companion": "yes" if yes else "no",
                                                    "astrometric_period": found["period_best"] if yes else 0.0,
                                                    "astrometric_parallax": found["parallax"]}


PACKS = {"transits": transits, "resonances": resonances, "cosmology": cosmology, "astrometry": astrometry}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_16_science_packs"))
    parser.add_argument("--quick", action="store_true", help="6 items per pack instead of 30")
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()
    out = prepare_output_dir(args.output_dir)
    n = 6 if args.quick else 30
    rows = []
    for pack, run in PACKS.items():
        for i in range(n):
            items, answers = run(args.seed * 1000 + i)
            for item in items:
                rows.append({"pack": pack, "item": i, "template": item["template"], "answer": item["answer"],
                             "response": answers[item["template"]], "correct": check_answer(item, answers[item["template"]]),
                             **{label: bool(item.get(label, False)) for label in LABELS}})
        print(f"{pack}: {n} items")
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "scores.csv"), index=False)
    lines = ["# Science packs scored with the standard baselines", "",
             f"{n} items per pack, seed {args.seed}. Accuracy overall and on items with each private label.", "",
             "| pack | question | accuracy | n | " + " | ".join(f"{label} (n)" for label in LABELS) + " |",
             "|---|---|---|---|" + "---|" * len(LABELS)]
    for (pack, template), g in table.groupby(["pack", "template"], sort=False):
        cells = []
        for label in LABELS:
            sel = g[g[label]]
            cells.append(f"{sel['correct'].mean():.2f} ({len(sel)})" if len(sel) else "-")
        lines.append(f"| {pack} | {template} | {g['correct'].mean():.2f} | {len(g)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
