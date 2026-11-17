"""Example 15: agent-based models with Mesa, and a planted effect to recover.

Part 1 runs the three models of the agent-based pack once each, with bundles: Schelling
segregation (social science), an epidemic with a randomised trial inside it (epidemiology), and
wealth exchange with saving (economics). Each has a planted parameter that the bundle asks back.

Part 2 is a simulation study (ADEMP) of the trial: how well do two estimators recover the planted
treatment effect, the relative reduction of infection risk per contact?

* attack-rate ratio: 1 - (share of treated infected by the end) / (share of controls infected);
* hazard ratio: 1 - (new infections per susceptible-step, treated) / (the same, control), pooled
  over the whole epidemic;
* stratified hazard ratio: the same, compared within each time step (Mantel-Haenszel).

When the epidemic is large, nearly everyone is infected in both arms and the attack-rate ratio is
pulled towards zero. Pooling over time biases the crude hazard ratio too: treated agents stay
susceptible into the late epidemic, when there is less infection about. Comparing within steps
removes most of that.

    python -m examples.example_15_agent_based.run_experiment           # about 45 s
    python -m examples.example_15_agent_based.run_experiment --quick   # about 12 s

Needs Mesa: pip install "scientific-data-simulator[mesa]".
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scidatasim.ademp import SimulationStudy  # noqa: E402
from scidatasim.engine import SimulatorEngine  # noqa: E402
from scidatasim.examples import prepare_output_dir  # noqa: E402

PACK = "scidatasim.experiments.social.logic."
ARMS = ["new_treated", "new_control", "susceptible_treated", "susceptible_control"]
TRIAL = {"experiment_type": PACK + "EpidemicTrialExperiment", "n_steps": 200, "effect": 0.4, "beta": 0.04,
         "observation": {"targets": ARMS, "operators": []}}  # complete case reports per arm


def hazard_ratio(data):
    """1 - (infections per susceptible-step, treated) / (the same, control)."""
    def rate(arm):
        new, sus = data[f"new_{arm}_observed"][1:], data[f"susceptible_{arm}_observed"][:-1]
        return new.sum() / sus.sum()
    return {"effect": 1.0 - rate("treated") / rate("control")}


def stratified_hazard_ratio(data):
    """Mantel-Haenszel rate ratio over time steps: treated and controls compared within each step,
    where both face the same amount of infection."""
    new_t, new_c = data["new_treated_observed"][1:], data["new_control_observed"][1:]
    sus_t, sus_c = data["susceptible_treated_observed"][:-1], data["susceptible_control_observed"][:-1]
    total = sus_t + sus_c
    keep = total > 0
    num = np.sum(new_t[keep] * sus_c[keep] / total[keep])
    den = np.sum(new_c[keep] * sus_t[keep] / total[keep])
    return {"effect": 1.0 - num / den}


def attack_rate_ratio(data):
    """1 - (share of the treated infected by the end) / (share of the controls infected)."""
    def attack(arm):
        sus = data[f"susceptible_{arm}_observed"]
        return (sus[0] - sus[-1]) / sus[0]
    return {"effect": 1.0 - attack("treated") / attack("control")}


def part1(out):
    engine = SimulatorEngine(os.path.join(out, "runs"), verbose=False)
    runs = {
        "segregation": {"experiment_type": PACK + "SegregationExperiment", "n_steps": 40, "homophily": 0.45},
        "epidemic_trial": dict(TRIAL, observation={"targets": ["new_treated", "new_control"],
                                                   "operators": [{"type": "poisson", "scale": 1.0}]}),
        "wealth_exchange": {"experiment_type": PACK + "WealthExchangeExperiment", "n_steps": 300, "saving": 0.4},
    }
    lines = ["## Part 1: one run of each model", ""]
    for name, cfg in runs.items():
        run = engine.run_config(dict(cfg, seed=15, bundle={"figure": {"show_latent": True}}))
        qa = json.load(open(os.path.join(run.experiment_dir, "bundle", "qa.json")))
        planted = next(q for q in qa if q["template"] == "planted_effect")
        lines += [f"- **{name}** ({run.experiment_id}): planted answer {planted['answer']!r}; "
                  f"{len(qa)} questions; figure in `{os.path.relpath(run.experiment_dir, out)}/bundle/`."]
        print(f"{name}: planted {planted['answer']!r}, {len(qa)} questions")
    return lines + [""]


def part2(out, n_sim):
    effects, betas = (0.0, 0.3, 0.6), (0.02, 0.05)
    mechanisms = pd.DataFrame([{"effect": e, "beta": b} for b in betas for e in effects])
    labels = [f"effect {e}, beta {b} ({'small' if b == betas[0] else 'large'} epidemic)"
              for b in betas for e in effects]
    study = SimulationStudy(
        aims="Recovering a randomised treatment's effect on infection risk from case counts per arm.",
        base_config=TRIAL, mechanisms=mechanisms, estimands={"effect": "effect"},
        methods={"attack-rate ratio": attack_rate_ratio, "hazard ratio": hazard_ratio,
                 "stratified hazard ratio": stratified_hazard_ratio},
        n_sim=n_sim, base_seed=15, mechanism_labels=labels)
    result = study.run(os.path.join(out, "study"))
    perf = result.performance.merge(result.mechanisms[["design_point", "label"]], on="design_point")
    table = perf[["label", "method", "bias", "bias_mcse", "empse", "mse"]].round(3)
    with pd.option_context("display.width", 140, "display.max_colwidth", 50):
        print(table.to_string(index=False))
    rows = ["## Part 2: estimating the planted treatment effect", "",
            "| mechanism | method | bias (MCSE) | empirical SE | MSE |", "|---|---|---|---|---|"]
    rows += [f"| {r['label']} | {r['method']} | {r['bias']:.3f} ({r['bias_mcse']:.3f}) | {r['empse']:.3f} | "
             f"{r['mse']:.4f} |" for r in perf.to_dict("records")]
    return rows + ["", f"{n_sim} repetitions per mechanism; full tables in `study/report.md`.", ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_15_agent_based"))
    parser.add_argument("--quick", action="store_true", help="20 repetitions instead of 100")
    args = parser.parse_args()
    out = prepare_output_dir(args.output_dir)  # keeps the runner's run.log / run.json
    lines = ["# Agent-based models with a planted effect", ""] + part1(out) + part2(out, 20 if args.quick else 100)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {os.path.join(out, 'report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
