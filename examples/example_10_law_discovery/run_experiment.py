#!/usr/bin/env python3
# examples/example_10_law_discovery/run_experiment.py
"""Law-discovery suite and interactive environment, with a classical baseline.

Suite: every law family x {textbook, counterfactual} x two noise levels; each
run's bundle asks for the family, the formula (scored by function equivalence
on the data domain) and an extrapolated value. The figure shows observations
only, so every question is an inference task.

Baseline: least-squares fits of every family in the catalogue to the
observations, keeping the best by AIC. This is a deliberately simple reference
point for AI systems: anything claiming law discovery should beat it, and it
fails where noise or the counterfactual structure defeats fitting.

    python -m examples.example_10_law_discovery.run_experiment                 # suite + baseline score
    python -m examples.example_10_law_discovery.run_experiment --env 10        # interactive episodes
    python -m examples.example_10_law_discovery.run_experiment --export        # + Hugging Face dataset
"""
import argparse
import os
import re
import warnings

import numpy as np

from experiments.law_discovery.logic import LAWS, law_expression, law_suite_design
from simulator.config import load_config, yaml_load
from simulator.doe import run_design
from simulator.expressions import evaluate


def fit_best_law(x, y):
    """Least-squares fit of every catalogue family (all constants free and positive); best by AIC.
    Returns (family, expression) or (None, None)."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    best = (np.inf, None, None)
    for family, spec in LAWS.items():
        names = list(spec["priors"]) + list(spec["structural"])
        guess = [np.mean(spec["priors"][n]) if n in spec["priors"] else spec["structural"][n][0] for n in names]

        def model(xv, *p):  # full precision: the fitter's finite-difference steps are tiny
            return evaluate(law_expression(family, dict(zip(names, p)), digits=17), x=xv)
        for start in (guess, [g * 1.5 for g in guess], [g * 0.6 for g in guess]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # every catalogue constant is positive; the bound also keeps families apart
                    # (a/x**p with p < 0 would be a power law)
                    params, _ = curve_fit(model, x, y, p0=start, bounds=(1e-9, np.inf), max_nfev=4000)
            except (RuntimeError, ValueError, TypeError):
                continue
            residual = y - model(x, *params)
            rss = float(np.sum(residual ** 2))
            if not np.isfinite(rss):
                continue
            aic = len(x) * np.log(rss / len(x) + 1e-300) + 2 * len(names)
            if aic < best[0]:
                best = (aic, family, law_expression(family, dict(zip(names, params))))
    return best[1], best[2]


def run_suite(args):
    from simulator.bundle import check_answer
    from simulator.export.tasks import collect_bundle_tasks
    base = load_config(args.config)
    design = run_design(base, law_suite_design(snr_db=(40, 20), seed=args.suite_seed), output_dir=args.output_dir,
                        replicates=args.replicates)
    print(f"{len(design.table)} hidden laws in {design.output_dir}")
    scores = {"law_family": [], "law_formula": [], "extrapolation": []}
    for result, row in zip(design.results, design.table.to_dict("records")):
        family, formula = fit_best_law(result["x"]["data"], result["y_observed"]["data"])
        tasks = [t for t in collect_bundle_tasks(os.path.join(args.output_dir, _run_dir(args.output_dir,
                                                                                      row["experiment_id"])))]
        for task in tasks:
            if task["template"] == "law_family":
                scores["law_family"].append(check_answer(task, family or ""))
            elif task["template"] == "law_formula":
                scores["law_formula"].append(check_answer(task, formula or "0"))
            elif task["template"] == "extrapolation" and formula:
                x_star = float(re.findall(r"x = ([-+0-9.e]+)", task["question"])[0])
                scores["extrapolation"].append(check_answer(task, float(evaluate(formula, x=x_star))))
    for name, values in scores.items():
        print(f"baseline {name:14s}: {np.mean(values):.2f} correct ({len(values)} questions)")
    if args.export:
        from simulator.export.huggingface import export_huggingface
        path = export_huggingface(args.output_dir, os.path.join(args.output_dir, "hf_dataset"),
                                  name="sds-law-discovery", overwrite=True,
                                  description="Hidden-law discovery tasks (textbook and counterfactual).")
        print(f"Hugging Face dataset folder: {path}")
        from simulator.audit import audit_export
        print(audit_export(path, args.output_dir).summary())  # try to recover the answers from it


def _run_dir(output_dir, experiment_id):
    return next(d for d in os.listdir(output_dir) if d.endswith(experiment_id))


def run_env(args):
    from simulator.envs import ExperimentEnv
    with open(args.env_spec, encoding="utf-8") as f:
        env = ExperimentEnv(yaml_load(f))
    rewards = []
    for episode in range(args.env):
        env.reset(seed=episode)
        out = env.step({"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}})
        m = out["observation"]["measurements"]["y_observed"]
        _, formula = fit_best_law(np.array(m["x"], float), np.array(m["values"], float))
        result = env.step({"type": "answer", "value": formula or "0"})
        rewards.append(result["reward"])
    print(f"baseline in the environment: mean reward {np.mean(rewards):.2f} over {args.env} episodes")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config", default=os.path.join(here, "config.yaml"))
    parser.add_argument("--env-spec", default=os.path.join(here, "env.yaml"))
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_10_law_discovery"))
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--env", type=int, default=0, help="play this many environment episodes instead")
    parser.add_argument("--export", action="store_true", help="also export a Hugging Face dataset")
    parser.add_argument("--suite-seed", type=int, default=None,
                        help="fix the (otherwise secret, random) row order of the suite, for a reproducible suite")
    args = parser.parse_args()
    if args.env:
        run_env(args)
    else:
        run_suite(args)


if __name__ == "__main__":
    main()
