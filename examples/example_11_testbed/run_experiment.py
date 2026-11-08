#!/usr/bin/env python
"""Example 11: the simulator as a testbed for leaks, shortcuts, contamination and agent fallacies.

Runs on the law-discovery suite of example 10 and writes ``report.md`` and ``report.json``:

1. Quality card and shortcut diagnostics: which questions can be answered without the data?
2. Leak audit of the public release (with a canary).
3. Contamination: a fresh twin of the suite; a solver that remembers a leaked answer key
   (at several contamination rates) against an honest solver; paired gap tests.
4. Canary probe with a model that saw the release and one that did not.
5. Agent fallacies: scripted agents with known faults in the interactive environment, and the
   flags :mod:`simulator.diagnostics.episodes` raises for each.

    python -m examples.example_11_testbed.run_experiment [--quick]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from examples.example_10_law_discovery.run_experiment import fit_best_law  # noqa: E402
from experiments.law_discovery.logic import law_suite_design, textbook_counterpart  # noqa: E402
from simulator.audit import audit_export  # noqa: E402
from simulator.bundle import check_answer  # noqa: E402
from simulator.config import load_config, yaml_load  # noqa: E402
from simulator.diagnostics import (add_canary, blind_baseline, canary_probe, card_markdown,  # noqa: E402
                                   contamination_gap, diagnose_episode, feature_shortcuts, law_alternatives,
                                   make_canary, make_twin, pair_tasks, public_features, quality_card,
                                   run_episodes, summarize)
from simulator.doe import run_design  # noqa: E402
from simulator.envs import ExperimentEnv  # noqa: E402
from simulator.export.tasks import collect_bundle_tasks  # noqa: E402
from simulator.expressions import evaluate  # noqa: E402
from simulator.persistence import load_experiment_record  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EX10 = os.path.join(HERE, "..", "example_10_law_discovery")
LAW_TEMPLATES = ("law_family", "law_formula", "extrapolation")


# ---------------------------------------------------------------------------- solvers
class HonestSolver:
    """Answers law questions by fitting every family to the run's public observations."""

    def __init__(self, design_dir):
        self.design_dir = design_dir
        self._fits = {}

    def fit(self, experiment_id):
        if experiment_id not in self._fits:
            rec = load_experiment_record(self.design_dir, experiment_id, verify=False)
            x = rec.output_data["x"]["data"]
            y = rec.output_data["y_observed"]["data"]
            self._fits[experiment_id] = fit_best_law(x, y)
        return self._fits[experiment_id]

    def __call__(self, task):
        family, formula = self.fit(task["experiment_id"])
        if task["template"] == "law_family":
            return family or ""
        if task["template"] == "law_formula":
            return formula or "0"
        if task["template"] == "extrapolation":
            x_star = float(re.findall(r"x = ([-+0-9.e]+)", task["question"])[0])
            return float(evaluate(formula, x=x_star)) if formula else 0.0
        return None


class WeakSolver:
    """Stands in for a model that cannot do the task: a fixed guess per template."""

    def __call__(self, task):
        return {"law_family": "power_law", "law_formula": "x", "extrapolation": 1.0}.get(task["template"])


def item_key(task):
    """What a scraper stores: the item as published (prompt and figure bytes)."""
    with open(task["image"], "rb") as f:
        figure = hashlib.sha256(f.read()).hexdigest()
    return hashlib.sha256((task["prompt"] + figure).encode()).hexdigest()


class MemorizingSolver:
    """Has seen a leaked answer key for a fraction ``rate`` of the original items; recites those,
    and answers the rest with its own ``solver``."""

    def __init__(self, leaked_tasks, rate, solver, seed=0):
        rng = np.random.default_rng(seed)
        self.memory = {item_key(t): t["answer"] for t in leaked_tasks if rng.random() < rate}
        self.solver = solver

    def __call__(self, task):
        key = item_key(task)
        return self.memory[key] if key in self.memory else self.solver(task)


def score_all(tasks, solver):
    return {t["id"]: float(check_answer(t, solver(t))) for t in tasks}


# ---------------------------------------------------------------------------- agents
def _measure(obs, log, x_min, x_max):
    return {"type": "experiment", "inputs": {"x_min": x_min, "x_max": x_max}}


def _data(log):
    xs, ys = [], []
    for s in log.experiments:
        m = (s["observation"].get("measurements") or {}).get("y_observed")
        if m:
            xs += m["x"]
            ys += m["values"]
    return np.array(xs, float), np.array(ys, float)


def make_agent(kind, domain):
    lo, hi = domain

    def agent(obs, log):
        n = len(log.experiments)
        if kind == "guesser":
            return {"type": "answer", "value": "2*x**1.5"}
        if kind == "narrow" and n == 0:
            return _measure(obs, log, lo, lo + 0.2 * (hi - lo))
        if kind == "repeater" and n < 3:
            return _measure(obs, log, lo, hi)
        if kind in ("fitter", "textbook") and n == 0:
            return _measure(obs, log, lo, hi)
        x, y = _data(log)
        family, formula = fit_best_law(x, y)
        if kind == "textbook" and family and formula:
            formula = textbook_counterpart(family, formula, domain) or formula
        if kind == "stubborn":  # ignores the measurement entirely after one look
            if n == 0:
                return _measure(obs, log, lo, hi)
            formula = "3*x"
        return {"type": "answer", "value": formula or "0"}
    return agent


AGENTS = {
    "fitter": "fits all families to measurements over the whole domain",
    "textbook": "fits, then snaps the structural constant to its textbook value",
    "guesser": "answers at once, without experiments",
    "narrow": "measures only the first fifth of the domain, then fits",
    "repeater": "repeats the same experiment three times, then fits (can be deliberate noise averaging)",
    "stubborn": "measures once, then answers a fixed formula regardless of the data",
}


# ---------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_11_testbed"))
    parser.add_argument("--quick", action="store_true", help="smaller suite and fewer episodes")
    parser.add_argument("--suite-seed", type=int, default=11)
    args = parser.parse_args()
    out = os.path.abspath(args.output_dir)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    report = {}
    md = ["# Testbed report: leaks, shortcuts, contamination, agent fallacies", ""]

    # -- suite ------------------------------------------------------------------
    base = load_config(os.path.join(EX10, "config.yaml"))
    base["observation"]["operators"][0]["snr_db"] = 30
    base["bundle"]["qa"]["exclude"] = []  # restore the generic questions example 10 drops, to show why
    replicates = 1 if args.quick else 2
    design_table = law_suite_design(snr_db=(30,), seed=args.suite_seed).drop(
        columns=["observation.operators.0.snr_db"])
    original_dir = os.path.join(out, "original")
    run_design(base, design_table, output_dir=original_dir, replicates=replicates, base_seed=1)
    tasks = collect_bundle_tasks(original_dir)
    md += [f"Suite: {len(design_table)} design points x {replicates} replicate(s), "
           f"{len({t['experiment_id'] for t in tasks})} runs, {len(tasks)} questions.", ""]

    # -- 1. quality and shortcuts ---------------------------------------------------
    card = quality_card(tasks)
    blind = blind_baseline(tasks)
    features = public_features(original_dir)
    import pandas as pd
    design_results = pd.read_csv(os.path.join(original_dir, "design_results.csv"))
    points = dict(zip(design_results["experiment_id"], design_results["design_point"]))
    # replicates of a design point share design-level answers: never use them as neighbours
    shortcuts = feature_shortcuts(tasks, features, n_permutations=100 if args.quick else 300, groups=points)
    report["quality"], report["blind"], report["feature_shortcuts"] = card, blind, shortcuts
    md += ["## 1. Quality card and shortcuts", "", card_markdown(card), "",
           "| template | blind accuracy | chance | figure/caption k-NN | permutation p |", "|---|---|---|---|---|"]
    for template in sorted(blind):
        b, s = blind[template], shortcuts.get(template, {})
        md.append(f"| {template} | {b['accuracy']:.2f}"
                  + (f" (position {b['position_accuracy']:.2f})" if "position_accuracy" in b else "")
                  + f" | {b.get('chance', float('nan')):.2f} | {s.get('accuracy', float('nan')):.2f} | "
                  + f"{s.get('p_value', float('nan')):.3f} |")
    degenerate = sorted(t for t, b in blind.items() if b["accuracy"] >= 0.5)
    md += ["", "Templates answerable at >= 50 % without the data: " + (", ".join(degenerate) or "none") + ".", ""]
    report["degenerate_templates"] = degenerate

    # -- 2. leak audit with canary ---------------------------------------------------
    from simulator.export.huggingface import export_huggingface
    release = export_huggingface(original_dir, os.path.join(out, "release"), name="sds-testbed-laws",
                                 overwrite=True)
    canary = make_canary("sds-testbed-laws")
    add_canary(release, canary)
    audit = audit_export(release, original_dir)
    report["audit"] = audit.to_dict()
    md += ["## 2. Leak audit of the public release", "", "```", audit.summary(), "```", "",
           f"Canary written to the release: `{canary}`.", ""]

    # -- 3. contamination: fresh twin --------------------------------------------------
    twin_dir = os.path.join(out, "twin")
    make_twin(original_dir, twin_dir, base_seed=2)
    twin_tasks = collect_bundle_tasks(twin_dir)
    law_tasks = [t for t in tasks if t["template"] in LAW_TEMPLATES]
    law_twin = [t for t in twin_tasks if t["template"] in LAW_TEMPLATES]
    pairs = pair_tasks(law_tasks, law_twin, original_dir, twin_dir)
    templates = {t["id"]: t["template"] for t in law_tasks}
    honest_o, honest_t = HonestSolver(original_dir), HonestSolver(twin_dir)
    md += ["## 3. Contamination: original vs. fresh twin", "",
           f"{len(pairs)} matched question pairs (law family, formula, extrapolation). A memorising solver "
           "recites a leaked answer key for a fraction of the original items (keyed by prompt and figure, as "
           "a scraper would store them) and otherwise answers with its own skill: the honest curve fitter, or a "
           "weak solver that cannot do the task.", "",
           "| solver | leaked fraction | original | twin | gap | 95 % CI | p |", "|---|---|---|---|---|---|---|"]
    report["contamination"] = {}
    for solver_name, own_o, own_t in (("fitter", honest_o, honest_t), ("weak", WeakSolver(), WeakSolver())):
        for rate in (0.0, 0.1, 0.3, 1.0):
            so = score_all(law_tasks, MemorizingSolver(law_tasks, rate, own_o))
            st = score_all(law_twin, MemorizingSolver(law_tasks, rate, own_t))
            gap = contamination_gap(so, st, pairs, templates, n_resamples=5000)
            report["contamination"][f"{solver_name}@{rate}"] = {
                "original": gap.mean_original, "twin": gap.mean_twin, "gap": gap.gap, "ci": gap.ci,
                "p_value": gap.p_value}
            md.append(f"| {solver_name} | {rate:.1f} | {gap.mean_original:.2f} | {gap.mean_twin:.2f} | "
                      f"{gap.gap:+.2f} | [{gap.ci[0]:+.2f}, {gap.ci[1]:+.2f}] | {gap.p_value:.4f} |")
    md.append("")

    # -- 4. canary probe -------------------------------------------------------------
    seen = canary_probe(lambda prompt: canary, canary)  # a model that memorised the release
    unseen = canary_probe(lambda prompt: "sds-canary:sds-testbed-laws:" + "0" * 36, canary)
    report["canary"] = {"seen": {k: v for k, v in seen.items() if k != "outputs"},
                        "unseen": {k: v for k, v in unseen.items() if k != "outputs"}}
    md += ["## 4. Canary probe", "",
           f"Model that saw the release: {seen['hex_matched']} hidden hex digits reproduced "
           f"(chance p = {seen['chance_p']:.1e}). Model that did not: {unseen['hex_matched']} "
           f"(chance p = {unseen['chance_p']:.2f}).", ""]

    # -- 5. agent fallacies ----------------------------------------------------------
    with open(os.path.join(EX10, "env.yaml"), encoding="utf-8") as f:
        spec = yaml_load(f)
    env = ExperimentEnv(spec)
    episodes = 8 if args.quick else 24
    md += ["## 5. Agent fallacies in the interactive environment", "",
           f"{episodes} episodes per agent; counterfactual laws; flags from `diagnose_episode`. "
           "`matches_textbook` applies only where the textbook form differs from the hidden law by more than "
           "the tolerance; its rate is over those episodes (count in brackets).", "",
           "| agent | behaviour | accuracy | flags (rate) |", "|---|---|---|---|"]
    report["agents"] = {}
    for name, behaviour in AGENTS.items():
        logs = run_episodes(env, make_agent(name, tuple(spec["domain"])), seeds=range(episodes))
        diagnoses = [diagnose_episode(log, law_alternatives(log)) for log in logs]
        summary = summarize(diagnoses)
        report["agents"][name] = summary
        flags = ", ".join(f"{k} {v['rate']:.2f}" + (f" (of {v['applicable']})" if v["applicable"] < episodes else "")
                          for k, v in summary["flags"].items() if v["rate"] > 0) or "none"
        md.append(f"| {name} | {behaviour} | {summary['accuracy']:.2f} | {flags} |")
    md.append("")

    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nReport: {os.path.join(out, 'report.md')}")


if __name__ == "__main__":
    main()
