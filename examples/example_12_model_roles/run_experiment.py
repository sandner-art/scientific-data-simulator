#!/usr/bin/env python
"""Example 12: a language model as attacker, question writer and agent, with logged, replayable calls.

    python -m examples.example_12_model_roles.run_experiment                       # scripted stand-in
    python -m examples.example_12_model_roles.run_experiment --model mymodule:ask --model-name provider/model-x

``--model`` names a function ``ask(prompt: str) -> str`` (``module:function``) that calls your
model; without it a scripted stand-in plays every role, so the pipeline can be run and tested
offline. The stand-in is not a language model and its numbers say nothing about models.

Steps (written to ``report.md`` in the output folder):
1. a law-discovery suite and its Hugging Face release, leak-audited;
2. attacker: the model answers the release's questions with the evidence removed;
3. writer: the model rephrases every question; rewrites are accepted only if they keep the
   numbers and options and add no answer; the rewritten set is re-checked for shortcuts and leaks;
4. agent: the model plays the interactive law-discovery environment; episodes are diagnosed;
5. provenance: every call is in ``logs/<role>.jsonl``; ``llm_provenance.json`` summarises them;
   the agent episodes are replayed from the log without the model and must match.
"""
import argparse
import importlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from examples.example_10_law_discovery.run_experiment import fit_best_law  # noqa: E402
from experiments.law_discovery.logic import law_suite_design  # noqa: E402
from simulator.ai import (LLMAgent, LoggedClient, ReplayClient, llm_attack, paraphrase_tasks,  # noqa: E402
                          roles_of, write_llm_provenance)
from simulator.ai.writer import WRITER_INSTRUCTIONS  # noqa: E402
from simulator.audit import audit_export  # noqa: E402
from simulator.config import load_config, yaml_load  # noqa: E402
from simulator.diagnostics import (card_markdown, diagnose_episode, feature_shortcuts,  # noqa: E402
                                   law_alternatives, quality_card, run_episodes, summarize)
from simulator.doe import run_design  # noqa: E402
from simulator.envs import ExperimentEnv  # noqa: E402
from simulator.examples import prepare_output_dir  # noqa: E402
from simulator.export.tasks import collect_bundle_tasks, public_view, write_tasks_jsonl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EX10 = os.path.join(HERE, "..", "example_10_law_discovery")


# ---------------------------------------------------------------------------- scripted stand-in
def scripted_model(prompt: str) -> str:
    """Not a language model: deterministic replies per role, so the pipeline runs offline."""
    if prompt.startswith(WRITER_INSTRUCTIONS):
        question = prompt.split("QUESTION\n", 1)[1].split("\n\n")[0]
        return "Looking at the chart: " + question[0].lower() + question[1:]
    if "dataset release for answer leakage" in prompt:
        ids = re.findall(r'"id": "([^"]+)"', prompt)
        return json.dumps({i: "power_law" for i in ids})  # a blind guess for everything
    if "EPISODE SO FAR" not in prompt:
        return '{"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}}'
    xs, ys = [], []
    for result in re.findall(r"^Result: (\{.*\})", prompt, flags=re.M):
        for entry in json.loads(result).get("measurements", {}).values():
            xs += entry.get("x", [])
            ys += entry.get("values", [])
    _, formula = fit_best_law(np.array(xs, float), np.array(ys, float))
    return json.dumps({"type": "answer", "value": formula or "0"})


def load_model(spec):
    if spec in (None, "scripted"):
        return scripted_model, "scripted/stand-in"
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name or "ask"), None


# ---------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_12_model_roles"))
    parser.add_argument("--model", default="scripted", help="module:function calling your model, or 'scripted'")
    parser.add_argument("--model-name", default=None, help="model name and version to record (required with --model)")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--suite-seed", type=int, default=12)
    args = parser.parse_args()
    ask, default_name = load_model(args.model)
    model_name = args.model_name or default_name
    if not model_name:
        parser.error("--model-name is required with --model (record what you ran)")
    out = prepare_output_dir(args.output_dir)  # keeps the runner's run.log / run.json
    os.makedirs(os.path.join(out, "logs"))
    logs = {role: os.path.join(out, "logs", f"{role}.jsonl") for role in ("attacker", "writer", "agent")}
    clients = {role: LoggedClient(ask, path, role=role, model=model_name) for role, path in logs.items()}
    md = [f"# Model roles report ({model_name})", ""]
    if model_name == "scripted/stand-in":
        md += ["*Scripted stand-in, not a language model: these numbers only show that the pipeline works.*", ""]

    # 1. suite and release ----------------------------------------------------------
    base = load_config(os.path.join(EX10, "config.yaml"))
    table = law_suite_design(snr_db=(30,), seed=args.suite_seed).drop(columns=["observation.operators.0.snr_db"])
    runs = os.path.join(out, "runs")
    run_design(base, table, output_dir=runs, base_seed=1)
    from simulator.export.huggingface import export_huggingface
    release = export_huggingface(runs, os.path.join(out, "release"), name="sds-model-roles", overwrite=True)
    audit = audit_export(release, runs)
    tasks = collect_bundle_tasks(runs)
    md += ["## 1. Release", "", f"{len(table)} runs, {len(tasks)} questions. Leak audit: "
           f"{'PASS' if audit.ok else 'FAIL'}.", ""]

    # 2. attacker ----------------------------------------------------------------
    attack = llm_attack(clients["attacker"], release, runs, batch_size=40)
    md += ["## 2. Attacker (evidence removed)", "", "```", attack.summary(), "```", ""]

    # 3. writer ------------------------------------------------------------------
    rewritten, report = paraphrase_tasks(tasks, clients["writer"], writer=model_name)
    public = os.path.join(out, "rewritten_public")
    os.makedirs(public)
    write_tasks_jsonl([public_view(t) for t in rewritten], os.path.join(public, "tasks.jsonl"))
    rewritten_audit = audit_export(public, runs)
    # wording cues: does the length of each question predict its answer? (templates vs rewrites)
    def length_features(task_list):
        feats = {}
        for t in task_list:
            feats.setdefault(t["experiment_id"], {})[f"chars_{t['template']}"] = float(len(t["question"]))
        return feats
    cues_before = feature_shortcuts(tasks, length_features(tasks), n_permutations=100)
    cues_after = feature_shortcuts(rewritten, length_features(rewritten), n_permutations=100)
    md += ["## 3. Writer", "",
           f"{report.accepted} of {report.total} rewrites accepted; rejected by reason: "
           f"{json.dumps(report.rejected) if report.rejected else 'none'}. Leak audit of the rewritten public "
           f"set: {'PASS' if rewritten_audit.ok else 'FAIL'}.", "",
           "Do question lengths predict the answers (k-NN on lengths, permutation p)?", "",
           "| template | templates: accuracy, p | rewritten: accuracy, p |", "|---|---|---|"]
    for t in sorted(cues_after):
        b, a = cues_before.get(t, {}), cues_after[t]
        md.append(f"| {t} | {b.get('accuracy', float('nan')):.2f}, {b.get('p_value', float('nan')):.2f} | "
                  f"{a['accuracy']:.2f}, {a['p_value']:.2f} |")
    md += ["", card_markdown(quality_card(rewritten)), ""]
    if report.examples:
        md += ["Rejected examples:", ""] + [f"- `{e['id']}`: {', '.join(e['problems'])}" for e in report.examples[:5]]
        md.append("")

    # 4. agent -------------------------------------------------------------------
    with open(os.path.join(EX10, "env.yaml"), encoding="utf-8") as f:
        spec = yaml_load(f)
    agent = LLMAgent(clients["agent"])
    episodes = run_episodes(ExperimentEnv(spec), agent, seeds=range(args.episodes))
    summary = summarize([diagnose_episode(log, law_alternatives(log)) for log in episodes])
    flags = ", ".join(f"{k} {v['rate']:.2f}" for k, v in summary["flags"].items() if v["rate"] > 0) or "none"
    md += ["## 4. Agent", "", f"{args.episodes} episodes: accuracy {summary['accuracy']:.2f}, "
           f"mean experiments {summary['mean_experiments']:.1f}, flags: {flags}. "
           f"Parse failures: {agent.stats['parse_failures']}, gave up: {agent.stats['gave_up']}.", ""]

    # 5. provenance and replay ------------------------------------------------------
    prov = write_llm_provenance(out, list(logs.values()))
    replayed = run_episodes(ExperimentEnv(spec), LLMAgent(ReplayClient(logs["agent"], role="agent")),
                            seeds=range(args.episodes))
    same = [a.steps == b.steps for a, b in zip(episodes, replayed)]
    roles = roles_of(list(logs.values()))
    md += ["## 5. Provenance and replay", "",
           f"Calls logged per role: " + ", ".join(f"{r} {c.calls}" for r, c in clients.items()) + ". "
           f"Summary: `{os.path.relpath(prov, out)}`. Agent episodes replayed from the log without the model: "
           f"{sum(same)}/{len(same)} identical.", ""]
    md += ["Roles per model: " + "; ".join(f"{m}: {', '.join(sorted(r))}" for m, r in roles.items()) + ". "
           "Anything evaluated on the rewritten questions (a solver, an attacker) should be a model other than "
           "the writer.", ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"Report: {os.path.join(out, 'report.md')}")


if __name__ == "__main__":
    main()
