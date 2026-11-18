"""Example 13: a frozen law-discovery release.

Builds one fixed law-discovery dataset to publish and cite: the suite of example 10, run under
secret seeds, exported for the Hugging Face Hub with answers withheld, audited for leaks, with
reference baselines and checksums. The answer key stays in a private folder.

    python -m examples.example_13_law_release.run_experiment --output_dir release
    python -m examples.example_13_law_release.run_experiment --output_dir release --score predictions.jsonl
    python -m examples.example_13_law_release.run_experiment --output_dir release \\
        --solve mymodels.client:ask --model-name provider/model-2026-11

Layout of ``--output_dir``:

* ``public/`` - upload this: data, figures, questions without answers, the dataset card
  (``README.md`` with the baselines), Croissant metadata, ``CANARY.txt`` and ``SHA256SUMS``.
* ``private/`` - never upload: the runs (the answer key), ``answer_key.jsonl``, the secret seeds
  (``release.json``), the audit, a fresh twin for later contamination tests, scores, model logs.

The build stops without checksums if the leak audit of ``public/`` fails.

``--score FILE`` scores predictions (JSON lines ``{"id": ..., "response": ...}``; free text is
accepted) against the private key. ``--solve module:function`` asks a model every question through
:class:`scidatasim.ai.LoggedClient` (role ``solver``) and scores it; the log lets the evaluation be
replayed without the model.
"""
import argparse
import hashlib
import importlib
import json
import os
import re
import secrets
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from examples.example_10_law_discovery.run_experiment import fit_best_law  # noqa: E402
from scidatasim.audit import audit_export  # noqa: E402
from scidatasim.config import load_config  # noqa: E402
from scidatasim.diagnostics import (accuracy_by, add_canary, approximator_baseline, blind_baseline,  # noqa: E402
                                    card_markdown, make_canary, make_twin, quality_card)
from scidatasim.doe import run_design  # noqa: E402
from scidatasim.examples import prepare_output_dir  # noqa: E402
from scidatasim.experiments.law_discovery.logic import law_suite_design  # noqa: E402
from scidatasim.export.tasks import collect_bundle_tasks, read_tasks_jsonl, score, write_tasks_jsonl  # noqa: E402
from scidatasim.expressions import evaluate  # noqa: E402
from scidatasim.persistence import load_experiment_record  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EX10 = os.path.join(HERE, "..", "example_10_law_discovery")
TEMPLATES = ("law_family", "law_formula", "extrapolation")


# ---------------------------------------------------------------------------- helpers
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_checksums(folder):
    """``SHA256SUMS`` over every file in ``folder`` (``sha256sum -c SHA256SUMS`` checks it)."""
    lines = []
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, folder).replace(os.sep, "/")
            if rel != "SHA256SUMS":
                lines.append(f"{sha256(path)}  {rel}")
    target = os.path.join(folder, "SHA256SUMS")
    with open(target, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(lines, key=lambda line: line.split("  ", 1)[1])) + "\n")
    return target, len(lines)


def observations(runs_dir, tasks):
    """``{experiment_id: (x, y_observed)}``: the measurements every solver sees."""
    out = {}
    for eid in sorted({t["experiment_id"] for t in tasks}):
        data = load_experiment_record(runs_dir, eid, verify=False).output_data
        out[eid] = (np.asarray(data["x"]["data"], float), np.asarray(data["y_observed"]["data"], float))
    return out


def classical_answers(tasks, obs):
    """The curve-fitting baseline of example 10: fit every family, keep the best by AIC."""
    fits = {eid: fit_best_law(x, y) for eid, (x, y) in obs.items()}
    answers = {}
    for t in tasks:
        family, formula = fits[t["experiment_id"]]
        if t["template"] == "law_family" and family:
            answers[t["id"]] = family
        elif t["template"] == "law_formula" and formula:
            answers[t["id"]] = formula
        elif t["template"] == "extrapolation" and formula:
            x_star = float(re.findall(r"x = ([-+0-9.e]+)", t["question"])[0])
            answers[t["id"]] = float(evaluate(formula, x=x_star))
    return answers


def score_answers(tasks, answers):
    """Accuracy per template (a missing answer is wrong), the formula score without the items
    the private ``polynomial_reachable`` label marks, and the extrapolation score without those
    ``zero_reachable`` or ``naive_reachable`` mark (0, the last observed value or a line through the
    tail passes them)."""
    hits = {t["id"]: bool(t["id"] in answers and score(t, answers[t["id"]])) for t in tasks}
    out = {"n": len(tasks), "answered": sum(t["id"] in answers for t in tasks),
           "accuracy": sum(hits.values()) / max(1, len(tasks)), "templates": {}}
    for template in sorted({t["template"] for t in tasks}):
        group = [t for t in tasks if t["template"] == template]
        out["templates"][template] = {"n": len(group), "accuracy": sum(hits[t["id"]] for t in group) / len(group)}
    formula = [t for t in tasks if t["template"] == "law_formula"]
    if formula:
        split = accuracy_by(formula, hits, "polynomial_reachable")
        out["law_formula_without_polynomial_reachable"] = split.get("False", {}).get("accuracy")
    extrapolation = [dict(t, discriminating=not (t.get("zero_reachable") or t.get("naive_reachable")))
                     for t in tasks if t["template"] == "extrapolation"]
    if extrapolation:
        split = accuracy_by(extrapolation, hits, "discriminating")
        out["extrapolation_without_zero_or_naive_reachable"] = split.get("True", {}).get("accuracy")
    return out


def results_table(rows):
    """Markdown rows: name -> score_answers() result."""
    head = ["| solver | " + " | ".join(TEMPLATES) + " | formula without polynomial-reachable items | "
            "extrapolation without zero- or naive-reachable items |",
            "|---|" + "---|" * (len(TEMPLATES) + 2)]
    lines = []
    for name, r in rows.items():
        cells = [f"{r['templates'][t]['accuracy']:.2f}" if t in r["templates"] else "" for t in TEMPLATES]
        rest = r.get("law_formula_without_polynomial_reachable")
        rest_x = r.get("extrapolation_without_zero_or_naive_reachable")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {'' if rest is None else f'{rest:.2f}'} | "
                     f"{'' if rest_x is None else f'{rest_x:.2f}'} |")
    return head + lines


def load_client(spec):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name or "ask")


# ---------------------------------------------------------------------------- build
def build(args):
    out = prepare_output_dir(args.output_dir)  # keeps the runner's run.log / run.json
    public, private = os.path.join(out, "public"), os.path.join(out, "private")
    runs = os.path.join(private, "runs")
    snr = (40,) if args.quick else tuple(args.snr_db)
    replicates = 1 if args.quick else args.replicates
    # Secret seeds: with them anyone could regenerate the runs and their answers.
    suite_seed = secrets.randbits(62) if args.suite_seed is None else args.suite_seed
    base_seed = secrets.randbits(62) if args.base_seed is None else args.base_seed

    base = load_config(os.path.join(EX10, "config.yaml"))
    # Only the law questions: the generic chart read-outs (max, min, final value) are not law
    # discovery, and some are predictable without the data (min: 0.67 blind in a trial build).
    base["bundle"]["qa"]["exclude"] = sorted(set(base["bundle"]["qa"]["exclude"]) | {"max", "min", "final"})
    design = law_suite_design(snr_db=snr, seed=suite_seed)
    run_design(base, design, output_dir=runs, replicates=replicates, base_seed=base_seed)
    tasks = collect_bundle_tasks(runs)
    write_tasks_jsonl(tasks, os.path.join(private, "answer_key.jsonl"), include_answers=True, include_seeds=True)
    print(f"{len(design)} design points x {replicates} replicate(s): "
          f"{len({t['experiment_id'] for t in tasks})} runs, {len(tasks)} questions")

    from scidatasim.export.huggingface import export_huggingface
    export_huggingface(runs, public, name=args.name, version=args.version, overwrite=True,
                       date_published=date.today().isoformat(), url=args.url,
                       pretty_name="Scientific Data Simulator: hidden-law discovery",
                       description=("Hidden-law discovery from noisy measurements: nine families of laws y = f(x), "
                                    "each with textbook or counterfactual constants, at "
                                    + " and ".join(f"{s} dB" for s in snr) + ". Questions ask for the family, "
                                    "the formula and an extrapolated value. Answers are withheld."))

    # Reference baselines, from the measurements alone.
    obs = observations(runs, tasks)
    rows = {"curve fitting (all families, AIC)": score_answers(tasks, classical_answers(tasks, obs))}
    poly = approximator_baseline(tasks, obs).get("law_formula", {})
    blind = blind_baseline(tasks)
    card = quality_card(tasks)
    reachable = sum(bool(t.get("polynomial_reachable")) for t in tasks if t["template"] == "law_formula")
    n_formula = sum(t["template"] == "law_formula" for t in tasks)
    zero = sum(bool(t.get("zero_reachable")) for t in tasks if t["template"] == "extrapolation")
    naive = sum(bool(t.get("naive_reachable")) for t in tasks if t["template"] == "extrapolation")
    n_extrapolation = sum(t["template"] == "extrapolation" for t in tasks)

    lines = ["", "## Baselines", "",
             "Accuracy of reference solvers that see only the measurements (higher is better):", "",
             *results_table(rows), "",
             f"- A polynomial fitted to the measurements and submitted as the formula passes "
             f"{poly.get('accuracy', 0):.2f} of formula questions ({reachable} of {n_formula}). Formulas are "
             "scored on the data range and half of it beyond, so curve fitting alone rarely passes; power laws "
             "with exponents near 1 or 2 are the exception; the column \"formula without polynomial-reachable items\" "
             "leaves those items out.",
             "- Answering without the data (majority answer, median, from the other questions): "
             + ", ".join(f"{t} {b['accuracy']:.2f}" for t, b in sorted(blind.items())) + ".",
             *[f"- One constant answer, chosen on the other questions, passes {b['constant_accuracy']:.2f} of "
               f"{t} questions ({b['constant']:.3g})."
               for t, b in sorted(blind.items()) if b.get("constant_accuracy", 0) >= 0.05],
             f"- Of {n_extrapolation} extrapolation questions, {naive} pass with the last observed value or a line "
             f"through the last quarter of the data (laws that barely change beyond it, such as plateaus) and "
             f"{zero} with the answer 0 (laws that have decayed below the tolerance). Private labels mark them; "
             "the last column leaves them out.", "",
             "Answers are withheld. To score predictions, send them to the maintainers or use the scoring "
             "mode of example 13 in the Scientific Data Simulator repository with the private key.", ""]
    readme = os.path.join(public, "README.md")
    with open(readme, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    canary = make_canary(args.name)
    add_canary(public, canary)

    # Audit the public folder exactly as it will be uploaded.
    audit = audit_export(public, private=runs)
    with open(os.path.join(private, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit.to_dict(), f, indent=2, default=str)
    print(audit.summary())
    if not audit.ok:
        print("The release leaks answers: not finished (no checksums). See private/audit.json.")
        return 1

    twin_dir = None
    if not args.quick and not args.no_twin:  # held back, never released: for contamination tests later
        twin_dir = os.path.join(private, "twin")
        make_twin(runs, twin_dir, base_seed=secrets.randbits(62))

    sums, n_files = write_checksums(public)
    record = {"name": args.name, "version": args.version, "date": date.today().isoformat(),
              "suite_seed": suite_seed, "base_seed": base_seed, "snr_db": list(snr), "replicates": replicates,
              "runs": len(obs), "questions": len(tasks), "canary": canary, "audit_ok": audit.ok,
              "public_files": n_files, "sha256sums": sha256(sums), "twin": bool(twin_dir),
              "baselines": rows, "polynomial_formula_accuracy": poly.get("accuracy"),
              "polynomial_reachable": {"n": reachable, "of": n_formula},
              "zero_reachable": {"n": zero, "of": n_extrapolation},
              "naive_reachable": {"n": naive, "of": n_extrapolation}}
    with open(os.path.join(private, "release.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    with open(os.path.join(private, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join([f"# Release {args.name} {args.version}", "",
                           f"{len(obs)} runs, {len(tasks)} questions; leak audit PASS; {n_files} public files "
                           f"(SHA256SUMS {record['sha256sums'][:12]}...).", "",
                           "## Baselines", "", *results_table(rows), "",
                           "## Quality card", "", card_markdown(card), "",
                           "## Secrets (do not publish)", "", "- seeds and canary: `release.json`",
                           "- answer key: `answer_key.jsonl`, `runs/`", ""]))
    print(f"\nPublic release: {public} ({n_files} files)\nPrivate: {private} (answer key, seeds, audit, report)")
    print("Next: upload public/ (see the README of example 13), then cite its DOI.")
    return 0


# ---------------------------------------------------------------------------- score / solve
def key_tasks(output_dir):
    path = os.path.join(output_dir, "private", "answer_key.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"no answer key at {path}: build the release first")
    return read_tasks_jsonl(path)


def save_scores(output_dir, label, result):
    folder = os.path.join(output_dir, "private", "scores")
    os.makedirs(folder, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
    with open(os.path.join(folder, f"{safe}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\n".join(results_table({label: result})))
    print(f"answered {result['answered']} of {result['n']}; overall {result['accuracy']:.2f}")


def score_file(args):
    tasks = key_tasks(args.output_dir)
    answers = {}
    with open(args.score, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                answers[row["id"]] = row.get("response", row.get("answer"))
    save_scores(args.output_dir, args.label or os.path.basename(args.score), score_answers(tasks, answers))
    return 0


def solve(args):
    from scidatasim.ai import LoggedClient
    from scidatasim.ai.solver import solve_tasks
    if not args.model_name:
        raise SystemExit("--model-name is required with --solve (it is recorded with every call)")
    tasks = key_tasks(args.output_dir)[: args.limit or None]
    log = os.path.join(args.output_dir, "private", "llm", re.sub(r"[^A-Za-z0-9._-]+", "_", args.model_name) + ".jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    client = LoggedClient(load_client(args.solve), log, role="solver", model=args.model_name)
    responses = {}
    solve_tasks(tasks, client, responses=responses)
    save_scores(args.output_dir, args.model_name, score_answers(tasks, responses))
    print(f"model log: {log}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=os.path.join("experiments_output", "example_13_law_release"))
    parser.add_argument("--quick", action="store_true", help="one noise level, one replicate, no twin")
    parser.add_argument("--name", default="sds-law-discovery", help="dataset name (Hub repository id)")
    parser.add_argument("--version", default="1.0.0", help="dataset version")
    parser.add_argument("--url", default=None, help="dataset URL for the card and Croissant")
    parser.add_argument("--snr-db", type=float, nargs="+", default=[40, 20])
    parser.add_argument("--replicates", type=int, default=3, help="independent laws per design point")
    parser.add_argument("--suite-seed", type=int, default=None, help="fixed suite order (default: secret)")
    parser.add_argument("--base-seed", type=int, default=None,
                        help="fixed data seeds, for tests and calibration only: a published release must keep "
                             "its seeds secret (default: secret)")
    parser.add_argument("--no-twin", action="store_true", help="skip the private fresh twin")
    parser.add_argument("--score", metavar="FILE", help="score predictions (JSON lines id/response)")
    parser.add_argument("--label", help="name for the scored predictions")
    parser.add_argument("--solve", metavar="MODULE:FUNCTION", help="ask a model every question and score it")
    parser.add_argument("--model-name", help="model name and version to record (with --solve)")
    parser.add_argument("--limit", type=int, default=0, help="with --solve: only the first N questions")
    args = parser.parse_args()
    if args.score:
        return score_file(args)
    if args.solve:
        return solve(args)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
