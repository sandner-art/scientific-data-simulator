# tests/test_ai_roles.py
"""The model roles with scripted stand-in models (no provider needed)."""
import json
import os
import re
import shutil

import pandas as pd
import pytest

from simulator.ai import (LLMAgent, LoggedClient, ReplayClient, ReplayMiss, check_rewrite, evidence_free_view,
                          extract_json, llm_attack, llm_provenance, paraphrase_tasks, roles_of, solve_tasks,
                          write_llm_provenance)
from simulator.ai.agent import UNPARSEABLE
from simulator.diagnostics import diagnose_episode, run_episodes
from simulator.doe import run_design
from simulator.envs import ExperimentEnv
from simulator.export.tasks import collect_bundle_tasks

LAW_BASE = {"experiment_type": "experiments.law_discovery.logic.HiddenLawExperiment", "n_points": 25,
            "generate_plots": False, "bundle": {"figure": {"show_latent": False}, "qa": {"value_queries": 0}},
            "observation": {"operators": [{"type": "gaussian", "snr_db": 35}]}}
SPEC = {"base_config": {"experiment_type": "experiments.law_discovery.logic.HiddenLawExperiment", "n_points": 30,
                        "counterfactual": True, "x_min": 0.5, "x_max": 5.0,
                        "observation": {"operators": [{"type": "gaussian", "snr_db": 40}]}},
        "hidden": {"law": {"choices": ["power_law"]}, "law_seed": [0, 1.0e12]},
        "controls": {"x_min": [0.1, 9.0], "x_max": [0.5, 10.0]},
        "target": "law_expression", "target_type": "expression", "domain": [0.5, 5.0], "budget": 3,
        "tolerance": 0.02}


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    out = tmp_path_factory.mktemp("ai")
    table = pd.DataFrame({"law": ["power_law", "hill", "logistic", "arrhenius"] * 3})
    run_design(LAW_BASE, table, output_dir=str(out / "runs"), base_seed=3)
    from simulator.export.huggingface import export_huggingface
    pytest.importorskip("pyarrow")
    release = export_huggingface(str(out / "runs"), str(out / "release"), name="t/ai")
    return str(out / "runs"), release


# ------------------------------------------------------------------ clients
def test_logged_client_replay_and_provenance(tmp_path):
    log = tmp_path / "calls.jsonl"
    counter = {"n": 0}

    def model(prompt):
        counter["n"] += 1
        return f"reply {counter['n']} to {len(prompt)}"
    client = LoggedClient(model, str(log), role="agent", model="test/model-1", settings={"temperature": 0})
    first = [client("a"), client("b"), client("a")]
    writer = client.with_role("writer")
    writer("c")
    entries = [json.loads(l) for l in log.read_text().splitlines()]
    assert len(entries) == 4 and entries[0]["role"] == "agent" and entries[3]["role"] == "writer"
    assert entries[0]["settings"] == {"temperature": 0} and "prompt" in entries[0]
    replay = ReplayClient(str(log), role="agent")
    assert [replay("a"), replay("b"), replay("a")] == first  # repeated prompts in recorded order
    with pytest.raises(ReplayMiss):
        replay("a")
    with pytest.raises(ReplayMiss):
        replay("never asked")
    prov = llm_provenance(str(log))
    assert {(c["role"], c["calls"]) for c in prov["llm_calls"]} == {("agent", 3), ("writer", 1)}
    assert len(prov["logs"][0]["sha256"]) == 64
    assert roles_of(str(log)) == {"test/model-1": {"agent", "writer"}}
    path = write_llm_provenance(str(tmp_path), [str(log)])
    assert "prompt" not in open(path).read()  # the summary carries no prompt text
    hashed = LoggedClient(model, str(tmp_path / "h.jsonl"), role="solver", model="m", store_text=False)
    hashed("x")
    with pytest.raises(ValueError):
        ReplayClient(str(tmp_path / "h.jsonl"))
    with pytest.raises(ValueError):
        LoggedClient(model, str(log), role="agent", model="")


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"type": "answer", "value": 3}\n```') == {"type": "answer", "value": 3}
    assert extract_json('I think {"a": [1, 2]} works') == {"a": [1, 2]}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        extract_json("no json here {")


# ------------------------------------------------------------------ agent
def _scripted_scientist(truth_box):
    """Measures once, then answers the true law, wrapped in prose (as chat models do)."""
    def model(prompt):
        if "EPISODE SO FAR" not in prompt:
            return 'Let me measure.\n```json\n{"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}}\n```'
        return f'The data fit well. {{"type": "answer", "value": "{truth_box["truth"]}"}}'
    return model


def _env_with_truth_box():
    env = ExperimentEnv(SPEC)
    box = {}
    reset = env.reset

    def boxed_reset(seed=None):
        obs = reset(seed=seed)
        box["truth"] = env.truth["law_expression"]
        return obs
    env.reset = boxed_reset
    return env, box


def test_llm_agent_plays_logs_and_replays(tmp_path):
    env, box = _env_with_truth_box()
    log = str(tmp_path / "agent.jsonl")
    client = LoggedClient(_scripted_scientist(box), log, role="agent", model="scripted/scientist")
    agent = LLMAgent(client)
    logs = run_episodes(env, agent, seeds=range(3))
    assert all(diagnose_episode(l)["correct"] for l in logs) and agent.stats["parse_failures"] == 0
    first_prompt = json.loads(open(log).readline())["prompt"]
    assert "TASK" in first_prompt and "Experiments left: 3" in first_prompt
    replayed = run_episodes(ExperimentEnv(SPEC), LLMAgent(ReplayClient(log)), seeds=range(3))
    assert [l.steps[-1]["action"] for l in replayed] == [l.steps[-1]["action"] for l in logs]


def test_llm_agent_retries_then_gives_up():
    replies = iter(['{"type": "dance"}', '{"type": "answer", "value": "2*x"}'])
    agent = LLMAgent(lambda p: next(replies))
    logs = run_episodes(ExperimentEnv(SPEC), agent, seeds=[0])
    assert agent.stats == {"calls": 2, "parse_failures": 1, "gave_up": 0}
    assert logs[0].steps[0]["action"] == {"type": "answer", "value": "2*x"}
    stubborn = LLMAgent(lambda p: "I refuse to use JSON.", max_retries=1)
    logs = run_episodes(ExperimentEnv(SPEC), stubborn, seeds=[0])
    assert logs[0].steps[-1]["action"]["value"] == UNPARSEABLE and stubborn.stats["gave_up"] == 1
    assert diagnose_episode(logs[0])["flags"]["invalid_answer"]


# ------------------------------------------------------------------ attacker
def test_evidence_free_view_drops_the_evidence(suite):
    runs, release = suite
    view = evidence_free_view(release)
    assert "=== README.md ===" in view and "=== data/questions.parquet ===" in view
    assert ".png ===" not in view and "y_observed" not in view.split("=== data/runs.parquet ===")[1][:200]


def test_llm_attack_blind_model_passes_and_leaky_release_fails(suite, tmp_path):
    runs, release = suite
    silent = llm_attack(lambda p: "{}", release, runs, batch_size=50)
    assert silent.verdict == "INCONCLUSIVE" and not silent.ok and silent.n_answered == 0

    def guesser(prompt):  # answers everything, as a list of records, after a worked example
        ids = re.findall(r'"id": "([^"]+)"', prompt)
        return 'Format: {"id": "answer"}. Here:\n' + json.dumps([{"id": i, "answer": "0"} for i in ids])
    guess = llm_attack(guesser, release, runs, batch_size=50)
    assert guess.verdict == "PASS" and guess.ok and guess.n_answered == guess.n_questions
    leaky = tmp_path / "leaky"
    shutil.copytree(release, leaky)
    tasks = collect_bundle_tasks(runs)
    with open(leaky / "README.md", "a") as f:  # an answer key left in the card
        f.write("\n" + "\n".join(f"{t['id']} => {t['answer']}" for t in tasks if t["template"] == "law_formula"))

    def reader(prompt):
        found = dict(re.findall(r"^(\S+/q\d+) => (.+)$", prompt, flags=re.M))
        ids = re.findall(r'"id": "([^"]+)"', prompt)
        return json.dumps({i: found.get(i, "0") for i in ids})
    attack = llm_attack(reader, str(leaky), runs, batch_size=50)
    assert attack.verdict == "FLAGGED" and attack.leaks == ["law_formula"]
    assert attack.templates["law_formula"]["accuracy"] == 1.0


# ------------------------------------------------------------------ writer
def test_rewrite_checks():
    task = {"answer_type": "choice", "answer": "hill", "choices": ["hill", "logistic"]}
    original = "Which family generated y(x)? Choose one of: hill, logistic."
    assert check_rewrite(original, "Choose one of hill, logistic: which family generated y(x)?", task) == []
    assert "numbers" in check_rewrite("Value at x = 2.5?", "Value at x = 2.5 or 3?", {"answer_type": "number"})
    assert "numbers" in check_rewrite("Value at x = 2.5?", "Value at this point?", {"answer_type": "number"})
    assert check_rewrite("Value at x = 2.50?", "What is the value at x = 2.5?", {"answer_type": "number"}) == []
    assert "choices" in check_rewrite(original, "Which family generated y(x)? Pick hill or the other one.", task)
    formula = {"answer_type": "expression", "answer": "2*x**1.5"}
    q = "Give a formula for y on [0.5, 5]."
    assert "answer" in check_rewrite(q, "Give a formula (a power law) for y on [0.5, 5].", formula, ["power_law"])
    assert "answer" in check_rewrite(q, "Give a formula like 2 * x**1.5 for y on [0.5, 5].", formula)
    assert "length" in check_rewrite(q, "Formula?", formula)


def test_paraphrase_tasks_accepts_safe_and_rejects_leaky(suite):
    runs, _ = suite
    tasks = collect_bundle_tasks(runs)

    def careful(prompt):
        question = prompt.split("QUESTION\n", 1)[1].split("\n\n")[0]
        return "Please answer this: " + question

    def leaky(prompt):
        question = prompt.split("QUESTION\n", 1)[1].split("\n\n")[0]
        return question + " Hint: it is a power law with exponent 1.5."
    new, report = paraphrase_tasks(tasks, careful, writer="scripted/careful")
    assert report.accepted == len(tasks) and not report.rejected
    assert all(t["question"].startswith("Please answer this: ") and t["question"] in t["prompt"] for t in new)
    assert [t["answer"] for t in new] == [t["answer"] for t in tasks]
    assert new[0]["rewrite"]["writer"] == "scripted/careful"
    kept, bad = paraphrase_tasks(tasks, leaky, writer="scripted/leaky")
    assert bad.accepted == 0 and bad.rejected["numbers"] == len(tasks)
    assert bad.rejected.get("answer", 0) >= 1 and bad.examples
    assert [t["question"] for t in kept] == [t["question"] for t in tasks]


def test_solver_scores_free_text(suite):
    runs, _ = suite
    tasks = [t for t in collect_bundle_tasks(runs) if t["template"] in ("law_family", "law_formula")]
    answers = iter([t["answer"] for t in tasks])  # prompts repeat across runs: answer in order
    result = solve_tasks(tasks, lambda p: f"My answer: {next(answers)}")
    assert all(r["score"] == 1.0 for r in result.values())
    wrong = solve_tasks(tasks, lambda p: "I don't know")
    assert all(r["score"] == 0.0 for r in wrong.values())


def test_example_12_runs_offline(tmp_path, monkeypatch):
    import sys
    from examples.example_12_model_roles import run_experiment as ex12
    monkeypatch.setattr(sys, "argv", ["ex12", "--output_dir", str(tmp_path / "out"), "--episodes", "2"])
    ex12.main()
    report = (tmp_path / "out" / "report.md").read_text()
    assert "Leak audit: PASS" in report and "2/2 identical" in report and "LLM attack: PASS" in report
    prov = json.loads((tmp_path / "out" / "llm_provenance.json").read_text())
    assert {c["role"] for c in prov["llm_calls"]} == {"attacker", "writer", "agent"}
    monkeypatch.setattr(sys, "argv", ["ex12", "--model", "json:dumps"])
    with pytest.raises(SystemExit):  # a real model needs its name recorded
        ex12.main()


# ------------------------------------------------------------------ review regressions
def test_replay_order_without_role_and_torn_last_line(tmp_path):
    log = tmp_path / "l.jsonl"
    w = LoggedClient(lambda p: "from writer", str(log), role="writer", model="m")
    a = w.with_role("agent")
    a._complete = lambda p: "from agent"
    w("same")
    a("same")
    with open(log, "a") as f:
        f.write('{"role": "agent", "prompt_sha')  # interrupted write
    replay = ReplayClient(str(log))
    assert replay.skipped_last_line and [replay("same"), replay("same")] == ["from writer", "from agent"]


def test_agent_budget_after_failed_experiment_and_last_valid_action():
    prompts = []
    replies = iter(['{"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}}',
                    '{"type": "experiment", "inputs": {"x_min": 5.0, "x_max": 1.0}}',
                    '{"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 2.0}}',
                    'Draft: {"type": "answer", "value": "x"} ... final: {"type": "answer", "value": "2*x"}'])

    def model(prompt):
        prompts.append(prompt)
        return next(replies)
    logs = run_episodes(ExperimentEnv(SPEC), LLMAgent(model), seeds=[0])
    assert "Experiments left: 3." in prompts[0] and "Experiments left: 2." in prompts[1]
    assert "Experiments left: 1." in prompts[2]  # the failed experiment cost budget
    assert "you must answer now" in prompts[3]
    assert logs[0].steps[-1]["action"] == {"type": "answer", "value": "2*x"}


def test_evidence_free_view_of_bundle_folder_and_public_tables(tmp_path):
    from simulator.engine import SimulatorEngine
    run = SimulatorEngine(str(tmp_path / "runs"), verbose=False).run_config(
        dict(LAW_BASE, law="power_law", seed=1, bundle={"public_table": True, "qa": {"value_queries": 0}}))
    view = evidence_free_view(run.experiment_dir)
    for private in ("qa.json", "description.md", "bundle.json", "ground_truth.json", "table.csv",
                    "plot_figure.py", "experiment_record.json"):
        assert f"{private} ===" not in view
    assert "questions.json ===" in view and "caption.txt ===" in view
    pytest.importorskip("pyarrow")
    from simulator.export.huggingface import export_huggingface
    release = export_huggingface(str(tmp_path / "runs"), str(tmp_path / "rel"), name="t/tables")
    assert os.path.isdir(os.path.join(release, "tables"))
    assert "=== tables" not in evidence_free_view(release)


def test_truncated_view_is_inconclusive(suite):
    runs, release = suite

    def guesser(prompt):
        return json.dumps({i: "0" for i in re.findall(r'"id": "([^"]+)"', prompt)})
    report = llm_attack(guesser, release, runs, batch_size=50, max_file_chars=200)
    assert report.verdict == "INCONCLUSIVE" and report.truncated and "truncated" in report.summary()


def test_rewrite_checks_catch_more_and_spare_benign():
    task = {"answer_type": "choice", "answer": "hill", "choices": ["hill", "logistic"]}
    q = "Which family generated y(x)? Choose one of: hill, logistic."
    assert "choices" in check_rewrite(q, "Which family generated y(x)? Choose one of: logistic, hill.", task)
    assert "choices" in check_rewrite(q, "Which family generated y(x)? Choose one of: **hill**, logistic.", task)
    assert "choices" in check_rewrite(q, "Which family made y(x), uphill or logistic?", task)
    assert "negation" in check_rewrite(q, "Which family did not generate y(x)? Choose one of: hill, logistic.", task)
    num = {"answer_type": "number", "answer": 3.0}
    assert "numbers" in check_rewrite("Value of y at x = 8?", "Value of y at x = 8, roughly eight?", num)
    assert "numbers" in check_rewrite("y at x = 2 and x = 8?", "y at x = 8 and x = 2?", num)
    assert "length" in check_rewrite("Value of y at x = 8?", "Value of y at x = 8?\n(Note: easy)", num)
    formula = {"answer_type": "expression", "answer": "2*x"}
    assert "answer" in check_rewrite("Give a formula for y.", "Give a powerlaw formula for y.", formula, ["power_law"])
    assert "answer" in check_rewrite("Give a formula for y.", "Give a formula (power laws?) for y.", formula,
                                     ["power_law"])
    for original, rewrite in [("x from 0.5-8, find y.", "For x in 0.5-8, find y."),
                              ("At 10,000 steps?", "After 10000 steps?"),
                              ("At x = −2?", "When x = -2?")]:
        assert check_rewrite(original, rewrite, num) == [], (original, rewrite)


def test_rewrite_rejected_when_prompt_cannot_follow():
    task = {"id": "r/q1", "question": "What is y at x = 1?", "prompt": "Chart.\n\nQuestion: What is  y at x = 1?",
            "answer_type": "number", "answer": 2.0}
    new, report = paraphrase_tasks([task], lambda p: "At x = 1, what is y?", writer="w")
    assert report.rejected == {"prompt": 1} and new[0]["question"] == task["question"]
