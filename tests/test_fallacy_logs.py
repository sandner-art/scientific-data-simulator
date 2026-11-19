"""Fallacy flags for logs recorded outside the simulator (agents' tool calls, labelling sessions)."""
import json

import numpy as np
import pandas as pd
import pytest

from scidatasim import cli
from scidatasim.diagnostics.logs import diagnose_log, read_key

X = np.linspace(0, 2, 9)


def _events():
    ev = [
        # e1: answers at once, wrong
        {"episode": "e1", "type": "answer", "value": 9.81},
        # e2: measures twice with the same inputs, then the right answer
        {"episode": "e2", "type": "experiment", "inputs": {"length": 1.0}, "y": [2.01]},
        {"episode": "e2", "type": "experiment", "inputs": {"length": 1.0}, "y": [1.99]},
        {"episode": "e2", "type": "answer", "value": 2.0},
        # e3: one measurement, then the textbook value although it is wrong here, budget left
        {"episode": "e3", "type": "measure", "inputs": {"h": 1}, "y": [0.45]},
        {"episode": "e3", "type": "submit", "value": 9.81},
        # e4: expression target, measures only near 0, answers the textbook law
        {"episode": "e4", "type": "experiment", "x": [0.0, 0.1, 0.2], "y": [0.0, 0.3, 0.6]},
        {"episode": "e4", "type": "answer", "value": "x**2"},
        # e5: never answers; one failed step
        {"episode": "e5", "type": "tool", "inputs": {"q": "x"}, "error": "timeout"},
        # e6: a label revised; the last one counts
        {"episode": "e6", "type": "view", "inputs": {"item": 6}},
        {"episode": "e6", "type": "label", "value": "cat"},
        {"episode": "e6", "type": "label", "value": "dog"},
    ]
    return ev


def _key():
    return {
        "e1": {"answer": 3.71, "tolerance": 0.05, "budget": 5},
        "e2": {"answer": 2.0, "tolerance": 0.05, "budget": 5},
        "e3": {"answer": 3.71, "tolerance": 0.05, "budget": 5, "alternatives": {"textbook": 9.81}},
        "e4": {"answer": "3*x", "answer_type": "expression", "tolerance": 0.05, "domain": [0, 2],
               "alternatives": {"textbook": "x**2"}},
        "e5": {"answer": 1.0},
        "e6": {"answer": "dog", "answer_type": "choice", "alternatives": {"popular": "cat"}},
    }


def test_flags_from_a_generic_log():
    result = diagnose_log(_events(), read_key(_key()))
    d = {e["episode"]: e for e in result["episodes"]}
    assert d["e1"]["flags"]["answer_without_evidence"] and d["e1"]["flags"]["premature_commitment"]
    assert d["e2"]["correct"] and d["e2"]["flags"]["repeated_experiment"] and d["e2"]["n_experiments"] == 2
    assert d["e3"]["flags"]["matches_textbook"] and d["e3"]["flags"]["premature_commitment"]
    assert d["e4"]["flags"]["narrow_sampling"] and d["e4"]["flags"]["contradicts_own_data"]
    assert d["e4"]["flags"]["matches_textbook"] and d["e4"]["domain_coverage"] < 0.5
    assert d["e5"]["flags"]["no_answer"] and d["e5"]["n_failed"] == 1
    assert d["e6"]["correct"] and d["e6"]["revisions"] == 1 and not d["e6"]["flags"]["matches_popular"]
    assert "premature_commitment" not in d["e5"]["flags"]  # no budget in the key: the flag does not apply
    assert "narrow_sampling" not in d["e2"]["flags"]  # only for expression targets
    s = result["summary"]
    assert s["episodes"] == 6 and s["accuracy"] == pytest.approx(2 / 6)
    assert s["flags"]["premature_commitment"]["applicable"] == 3


def test_files_and_cli(tmp_path, capsys):
    rows = [dict(e, inputs=json.dumps(e["inputs"])) if "inputs" in e else dict(e) for e in _events()]
    for r in rows:
        for k in ("x", "y"):
            if k in r:
                r[k] = json.dumps(r[k])
    pd.DataFrame(rows).to_csv(tmp_path / "events.csv", index=False)
    (tmp_path / "key.json").write_text(json.dumps(_key()))
    result = diagnose_log(str(tmp_path / "events.csv"), str(tmp_path / "key.json"))
    assert {e["episode"]: e["flags"].get("matches_textbook") for e in result["episodes"]}["e3"]
    with open(tmp_path / "events.jsonl", "w") as f:
        f.write("\n".join(json.dumps(e) for e in _events()))
    assert cli.main(["fallacies", str(tmp_path / "events.jsonl"), "--key", str(tmp_path / "key.json")]) == 0
    text = capsys.readouterr().out
    assert "| answer_without_evidence |" in text and "6 episodes" in text
    with pytest.raises(KeyError):
        diagnose_log([{"type": "answer"}], read_key(_key()))
