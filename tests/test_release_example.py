# tests/test_release_example.py - example 13 builds a release that is audited, checksummed and scorable
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = "examples.example_13_law_release.run_experiment"
pytest.importorskip("pyarrow")


def _run(*args, env=None):
    result = subprocess.run([sys.executable, "-W", "ignore", "-m", ENTRY, *args], cwd=ROOT, capture_output=True,
                            text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    return result.stdout


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    out = tmp_path_factory.mktemp("release")
    _run("--quick", "--output_dir", str(out))
    return out


def test_release_is_audited_and_checksummed(release):
    public, private = release / "public", release / "private"
    record = json.loads((private / "release.json").read_text())
    assert record["audit_ok"] and record["questions"] == 3 * record["runs"]
    assert {"suite_seed", "base_seed", "canary"} <= set(record)
    sums = (public / "SHA256SUMS").read_text().splitlines()
    import hashlib
    for line in sums:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((public / rel).read_bytes()).hexdigest() == digest
    listed = {line.split("  ", 1)[1] for line in sums}
    on_disk = {os.path.relpath(os.path.join(r, f), public).replace(os.sep, "/")
               for r, _, fs in os.walk(public) for f in fs} - {"SHA256SUMS"}
    assert listed == on_disk


def test_public_release_holds_no_answers_or_seeds(release):
    import pyarrow.parquet as pq
    public = release / "public"
    questions = pq.read_table(public / "data" / "questions.parquet")
    assert not {"answer", "tolerance", "polynomial_reachable", "seed"} & set(questions.column_names)
    assert set(questions.column("template").to_pylist()) == {"law_family", "law_formula", "extrapolation"}
    record = json.loads((release / "private" / "release.json").read_text())
    text = "".join(p.read_text(errors="ignore") for p in public.rglob("*") if p.suffix in (".md", ".json", ".txt"))
    assert str(record["suite_seed"]) not in text and str(record["base_seed"]) not in text
    card = (public / "README.md").read_text()
    assert "## Baselines" in card and record["canary"] in card
    assert card.rstrip().endswith("-->")  # the canary comes last


def test_release_scores_predictions(release, tmp_path):
    key = [json.loads(line) for line in (release / "private" / "answer_key.jsonl").read_text().splitlines()]
    preds = tmp_path / "preds.jsonl"
    preds.write_text("\n".join(json.dumps({"id": t["id"], "response": str(t["answer"])}) for t in key))
    out = _run("--output_dir", str(release), "--score", str(preds), "--label", "oracle")
    assert "overall 1.00" in out
    scored = json.loads((release / "private" / "scores" / "oracle.json").read_text())
    assert scored["accuracy"] == 1.0 and scored["answered"] == len(key)


def test_release_solves_with_a_logged_model(release, tmp_path):
    (tmp_path / "standin.py").write_text("def ask(prompt):\n    return 'power_law'\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    out = _run("--output_dir", str(release), "--solve", "standin:ask", "--model-name", "test/standin", "--limit", "6",
               env=env)
    assert "answered 6 of 6" in out
    log = (release / "private" / "llm" / "test_standin.jsonl").read_text().splitlines()
    assert len(log) == 6 and all(json.loads(line)["role"] == "solver" for line in log)


def test_solve_requires_a_model_name(release):
    result = subprocess.run([sys.executable, "-m", ENTRY, "--output_dir", str(release), "--solve", "x:y"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "--model-name" in (result.stderr + result.stdout)
