# tests/test_audit.py
import json
import os
import shutil

import pandas as pd
import pytest

from simulator.audit import AuditReport, _nontrivial, audit_environment, audit_export, main
from simulator.doe import run_design
from simulator.engine import SimulatorEngine
from simulator.envs import ExperimentEnv
from simulator.export.tasks import collect_bundle_tasks, public_view, write_tasks_jsonl

BASE = {"experiment_type": "experiments.oscillator.logic.OscillatorExperiment", "n_steps": 150, "stiffness": 4.0,
        "damping": 0.2, "generate_plots": False, "bundle": True,
        "observation": {"operators": [{"type": "gaussian", "snr_db": 30}]}}


@pytest.fixture(scope="module")
def design(tmp_path_factory):
    out = tmp_path_factory.mktemp("design")
    run = run_design(BASE, pd.DataFrame({"stiffness": [2.0, 6.0]}), output_dir=str(out), base_seed=11)
    return run.output_dir


def _attacks(report: AuditReport):
    return {f.attack for f in report.findings if f.severity == "high"}


def test_public_task_file_passes(design, tmp_path):
    path = tmp_path / "tasks.jsonl"
    write_tasks_jsonl(collect_bundle_tasks(design), str(path))
    report = audit_export(str(path), design)
    assert report.ok, report.summary()
    assert report.exposure["answer_lookup"]["total"] > 0 and report.exposure["seed_replay"]["total"] == 2


def test_each_leak_is_detected(design, tmp_path):
    tasks = collect_bundle_tasks(design)
    leaky = tmp_path / "leaky.jsonl"
    with open(leaky, "w") as f:
        for t in tasks:
            f.write(json.dumps({k: v for k, v in t.items() if k != "answer"}) + "\n")  # keeps seed and tolerance
    report = audit_export(str(leaky), design)
    assert {"seed_replay", "tolerance_inversion"} <= _attacks(report)
    inv = report.exposure["tolerance_inversion"]
    assert inv["exposed"] >= 2  # stiffness and damping: "5 % of the true value"
    assert any(f.attack == "private_field" and "tolerance" in f.detail for f in report.findings)

    answers = tmp_path / "answers.jsonl"
    write_tasks_jsonl(tasks, str(answers), include_answers=True)
    report = audit_export(str(answers), design)
    assert {"private_field", "answer_lookup", "seed_replay"} <= _attacks(report)
    assert report.exposure["answer_lookup"]["exposed"] == report.exposure["answer_lookup"]["total"]
    assert not report.ok and "FAIL" in report.summary()


def test_noise_free_table_is_an_answer_lookup(tmp_path):
    cfg = dict(BASE, bundle={"public_table": True}, observation=None, seed=5)
    cfg.pop("observation")
    run = SimulatorEngine(str(tmp_path / "runs"), verbose=False).run_config(cfg)
    public = tmp_path / "public"
    os.makedirs(public)
    shutil.copy(os.path.join(run.experiment_dir, "bundle", "table.csv"), public / "table.csv")
    report = audit_export(str(public), run.experiment_dir)
    assert "answer_lookup" in _attacks(report)
    assert "latent_exposure" in _attacks(report)  # the exact position/velocity series


def test_config_hash_and_ground_truth_files(design, tmp_path):
    record_dir = next(d for d in (os.path.join(design, n) for n in sorted(os.listdir(design)))
                      if os.path.exists(os.path.join(d, "experiment_record.json")))
    public = tmp_path / "pub"
    os.makedirs(public)
    shutil.copy(os.path.join(record_dir, "experiment_record.json"), public / "record.json")
    report = audit_export(str(public), design)
    found = {f.attack for f in report.findings}
    assert {"config_hash", "seed_replay"} <= found
    assert report.exposure["config_hash"] == {"exposed": 1, "total": 2}
    assert any(f.attack == "private_field" and "ground_truth" in f.detail for f in report.findings)


def test_huggingface_export_default_vs_open(design, tmp_path):
    pytest.importorskip("pyarrow")
    from simulator.export.huggingface import export_huggingface
    closed = export_huggingface(design, str(tmp_path / "closed"), name="t/closed")
    assert audit_export(closed, design).ok
    opened = export_huggingface(design, str(tmp_path / "open"), name="t/open", include_answers=True,
                                include_seeds=True, include_ground_truth=True)
    attacks = _attacks(audit_export(opened, design))
    assert {"private_field", "seed_replay", "answer_lookup", "latent_exposure"} <= attacks


def test_environment_audit():
    spec = {"base_config": {"experiment_type": "experiments.oscillator.logic.OscillatorExperiment", "n_steps": 100, "dt": 0.05, "x0": 1.0,
                            "observation": {"operators": [{"type": "gaussian", "sigma": 0.01}]}},
            "hidden": {"stiffness": [1.0, 9.0]}, "controls": {"x0": [0.5, 2.0]},
            "target": "stiffness", "budget": 2}
    env = ExperimentEnv(spec)
    report = audit_environment(env, seeds=(1, 2), probe={"type": "experiment", "inputs": {"x0": 1.0}})
    assert report.ok, report.summary()

    class LeakyEnv(ExperimentEnv):
        @property
        def state(self):
            return {**ExperimentEnv.state.fget(self), "episode_seed": self._episode_seed, "truth": self.truth}

    leaky = audit_environment(LeakyEnv(spec), seeds=(1, 2), probe={"type": "experiment", "inputs": {"x0": 1.0}})
    assert {"seed_replay", "answer_lookup"} <= _attacks(leaky)


def test_cli_and_helpers(design, tmp_path, capsys):
    path = tmp_path / "t.jsonl"
    write_tasks_jsonl([public_view(t) for t in collect_bundle_tasks(design)], str(path))
    assert main([str(path), "--private", design]) == 0
    assert "PASS" in capsys.readouterr().out
    write_tasks_jsonl(collect_bundle_tasks(design), str(path), include_answers=True)
    assert main([str(path), "--private", design, "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    assert not _nontrivial(3.0) and not _nontrivial(0.25) and _nontrivial(2.3456) and _nontrivial(123456.0)


# ------------------------------------------------------------------ review regressions
def test_unreadable_or_empty_public_files_fail(design, tmp_path):
    tasks = collect_bundle_tasks(design)
    bad = tmp_path / "bad"
    os.makedirs(bad)
    with open(bad / "tasks.jsonl", "w") as f:
        f.write("{not json\n" + json.dumps({"id": "x", "answer": 1.234567, "seed": tasks[0]["seed"]}) + "\n")
    report = audit_export(str(bad), design)
    assert {"unreadable", "private_field", "seed_replay"} <= _attacks(report)
    (bad / "tasks.jsonl").unlink()
    (bad / "runs.parquet").write_bytes(b"PAR1 not really")
    assert "unreadable" in _attacks(audit_export(str(bad), design))
    (bad / "runs.parquet").unlink()
    (bad / "deep.json").write_text("[" * 5000 + "]" * 5000)
    assert "unreadable" in _attacks(audit_export(str(bad), design))
    (bad / "deep.json").unlink()
    (bad / "figure.png").write_bytes(b"\x89PNG")
    assert "nothing_scanned" in _attacks(audit_export(str(bad), design))


def test_rounded_spaced_and_float_leaks(design, tmp_path):
    tasks = [t for t in collect_bundle_tasks(design) if t["answer_type"] == "number" and _nontrivial(t["answer"])]
    pub = tmp_path / "pub"
    os.makedirs(pub)
    (pub / "notes.md").write_text(f"The maximum is about {tasks[0]['answer']:.4g}.")
    report = audit_export(str(pub), design)
    assert report.exposure["answer_lookup"]["exposed"] >= 1 and "answer_lookup" in _attacks(report)
    (pub / "notes.md").unlink()
    seeds = [t["seed"] for t in collect_bundle_tasks(design)]
    (pub / "runs.csv").write_text("run,seed\n" + "".join(f"{i},{float(s):.17g}\n" for i, s in enumerate(seeds)))
    assert "seed_replay" in _attacks(audit_export(str(pub), design))


def test_formula_with_other_spacing_and_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    run = SimulatorEngine(str(tmp_path / "runs"), verbose=False).run_config(
        {"experiment_type": "experiments.law_discovery.logic.HiddenLawExperiment", "law": "power_law",
         "n_points": 30, "seed": 2, "bundle": True, "generate_plots": False})
    formula = run.results["law_expression"]["data"]
    pub = tmp_path / "pub"
    os.makedirs(pub)
    (pub / "readme.md").write_text("Law: " + formula.replace("*", " * "))
    assert "answer_lookup" in _attacks(audit_export(str(pub), run.experiment_dir))
    (pub / "readme.md").unlink()
    with h5py.File(pub / "fields.hdf5", "w") as f:
        f["y"] = run.results["y"]["data"]
    assert "latent_exposure" in _attacks(audit_export(str(pub), run.experiment_dir))
    (pub / "fields.hdf5").unlink()
    np.savez(pub / "arrays.npz", y=run.results["y"]["data"])
    assert "latent_exposure" in _attacks(audit_export(str(pub), run.experiment_dir))


def test_coordinates_quoted_in_questions_are_not_a_leak(tmp_path):
    run = SimulatorEngine(str(tmp_path / "runs"), verbose=False).run_config(
        {"experiment_type": "experiments.geometry.simplex.SimplexExperiment", "dims": 3, "decimals": None,
         "seed": 4, "bundle": True, "generate_plots": False})
    pub = tmp_path / "pub"
    os.makedirs(pub)
    shutil.copy(os.path.join(run.experiment_dir, "bundle", "questions.json"), pub / "questions.json")
    report = audit_export(str(pub), run.experiment_dir)
    assert "latent_exposure" not in _attacks(report), report.summary()


def test_design_point_and_huggingface_hiding(tmp_path):
    pytest.importorskip("pyarrow")
    from experiments.law_discovery.logic import law_suite_design
    from simulator.export.huggingface import export_huggingface
    base = {"experiment_type": "experiments.law_discovery.logic.HiddenLawExperiment", "n_points": 20,
            "generate_plots": False, "bundle": {"qa": {"value_queries": 0}, "figure": {"show_latent": False}},
            "observation": {"operators": [{"type": "gaussian", "snr_db": 30}]}}
    design = run_design(base, law_suite_design(families=["power_law", "hill"], snr_db=(30,), seed=3),
                        output_dir=str(tmp_path / "d"), base_seed=5)
    out = export_huggingface(design.output_dir, str(tmp_path / "hf"), name="t/laws")
    runs = pd.read_parquet(os.path.join(out, "data", "runs.parquet"))
    assert "design_point" not in runs.columns
    assert list(runs["experiment_id"]) == sorted(runs["experiment_id"])
    report = audit_export(out, design.output_dir)
    assert report.ok, report.summary()
    leaky = tmp_path / "leaky.json"
    leaky.write_text(json.dumps([{"experiment_id": e, "design_point": i} for i, e in enumerate(runs["experiment_id"])]))
    assert any(f.attack == "design_point" for f in audit_export(str(leaky), design.output_dir).findings)
