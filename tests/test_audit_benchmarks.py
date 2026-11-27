# tests/test_audit_benchmarks.py - the audits of outside benchmarks (SciLaws-Bench, InfiniteScienceGym)
import importlib.util
import json
import os
import re
import pickle
import sys
import zlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------- a SciLaws-like release
LAW_I = '''import sys
raise SystemExit("formula.py must never be run by the audit")


def law(X):
    return 2.3457 + 3.1416 * X[:, 0] + 0.0  # the hidden law
'''

LAW_II = '''def law(X, k):
    return 7.2913 * (1.0 - 0.61803 * X[:, 0]) * k
'''


def _write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(int(v)) if c == "group_id" else repr(float(v)) for c, v in zip(header, r)) + "\n")


def _state(path, src, evil=False):
    blob = pickle.dumps({"formula_source": src, "noise": np.arange(3.0)}, protocol=4)
    if evil:  # what a load would run; the audit must list it, not run it
        blob = b"cos\nsystem\n(S'echo never'\ntR."
    with open(path, "wb") as f:
        f.write(zlib.compress(blob))


def make_release(root, rng):
    """Two tasks in the release's layout: Type I (one law) and Type II (per-group k)."""
    import yaml
    tasks = os.path.join(root, "tasks")
    # Type I: y = 2.3457 + 3.1416 x + noise; the test inputs extend beyond the training range
    t1 = os.path.join(tasks, "typeI", "line_task__y")
    for sub in ("data", "eval", "simulator"):
        os.makedirs(os.path.join(t1, sub))
    x_tr, x_te = rng.uniform(0, 1, 300), rng.uniform(0.5, 1.4, 120)
    y_tr = 2.3457 + 3.1416 * x_tr + rng.normal(0, 0.1, x_tr.size)
    y_te = 2.3457 + 3.1416 * x_te + rng.normal(0, 0.1, x_te.size)
    _write_csv(os.path.join(t1, "data", "train.csv"), ["y", "x"], zip(y_tr, x_tr))
    _write_csv(os.path.join(t1, "data", "test.csv"), ["y", "x"], zip(y_te, x_te))
    with open(os.path.join(t1, "metadata.yaml"), "w") as f:
        yaml.safe_dump({"task_id": "line_task__y", "type": "typeI", "metric": "rmse",
                        "target": {"name": "y"}, "inputs": [{"name": "x"}]}, f)
    ref_pred = 2.3457 + 3.1416 * x_te
    rmse = float(np.sqrt(np.mean((ref_pred - y_te) ** 2)))
    with open(os.path.join(t1, "eval", "reference_metrics.json"), "w") as f:
        json.dump({"type": "typeI", "derived_caps": {"max_law_constants": 2, "max_local_params": 0,
                                                    "max_init_size_per_param": 1, "fit_timeout_seconds": None},
                   "baselines": {"paper_a": {"kind": "reference", "law_constants": {"a": 2.3457, "b": 3.1416},
                                             "metrics": {"rmse": rmse}}}}, f)
    with open(os.path.join(t1, "simulator", "formula.py"), "w") as f:
        f.write(LAW_I)
    _state(os.path.join(t1, "simulator", "state.joblib"), LAW_I)
    # Type II: y = 7.2913 (1 - 0.61803 x) k_g; new groups at test time, each cut into fit/test by x
    t2 = os.path.join(tasks, "typeII", "decay_task__y")
    for sub in ("data", "eval", "simulator"):
        os.makedirs(os.path.join(t2, sub))

    def group_rows(g, n, lo, hi):
        x = np.sort(rng.uniform(lo, hi, n))
        k = 0.5 + g / 10.0
        y = 7.2913 * (1.0 - 0.61803 * x) * k * (1 + rng.normal(0, 0.02, n))
        return [(yy, g, xx) for yy, xx in zip(y, x)]
    train = [r for g in range(12) for r in group_rows(g, 30, 0.0, 1.0)]
    fit, test = [], []
    per_cluster = {}
    for g in range(20, 26):
        rows = group_rows(g, 20, 0.0, 1.0)
        fit += rows[:12]
        test += rows[12:]
        ty = np.array([r[0] for r in rows[12:]])
        tx = np.array([r[2] for r in rows[12:]])
        fy = np.array([r[0] for r in rows[:12]])
        fx = np.array([r[2] for r in rows[:12]])
        shape = 7.2913 * (1.0 - 0.61803 * fx)
        k = float(shape @ fy / (shape @ shape))
        pred = 7.2913 * (1.0 - 0.61803 * tx) * k
        per_cluster[str(g)] = {"rmse": float(np.sqrt(np.mean((pred - ty) ** 2)))}
    _write_csv(os.path.join(t2, "data", "train.csv"), ["y", "group_id", "x"], train)
    _write_csv(os.path.join(t2, "data", "test_fit.csv"), ["y", "group_id", "x"], fit)
    _write_csv(os.path.join(t2, "data", "test_test.csv"), ["y", "group_id", "x"], test)
    with open(os.path.join(t2, "metadata.yaml"), "w") as f:
        yaml.safe_dump({"task_id": "decay_task__y", "type": "typeII", "metric": "rmse",
                        "target": {"name": "y"}}, f)
    with open(os.path.join(t2, "eval", "reference_metrics.json"), "w") as f:
        json.dump({"type": "typeII", "derived_caps": {"max_law_constants": 1, "max_local_params": 1,
                                                     "max_init_size_per_param": 1, "fit_timeout_seconds": 10},
                   "baselines": {"paper_b": {"kind": "reference", "law_constants": {"c": 0.61803},
                                             "per_cluster": per_cluster}}}, f)
    with open(os.path.join(t2, "simulator", "formula.py"), "w") as f:
        f.write(LAW_II)
    _state(os.path.join(t2, "simulator", "state.joblib"), LAW_II, evil=True)
    return tasks


# ---------------------------------------------------------------------------- SciLaws tests
def test_metrics_and_unit_score_follow_the_harness():
    sl = _tool("audit_scilaws")
    yt, yp = np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.0, 3.0])
    assert sl.metric("rmse", yt, yp) == pytest.approx(np.sqrt((0.25 + 1.0) / 3))
    assert sl.metric("log_mae", yt, np.array([1.0, 2.0, -1.0])) > 100  # non-positive prediction: huge log error
    assert sl.metric("mape", np.zeros(3), yp) is None
    assert sl.unit_score(1.0, 1.0, "rmse") == 0.5 and sl.unit_score(0.0, 1.0, "rmse") == 1.0
    assert sl.unit_score(3.0, 1.0, "rmse") == 0.0
    assert sl.unit_score(0.9, 0.8, "r2") == pytest.approx(0.75)
    ref = {"baselines": {"a": {"kind": "reference", "metrics": {"rmse": 2.0}},
                         "b": {"kind": "reference", "metrics": {"rmse": 1.0}},
                         "naive": {"kind": "naive", "metrics": {"rmse": 0.1}}}}
    assert sl.best_reference(ref, "rmse") == "b"


def test_parse_law_and_state_are_read_without_running_them(tmp_path):
    sl = _tool("audit_scilaws")
    p = tmp_path / "formula.py"
    p.write_text(LAW_I)
    law = sl.parse_law(str(p))  # would raise SystemExit if executed
    assert any("3.1416" in e for e in law["expressions"])
    assert 3.1416 in law["constants"] and 2.3457 in law["constants"]
    good = tmp_path / "state.joblib"
    _state(str(good), LAW_I)
    info = sl.inspect_joblib(str(good), law)
    assert info["compression"] == "zlib" and info["formula_source_key"] and info["formula_py_in_state"]
    assert any(i.startswith("numpy") for i in info["imports"])
    evil = tmp_path / "evil.joblib"
    _state(str(evil), LAW_I, evil=True)
    assert "os.system" in sl.inspect_joblib(str(evil))["imports"]  # listed, not run


def test_audit_scilaws_end_to_end_without_harness(tmp_path):
    sl = _tool("audit_scilaws")
    tasks = make_release(str(tmp_path / "release"), np.random.default_rng(3))
    out = tmp_path / "out"
    assert sl.main(["--tasks", tasks, "--out", str(out), "--no-score", "--jobs", "1"]) == 0
    report = json.loads((out / "scilaws_audit.json").read_text())
    by = {t["task"]: t for t in report["tasks"]}
    line, decay = by["line_task__y"], by["decay_task__y"]
    # the law is in the grader-facing files (positive control), not in the solver-facing ones
    assert line["audit"]["grader_facing"]["exposure"]["answer_lookup"]["exposed"] > 0
    assert line["audit"]["solver_facing"]["ok"]
    assert decay["audit"]["state_joblib"]["imports"] == ["os.system"]
    assert line["split"]["class"] in ("mixed", "extrapolation") and line["split"]["direction"][1] == "above"
    assert line["fits"]["cv_scheme"].startswith("edge of x")
    # a straight line is the law here: the best non-law matches the reference (S_N about 0.5)
    tiers = line["fits"]["tiers"]
    assert tiers["prompt"]["n_global"] <= 2 and tiers["prompt"]["s_n"] > 0.4
    assert tiers["scorer"]["s_n"] >= tiers["prompt"]["s_n"] - 0.1
    # Type II: one per-group parameter, refitted on each new group's fit rows
    assert decay["split"]["class"].startswith("new groups") and decay["split"]["order"][1] == "above"
    t = decay["fits"]["tiers"]["scorer"]
    assert t["n_local"] == 1 and t["s_n"] is not None and t["max_fit_seconds"] < 10
    sub = sl._load_module(str(out / t["file"]))
    assert list(sub.LOCAL_FITTABLE) == ["c0"] and sub.LAW_CONSTANTS == {}
    md = (out / "scilaws_audit.md").read_text()
    assert "Leak audit" in md and "Non-law submissions" in md and "no marker" in md


def test_leak_hits_are_located_with_a_chance_estimate(tmp_path):
    sl = _tool("audit_scilaws")
    key = tmp_path / "key.jsonl"
    key.write_text("\n".join(json.dumps(r) for r in (
        {"id": "e", "template": "law_expression", "answer": "M_mol / rho_liq", "answer_type": "text"},
        {"id": "c", "template": "law_constant", "answer": 8.6912, "answer_type": "number"},
        {"id": "r", "template": "reference_constant", "answer": 2.3457, "answer_type": "number"},
        {"id": "q", "template": "law_constant", "answer": 0.25, "answer_type": "number"})) + "\n")
    task = tmp_path / "task"
    (task / "data").mkdir(parents=True)
    (task / "metadata.yaml").write_text("inputs:\n  - name: V\n    description: molar volume, M_mol/rho_liq\n")
    rng = np.random.default_rng(0)
    values = np.round(rng.uniform(8.6, 8.8, 20000), 4)  # dense 4-decimal data: 8.6912 is there by chance
    (task / "data" / "train.csv").write_text("y,x\n" + "\n".join(f"{v:.4f},1.0" for v in values)
                                             + "\n8.6912,2.34570000\n2.50000000e-01,1\n")
    hits = {h["id"]: h for h in sl.locate_hits(str(key), [str(task / "metadata.yaml"), str(task / "data")],
                                                base=str(task))}
    assert hits["e"]["file"] == "metadata.yaml" and hits["e"]["line"] == 3
    assert hits["c"]["chance_expected"] > 5 and not hits["c"]["counted"]
    assert hits["c"]["file"] == os.path.join("data", "train.csv")
    assert hits["r"]["count"] == 1 and hits["r"]["counted"]  # alone in its range, padded with zeros
    assert "q" not in hits  # 0.25 is trivial, however many zeros pad it


def test_audit_counts_digits_without_padding_and_discounts_chance_matches(tmp_path):
    import warnings
    from scidatasim.audit import _copied, _lookup_chance, _nontrivial, _printed_digits, audit_export
    assert [_printed_digits(t) for t in ("2.50000000e-01", "0.25", "1.482000", "8000", "13.7094", "1013.0")] \
        == [2, 2, 4, 1, 6, 4]
    assert not _nontrivial(1e16) and not _nontrivial(8000.0) and _nontrivial(123456.0) and _nontrivial(13.7094)
    assert _lookup_chance(np.array([0.0005, 3.0, 2e16]), 2e16, 1) == 0.0  # small numbers are not on a 1e16 grid
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _copied(np.array([1.0, 2.0]), 1e300)  # beyond float32: no overflow warning
    pub = tmp_path / "public"
    pub.mkdir()
    pressures = np.round(np.random.default_rng(1).normal(1013.25, 3.0, 5000), 1)
    (pub / "obs.csv").write_text("p\n" + "\n".join(f"{v:.1f}" for v in pressures) + "\n1013.3\n")
    key = tmp_path / "key.jsonl"
    key.write_text(json.dumps({"id": "a", "answer": 1013.3, "answer_type": "number"}) + "\n"
                   + json.dumps({"id": "b", "answer": 7.31942, "answer_type": "number"}) + "\n")
    report = audit_export(str(pub), str(key), check_arrays=False)
    assert report.exposure["answer_lookup"] == {"exposed": 0, "total": 2, "chance": 1}
    assert report.ok and report.by_attack("answer_lookup_chance")[0].severity == "info"
    (pub / "notes.md").write_text("the constant is 7.31942\n")
    report = audit_export(str(pub), str(key), check_arrays=False)
    assert not report.ok and report.exposure["answer_lookup"]["exposed"] == 1


def test_overlap_is_counted_on_scored_rows_against_the_data_s_own_repeats():
    import pandas as pd
    sl = _tool("audit_scilaws")
    train = pd.DataFrame({"y": [1.0, 1.0, 2.0, 3.0], "group_id": [1, 2, 2, 3], "x": [0.0, 0.0, 1.0, 2.0]})
    fit = pd.DataFrame({"y": [3.0], "group_id": [9], "x": [2.0]})  # calibration rows: not counted
    test = pd.DataFrame({"y": [1.0, 5.0], "group_id": [9, 9], "x": [0.0, 3.0]})
    task = {"type": "typeII", "inputs": ["x"], "target": "y", "train": train, "fit": fit, "test": test}
    prof = sl.split_profile(task)
    assert prof["test_rows_in_train"] == 1 and prof["n_test_rows"] == 2
    assert prof["train_rows_repeated_across_groups"] == pytest.approx(0.5)  # (0, 1) in groups 1 and 2


def test_a_crashing_harness_is_reported_and_local_s_n_used(tmp_path):
    sl = _tool("audit_scilaws")
    tasks = make_release(str(tmp_path / "release"), np.random.default_rng(3))
    fake = tmp_path / "Crashing" / "harness"
    fake.mkdir(parents=True)
    (fake / "evaluate_numeric.py").write_text("raise UnicodeDecodeError('charmap', b'\\x81', 0, 1, 'undefined')\n")
    out = tmp_path / "out"
    sl.main(["--tasks", tasks, "--harness", str(tmp_path / "Crashing"), "--out", str(out), "--skip-audit",
             "--jobs", "1", "--only", "line_task__y"])
    r = json.loads((out / "scilaws_audit.json").read_text())
    tier = r["tasks"][0]["fits"]["tiers"]["prompt"]
    assert tier["s_n_source"] == "local (harness crashed)" and tier["s_n"] == tier["s_n_local"] > 0.4
    assert "UnicodeDecodeError" in tier["harness_error"]
    assert r["summary"]["harness_crashes"]["tasks"] == ["line_task__y"]
    assert "The harness crashed" in (out / "scilaws_audit.md").read_text()


def test_harness_is_run_in_a_child_process_and_its_json_parsed(tmp_path):
    sl = _tool("audit_scilaws")
    fake = tmp_path / "SciLaws-Bench" / "harness"
    fake.mkdir(parents=True)
    (fake / "evaluate_numeric.py").write_text(
        "import json, sys\n"
        "assert sys.argv[1] == 'score'\n"
        "assert sys.flags.utf8_mode, 'run the harness in UTF-8 mode'\n"
        "print('BLAS warning before the JSON')\n"
        "print(json.dumps({'status': 'ok', 'contract_ok': True, 'numeric_score': 0.42, 'n': len(sys.argv)}))\n")
    sub = tmp_path / "sub.py"
    sub.write_text("x = 1\n")
    root = sl.harness_dir(str(tmp_path / "SciLaws-Bench"))
    res = sl.run_scorer(root, str(tmp_path), str(sub), str(tmp_path))
    assert res["numeric_score"] == 0.42 and res["status"] == "ok"
    bad = tmp_path / "Broken" / "harness"
    bad.mkdir(parents=True)
    (bad / "evaluate_numeric.py").write_text("raise RuntimeError('boom')\n")
    res = sl.run_scorer(str(tmp_path / "Broken"), str(tmp_path), str(sub), str(tmp_path))
    assert res["status"] == "score_error" and res["numeric_score"] == 0.0


# ---------------------------------------------------------------------------- InfiniteScienceGym-like items
def _qa(question, answer, kind, conditions=0, variables=None):
    has = answer is not None
    return {"question": {"question": question, "variables": variables or {}},
            "answer": {"answer": answer, "has_answer": has, "_type": kind if has else None},
            "scope": "SINGLE_FILE", "path_conditions": [], "file_conditions": [{"c": k} for k in range(conditions)],
            "paraphrases": {}}


def make_isg(root, rng, n_fs=40):
    """Question files in ISG's format (2_generate_questions.py) and a seeds.txt over them."""
    qa_dir = os.path.join(root, "qa")
    os.makedirs(qa_dir)
    keys = []
    for fs in range(n_fs):
        ext = [".csv", ".json", ".jsonl"][fs % 3]
        qa = {"has_readme": _qa("Yes or no, does this repository have a README file?", "yes" if fs % 2 else "no", "str"),
              "get_title": _qa("Is the project title in the README file? If yes, what is it?",
                               f"Title {fs}" if fs % 4 == 0 else None, "str"),
              "get_abstract": _qa("Is the project abstract in the README file? If yes, what is it?", None, "str"),
              "file_extension": _qa("What is the file extension for the data in this repository?", ext, "str")}
        for s in range(5):
            n_cond = int(rng.integers(0, 4))
            stat = ["median value", "average value"][s % 2]
            value = None if n_cond >= 2 else float(np.round(rng.normal(10, 3), 2))
            qa[f"univariate_statistic_conditions_{s}"] = _qa(
                'In the file "{path}", ' + ("only considering rows where the a > 1, " if n_cond else "")
                + f'what is the {stat} of the "v" variable? If your answer is numeric, report 3 significant figures. ',
                value, "float", conditions=n_cond, variables={"{path}": f"d/{fs}_{s}{ext}"})
            hyp = None if n_cond >= 2 else ["yes", "no"][int(rng.integers(0, 2))]
            qa[f"bivariate_hypothesis_{s}"] = _qa(
                'In the file "{path}", using "pearson" and a p-value of 0.05, can you reject the null hypothesis '
                '(yes/no) that x and y are unrelated?', hyp, "str", conditions=n_cond, variables={"{path}": "f"})
            qa[f"count_rows_{s}"] = _qa('How many rows of data (excluding headers) are in the file: "{path}"?',
                                        int(rng.integers(50, 60)), "int", variables={"{path}": "f"})
        with open(os.path.join(qa_dir, f"{fs}.json"), "w") as f:
            json.dump(qa, f)
        keys += [f"{fs}.has_readme", f"{fs}.get_title", f"{fs}.file_extension"]
        keys += [f"{fs}.{t}_{int(rng.integers(0, 5))}" for t in ("univariate_statistic_conditions",
                                                                   "bivariate_hypothesis", "count_rows")]
    keys.append(f"{n_fs + 7}.count_rows_0")  # a repository that failed to generate
    seeds = os.path.join(root, "seeds.txt")
    with open(seeds, "w") as f:
        f.write("\n".join(keys) + "\n")
    return qa_dir, seeds


def test_isg_grading_rules_and_their_hazards():
    ig = _tool("audit_isg")
    hyp_na = {"type": "bivariate_hypothesis", "has_answer": False, "answer": None, "answer_type": None,
              "choices": ["yes", "no"]}
    assert not ig.grade(hyp_na, "not possible", "literal")  # it contains the choice "no"
    assert ig.grade(hyp_na, "Not possible.", "exact")
    jsonl = {"type": "file_extension", "has_answer": True, "answer": ".jsonl", "answer_type": "str",
             "choices": [".csv", ".json", ".jsonl"]}
    assert not ig.grade(jsonl, ".jsonl", "literal") and ig.grade(jsonl, ".jsonl", "exact")
    json_ = dict(jsonl, answer=".json")
    assert ig.grade(json_, "The extension is .json", "literal") and not ig.grade(json_, ".jsonl", "literal")
    num = {"type": "count_rows", "has_answer": True, "answer": 10, "answer_type": "int", "choices": None}
    assert ig.grade(num, "10cm", "literal") and not ig.grade(num, "11", "literal")
    flt = {"type": "univariate_statistic_conditions", "has_answer": True, "answer": 1.23, "answer_type": "float",
           "sig_figs": 3, "choices": None}
    assert ig.grade(flt, "1.24", "literal") and not ig.grade(flt, "1.31", "literal")  # one figure fewer: 1.2
    assert ig.grade({**flt, "has_answer": False}, "not possible", "literal")


def test_isg_seeds_check_and_blind_baselines(tmp_path):
    ig = _tool("audit_isg")
    qa_dir, seeds = make_isg(str(tmp_path), np.random.default_rng(5))
    out = tmp_path / "out"
    assert ig.main(["audit", "--seeds", seeds, "--qa", qa_dir, "--out", str(out), "--permutations", "40",
                    "--repeats", "10"]) == 0
    r = json.loads((out / "isg_audit.json").read_text())
    assert r["seeds_check"]["n_keys"] == 241 and r["seeds_check"]["types"]["get_abstract"]["seeds_txt"] == 0
    assert r["regeneration"]["missing"] == 1 and r["regeneration"]["n_items"] == 240
    b = r["baselines"]
    abstain = b["always 'not possible'"]["exact"]["overall"]
    assert abstain == pytest.approx(r["regeneration"]["unanswerable_share"])
    # unanswerability follows the number of conditions, which the question text shows
    text = b["question text"]
    assert text["exact"]["overall"] > b["most common answer of the type"]["exact"]["overall"]
    assert text["exact"]["permutation_p"] < 0.05
    assert r["abstention"]["question text"]["recall"] > 0.8
    assert text["exact"]["by_type"]["get_title"] == pytest.approx(0.75)  # titles mostly hidden: abstain
    assert 0.3 < b["most common answer of the type"]["exact"]["by_type"]["has_readme"] < 0.7  # yes/no, balanced
    hz = r["grader_hazards"]
    assert hz["by_type"].get("file_extension", 0) > 0 and hz["by_type"].get("bivariate_hypothesis", 0) > 0
    md = (out / "isg_audit.md").read_text()
    assert "Table 7" in md and "Blind baselines" in md and "Grading rules" in md
    # the same files twice: identical
    assert ig.main(["audit", "--seeds", seeds, "--qa", qa_dir, "--compare-qa", qa_dir, "--out", str(out),
                    "--permutations", "5", "--repeats", "2"]) == 0
    c = json.loads((out / "isg_audit.json").read_text())["compare"]
    assert c["shared"] == c["same_answer"] == 240


def _fake_isg(tmp_path):
    """A stand-in for ISG's package (no model, no GPU); FAKE_LAYOUT sets the model's device map."""
    isg = tmp_path / "isg"
    pkg = isg / "simulator"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from .llm import LLM\n"
        "OUTPUTS = []\n"
        "SAMPLING = dict(do_sample=True, max_new_tokens=300000, temperature=0.7, top_p=0.8, top_k=20)\n"
        "class _FS:\n    def __init__(self, s): self.s = s\n    def to_json(self): return {'seed': self.s}\n"
        "class Simulator:\n"
        "    def __init__(self, story_config, fs_config, file_config, device_map, model_name):\n"
        "        self.llm = LLM(model_name, device_map)\n"
        "    def get_filesystem(self, seed):\n"
        "        if seed == 3:\n            raise RuntimeError('corrupted')\n"
        "        story = self.llm.pipe([{'role': 'user', 'content': 'story'}], **SAMPLING)\n"
        "        code = self.llm.pipe([{'role': 'user', 'content': 'code'}, {'role': 'assistant', 'content': 'PRE:'}],\n"
        "                             continue_final_message=True, **SAMPLING)\n"
        "        OUTPUTS.append((story, code))\n"
        "        return _FS(seed)\n")
    (pkg / "llm.py").write_text(
        "import json, os, types\n"
        "CALLS, GENERATE = [], []\n"
        "class _Tok:\n    def __call__(self, text, add_special_tokens=False): return {'input_ids': text.split()}\n"
        "class _Pipe:\n    tokenizer = _Tok()\n"
        "    model = types.SimpleNamespace(hf_device_map=json.loads(os.environ.get('FAKE_LAYOUT', 'null')))\n"
        "    def __call__(self, messages, **kw):\n        GENERATE.append(kw)\n"
        "        return [{'generated_text': messages + [{'role': 'assistant', 'content': 'a b c'}]}]\n"
        "def pipeline(*a, **kw):\n    CALLS.append(kw)\n    return _Pipe()\n"
        "class LLM:\n    def __init__(self, model_name, device_map):\n"
        "        self.pipe = pipeline('text-generation', model=model_name, device_map=device_map, trust_remote_code=True)\n")
    (pkg / "config.py").write_text("class StoryConfig:\n    def __init__(self, taxonomy=None): pass\n"
                                   "class FileSystemConfig: pass\nclass FileConfig: pass\n")
    (pkg / "utils.py").write_text("def load_taxonomy(path):\n    return {}\n")
    (isg / "seeds.txt").write_text("1.file_extension\n3.count_rows_1\n1.count_rows_0\n7.get_title\n")
    return isg


def _fake_torch(monkeypatch, cuda):
    import types
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32, torch.__version__ = "bf16", "f16", "f32", "0"
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda i: "Fake GPU",
                                       mem_get_info=lambda: (6 * 2 ** 30, 16 * 2 ** 30))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sys, "path", list(sys.path))


def _forget_fake_isg():
    for name in [m for m in sys.modules if m == "simulator" or m.startswith("simulator.")]:
        del sys.modules[name]


def test_isg_generate_runs_only_the_needed_seeds_and_resumes(tmp_path, monkeypatch):
    ig = _tool("audit_isg")
    isg = _fake_isg(tmp_path)
    _fake_torch(monkeypatch, cuda=False)
    try:
        fs_dir = tmp_path / "fs"
        assert ig.main(["generate", "--isg", str(isg), "--fs-dir", str(fs_dir)]) == 0
        assert sorted(p.name for p in fs_dir.glob("*.json")) == ["1.json", "7.json", "regeneration_manifest.json"]
        man = json.loads((fs_dir / "regeneration_manifest.json").read_text())
        assert man["n_done"] == 2 and list(man["failed"]) == ["3"] and man["dtype"] == "bfloat16"
        llm = sys.modules["simulator.llm"]
        assert llm.CALLS[0]["trust_remote_code"] is False and llm.CALLS[0]["torch_dtype"] == "bf16"
        assert llm.CALLS[0]["device_map"] == "cpu" and man["device_map"] == "cpu"  # no GPU here
        assert [g["max_new_tokens"] for g in llm.GENERATE] == [8192] * 4  # ISG's 300000, capped
        assert man["model_calls"]["1"] == {"n": 2, "tokens": 6, "cut": 0, "seconds": man["model_calls"]["1"]["seconds"],
                                           "prompt_tokens": None, "max_prompt": None,
                                           "max_context": None}  # transformers: not reported
        assert man["max_new_tokens"] == 8192 and man["shard"] == [1, 1]
        assert man["where"]["1"].startswith("transformers, Qwen/Qwen3-4B-Instruct-2507 in bfloat16")
        assert ig.main(["generate", "--isg", str(isg), "--fs-dir", str(fs_dir)]) == 0  # only seed 3 retried
        assert list(json.loads((fs_dir / "regeneration_manifest.json").read_text())["failed"]) == ["3"]
    finally:
        _forget_fake_isg()


def test_isg_generate_refuses_a_model_partly_on_the_cpu(tmp_path, monkeypatch):
    ig = _tool("audit_isg")
    isg = _fake_isg(tmp_path)
    _fake_torch(monkeypatch, cuda=True)
    monkeypatch.setenv("FAKE_LAYOUT", json.dumps({"model.embed_tokens": 0, "model.layers.0": 0, "model.layers.35": "cpu"}))
    fs_dir = tmp_path / "fs"
    try:
        with pytest.raises(SystemExit, match="part of the model is on cpu"):
            ig.main(["generate", "--isg", str(isg), "--fs-dir", str(fs_dir), "--device-map", "auto"])
        assert not fs_dir.joinpath("1.json").exists()
        assert ig.main(["generate", "--isg", str(isg), "--fs-dir", str(fs_dir), "--device-map", "auto",
                        "--allow-cpu"]) == 0
        man = json.loads((fs_dir / "regeneration_manifest.json").read_text())
        assert man["placement"] == {"cuda:0": 2, "cpu": 1} and man["gpu_free_gb_before_loading"] == 6.0
        _forget_fake_isg()
        monkeypatch.setenv("FAKE_LAYOUT", json.dumps({"": 0}))
        assert ig.main(["generate", "--isg", str(isg), "--fs-dir", str(tmp_path / "fs2")]) == 0
        assert sys.modules["simulator.llm"].CALLS[0]["device_map"] == "cuda:0"  # the default with a GPU
    finally:
        _forget_fake_isg()


class _FakeOllama:
    """An Ollama server stand-in on a free local port: records requests, answers like Ollama."""

    def __init__(self, vram_share=1.0, done_reason=None):
        import http.server
        import threading
        self.requests, self.vram_share, self.done_reason = [], vram_share, done_reason
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/version":
                    return self._send({"version": "0.0-test"})
                size = 9 * 2 ** 30
                self._send({"models": [{"name": "qwen3:4b-instruct-2507-fp16", "model": "qwen3:4b-instruct-2507-fp16",
                                        "size": size, "size_vram": int(size * outer.vram_share), "digest": "abc123",
                                        "details": {"quantization_level": "F16"}}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                if not body.get("prompt"):
                    return self._send({"done": True})
                reply = {"response": " written", "eval_count": 7, "eval_duration": 5e8, "prompt_eval_count": 5,
                         "done": True}
                if outer.done_reason:
                    reply["done_reason"] = outer.done_reason
                self._send(reply)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, continue_final_message):
        return json.dumps({"messages": messages, "generation_prompt": add_generation_prompt,
                           "continue": continue_final_message})

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_isg_generate_with_ollama_renders_prompts_like_the_pipeline(tmp_path, monkeypatch):
    ig = _tool("audit_isg")
    isg = _fake_isg(tmp_path)
    _fake_torch(monkeypatch, cuda=True)
    monkeypatch.setattr(ig, "load_tokenizer", lambda name: _FakeTokenizer())
    server = _FakeOllama()
    fs_dir = tmp_path / "fs"
    try:
        assert ig.main(["generate", "--isg", str(isg), "--fs-dir", str(fs_dir), "--backend", "ollama",
                        "--ollama-url", server.url, "--num-ctx", "20000"]) == 0
        assert sys.modules["simulator.llm"].CALLS == []  # transformers never loaded
        writes = [r for r in server.requests if r.get("prompt")]
        assert len(writes) == 4 and all(r["raw"] and not r["stream"] for r in writes)
        o = writes[0]["options"]
        assert (o["num_predict"], o["temperature"], o["top_p"], o["top_k"], o["min_p"], o["num_ctx"]) == \
            (8192, 0.7, 0.8, 20, 0.0, 20000)
        assert "<|im_end|>" in o["stop"] and len({r["options"]["seed"] for r in writes}) == 4
        first, prefill = (json.loads(r["prompt"]) for r in writes[:2])
        assert first["generation_prompt"] and not first["continue"]
        assert prefill["continue"] and not prefill["generation_prompt"]
        story, code = sys.modules["simulator"].OUTPUTS[0]
        assert story[0]["generated_text"][-1] == {"role": "assistant", "content": " written"}
        assert code[0]["generated_text"][-1] == {"role": "assistant", "content": "PRE: written"}
        man = json.loads((fs_dir / "regeneration_manifest.json").read_text())
        assert man["backend"]["name"] == "ollama" and man["backend"]["vram_gb"] == 9.0
        assert man["model_calls"]["1"]["tokens"] == 14  # Ollama's own counts
        assert man["dtype"] == "F16"  # what Ollama serves, not --dtype
        assert man["model_calls"]["1"]["prompt_tokens"] == 10 and man["model_calls"]["1"]["max_context"] == 12
        assert man["model_calls"]["1"]["max_prompt"] == 5
    finally:
        server.close()
        _forget_fake_isg()


def test_isg_generate_with_ollama_stops_when_the_model_spills_to_the_cpu(tmp_path, monkeypatch):
    ig = _tool("audit_isg")
    isg = _fake_isg(tmp_path)
    _fake_torch(monkeypatch, cuda=True)
    monkeypatch.setattr(ig, "load_tokenizer", lambda name: _FakeTokenizer())
    server = _FakeOllama(vram_share=0.7)
    try:
        with pytest.raises(SystemExit, match="part of the model is on the CPU"):
            ig.main(["generate", "--isg", str(isg), "--fs-dir", str(tmp_path / "fs"), "--backend", "ollama",
                     "--ollama-url", server.url])
        with pytest.raises(SystemExit, match="not reachable"):
            ig.main(["generate", "--isg", str(isg), "--fs-dir", str(tmp_path / "fs"), "--backend", "ollama",
                     "--ollama-url", "http://127.0.0.1:9"])
    finally:
        server.close()
        _forget_fake_isg()


def test_isg_generate_shards_split_the_seeds_and_record_where_each_was_built(tmp_path, monkeypatch, capsys):
    ig = _tool("audit_isg")
    isg = _fake_isg(tmp_path)  # seeds 1, 3 (fails) and 7
    _fake_torch(monkeypatch, cuda=True)
    monkeypatch.setattr(ig, "load_tokenizer", lambda name: _FakeTokenizer())
    server = _FakeOllama()
    fs_dir = tmp_path / "fs"
    common = ["generate", "--isg", str(isg), "--fs-dir", str(fs_dir), "--backend", "ollama", "--ollama-url", server.url]
    try:
        with pytest.raises(SystemExit, match="other weights"):
            ig.main(common + ["--shard", "1/2", "--ollama-digest", "def"])
        assert not list(fs_dir.glob("*.json"))
        with pytest.raises(SystemExit):  # argparse: not a shard
            ig.main(common + ["--shard", "3/2"])
        assert ig.main(common + ["--shard", "1/2", "--ollama-digest", "sha256:ABC1", "--machine", "box A"]) == 0
        assert sorted(p.name for p in fs_dir.glob("*.json")) == [
            "1.json", "7.json", "regeneration_manifest.shard-1-of-2.json"]  # round-robin: 1, 7 | 3
        assert "largest context in this run: 12 of 16384 tokens (largest prompt 5); a smaller --num-ctx changes no " \
            "reply while it is at least the largest prompt + 8208" in capsys.readouterr().out
        man = json.loads((fs_dir / "regeneration_manifest.shard-1-of-2.json").read_text())
        assert man["shard"] == [1, 2] and man["machine"] == "box A"
        assert man["where"]["7"] == "Ollama 0.0-test, qwen3:4b-instruct-2507-fp16 (abc123), context 16384 on box A"
        # a deadline already passed: nothing started, nothing lost
        assert ig.main(common + ["--shard", "2/2", "--stop-after", "1e-9", "--machine", "box B"]) == 0
        assert "stopping:" in capsys.readouterr().out and not (fs_dir / "3.json").exists()
        assert ig.main(common + ["--shard", "2/2", "--machine", "box B"]) == 0
        assert list(json.loads((fs_dir / "regeneration_manifest.shard-2-of-2.json").read_text())["failed"]) == ["3"]
        # a manifest from before shards (no per-seed record) is pooled with its run's description
        (fs_dir / "regeneration_manifest.json").write_text(json.dumps({
            "model": "Qwen/Qwen3-4B-Instruct-2507", "dtype": "bfloat16", "device": "Old GPU",
            "backend": {"name": "ollama", "version": "0.1", "model": "qwen3:4b-instruct-2507-fp16", "digest": "abc123ff"},
            "seconds": {"9": 3600.0, "3": 10.0}, "failed": {},
            "model_calls": {"9": {"n": 30, "tokens": 30000, "cut": 1, "seconds": 3500.0}}}))
        pooled = ig.merge_manifests(ig.manifest_paths(str(fs_dir)))
        assert pooled["n_done"] == 4 and pooled["n_failed"] == 0  # seed 3 was built by the older run
        assert pooled["built_with"] == {
            "Ollama 0.0-test, qwen3:4b-instruct-2507-fp16 (abc123), context 16384 on box A": 2,
            "Ollama 0.1, qwen3:4b-instruct-2507-fp16 (abc123ff) on Old GPU": 2}
        assert pooled["model_calls"] == 34 and pooled["cut"] == 1 and pooled["largest_context"] == 12
    finally:
        server.close()
        _forget_fake_isg()


def test_isg_audit_pools_the_manifests_next_to_the_questions(tmp_path):
    ig = _tool("audit_isg")
    qa_dir, seeds = make_isg(str(tmp_path), np.random.default_rng(6))
    fs_dir = os.path.join(os.path.dirname(qa_dir), "fs")
    os.makedirs(fs_dir, exist_ok=True)
    for k, (gpu, seeds_done) in enumerate([("Tesla T4", ["0", "1"]), ("RTX A4000", ["2"])], 1):
        with open(os.path.join(fs_dir, f"regeneration_manifest.shard-{k}-of-2.json"), "w") as f:
            json.dump({"seconds": {s: 60.0 for s in seeds_done}, "failed": {},
                       "model_calls": {s: {"n": 3, "tokens": 900, "cut": 0, "seconds": 50.0, "prompt_tokens": 2000,
                                           "max_prompt": 500, "max_context": 1000 + 100 * k} for s in seeds_done},
                       "where": {s: f"Ollama 1, m (d) on {gpu}" for s in seeds_done}}, f)
    out = tmp_path / "out"
    assert ig.main(["audit", "--seeds", seeds, "--qa", qa_dir, "--out", str(out), "--permutations", "2",
                    "--repeats", "2"]) == 0
    g = json.loads((out / "isg_audit.json").read_text())["generation_manifest"]
    assert g["n_done"] == 3 and len(g["files"]) == 2 and g["largest_context"] == 1200
    md = (out / "isg_audit.md").read_text()
    assert "| Ollama 1, m (d) on Tesla T4 | 2 |" in md and "largest context 1200 tokens (largest prompt 500)" in md


def test_isg_machine_label(monkeypatch):
    import subprocess
    import types
    ig = _tool("audit_isg")
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    assert ig.machine_label("my box", torch) == "my box"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="Tesla T4\nTesla T4\n"))
    assert ig.machine_label(None, torch) == "2 x Tesla T4"

    def missing(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", missing)
    assert ig.machine_label(None, torch) == "no GPU found"


def test_isg_generate_counts_a_reply_stopped_by_the_context_as_cut(tmp_path, monkeypatch):
    ig = _tool("audit_isg")
    isg = _fake_isg(tmp_path)
    _fake_torch(monkeypatch, cuda=True)
    monkeypatch.setattr(ig, "load_tokenizer", lambda name: _FakeTokenizer())
    server = _FakeOllama(done_reason="length")  # 7 tokens, far below the 8192 ceiling, but stopped by a limit
    fs_dir = tmp_path / "fs"
    try:
        assert ig.main(["generate", "--isg", str(isg), "--fs-dir", str(fs_dir), "--backend", "ollama",
                        "--ollama-url", server.url, "--limit", "1"]) == 0
        assert json.loads((fs_dir / "regeneration_manifest.json").read_text())["model_calls"]["1"]["cut"] == 2
    finally:
        server.close()
        _forget_fake_isg()


class _FakeVLLM:
    """An OpenAI-compatible server stand-in (vLLM's /v1/models, /version and /v1/completions)."""

    def __init__(self, max_model_len=16384):
        import http.server
        import threading
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/version":
                    return self._send({"version": "0.0-vllm"})
                self._send({"data": [{"id": "Qwen/Qwen3-4B-Instruct-2507", "root": "/models/qwen",
                                      "max_model_len": max_model_len}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                self._send({"choices": [{"text": " written", "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 7}})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_isg_generate_through_an_openai_compatible_server(tmp_path, monkeypatch):
    ig = _tool("audit_isg")
    isg = _fake_isg(tmp_path)
    _fake_torch(monkeypatch, cuda=False)
    monkeypatch.setattr(ig, "load_tokenizer", lambda name: _FakeTokenizer())
    server = _FakeVLLM()
    fs_dir = tmp_path / "fs"
    common = ["generate", "--isg", str(isg), "--fs-dir", str(fs_dir), "--backend", "openai", "--openai-url", server.url,
              "--limit", "1", "--machine", "box V"]
    try:
        with pytest.raises(SystemExit, match="holds 16384 tokens per sequence"):
            ig.main(common + ["--num-ctx", "8192"])
        with pytest.raises(SystemExit, match="--served-model must name one"):
            ig.main(common + ["--served-model", "other"])
        assert ig.main(common + ["--weights-revision", "deadbeef12345678", "--note", "vllm serve --dtype half"]) == 0
        assert (fs_dir / "1.json").exists()
        man = json.loads((fs_dir / "regeneration_manifest.json").read_text())
        assert man["backend"]["engine"] == "vllm" and man["backend"]["note"] == "vllm serve --dtype half"
        assert man["where"]["1"] == ("vLLM 0.0-vllm, Qwen/Qwen3-4B-Instruct-2507 (deadbeef1234), context 16384 "
                                     "on box V")
        assert man["model_calls"]["1"]["max_context"] == 12
        first = server.requests[0]
        assert first["top_k"] == 20 and first["temperature"] == 0.7 and first["stop"] == ig.QWEN_STOP
        assert first["max_tokens"] == 8192 and isinstance(first["seed"], int)
        assert json.loads(first["prompt"])["generation_prompt"] is True  # rendered here, by the model's template
        assert len({r["seed"] for r in server.requests}) == len(server.requests)  # a seed per call
    finally:
        server.close()
        _forget_fake_isg()


def test_isg_audit_complete_prefix_is_fixed_by_seed_order(tmp_path):
    ig = _tool("audit_isg")
    qa_dir, seeds = make_isg(str(tmp_path), np.random.default_rng(8))
    fs_dir = os.path.join(os.path.dirname(qa_dir), "fs")
    os.makedirs(fs_dir, exist_ok=True)
    built = [s for s in range(10) if s != 4] + [12]  # 4 failed; 10 and 11 not reached; 12 was quick
    with open(os.path.join(fs_dir, "regeneration_manifest.shard-1-of-2.json"), "w") as f:
        json.dump({"seconds": {str(s): 60.0 for s in built}, "failed": {"4": "ValueError: x"}, "model_calls": {}}, f)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="needs the generation manifests"):
        ig.main(["audit", "--seeds", seeds, "--qa", qa_dir, "--regeneration-manifest", str(tmp_path / "none"),
                 "--complete-prefix", "--out", str(out)])
    assert ig.main(["audit", "--seeds", seeds, "--qa", qa_dir, "--out", str(out), "--complete-prefix",
                    "--permutations", "2", "--repeats", "2"]) == 0
    r = json.loads((out / "isg_audit.json").read_text())
    sm = r["sample"]
    assert (sm["repositories"], sm["last_seed"], sm["failed_inside"], sm["built_but_left_out"]) == (10, 9, [4], 1)
    keys = [k for k in open(seeds).read().split() if k.strip()]
    assert sm["items"] == sum(int(k.split(".")[0]) < 10 for k in keys)
    assert "up to seed 9" in (out / "isg_audit.md").read_text()


def test_kaggle_notebooks_match_their_generator_and_parse(tmp_path):
    import ast
    import subprocess
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, os.path.join(root, "tools", "make_kaggle_notebooks.py"), str(tmp_path)], check=True)
    for name in ("isg_prepare.ipynb", "isg_generate.ipynb"):
        committed = json.load(open(os.path.join(root, "kaggle", name)))
        assert committed == json.load(open(tmp_path / name)), f"kaggle/{name} differs: rerun the generator"
        for cell in committed["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
    cells = json.load(open(os.path.join(root, "kaggle", "isg_generate.ipynb")))["cells"]
    generate = "".join("".join(c["source"]) for c in cells)
    usage = subprocess.run([sys.executable, os.path.join(root, "tools", "audit_isg.py"), "generate", "--help"],
                           capture_output=True, text=True, check=True).stdout
    passed = set(re.findall(r'"(--[a-z][a-z-]+)"', generate))
    assert {"--backend", "--shard", "--stop-after", "--weights-revision", "--tokenizer"} <= passed
    assert all(flag in usage for flag in passed - {"--complete-prefix"}), passed  # the tool knows every option
