# tests/test_audit_benchmarks.py - the audits of outside benchmarks (SciLaws-Bench, InfiniteScienceGym)
import importlib.util
import json
import os
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


def test_harness_is_run_in_a_child_process_and_its_json_parsed(tmp_path):
    sl = _tool("audit_scilaws")
    fake = tmp_path / "SciLaws-Bench" / "harness"
    fake.mkdir(parents=True)
    (fake / "evaluate_numeric.py").write_text(
        "import json, sys\n"
        "assert sys.argv[1] == 'score'\n"
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


def test_isg_generate_runs_only_the_needed_seeds_and_resumes(tmp_path, monkeypatch):
    """The 'generate' loop against a stand-in for ISG's package (no model, no GPU)."""
    import types
    ig = _tool("audit_isg")
    isg = tmp_path / "isg"
    pkg = isg / "simulator"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from .llm import LLM\n"
        "class _FS:\n    def __init__(self, s): self.s = s\n    def to_json(self): return {'seed': self.s}\n"
        "class Simulator:\n"
        "    def __init__(self, story_config, fs_config, file_config, device_map, model_name):\n"
        "        self.llm = LLM(model_name, device_map)\n"
        "    def get_filesystem(self, seed):\n"
        "        if seed == 3:\n            raise RuntimeError('corrupted')\n"
        "        return _FS(seed)\n")
    (pkg / "llm.py").write_text(
        "CALLS = []\n"
        "def pipeline(*a, **kw):\n    CALLS.append(kw)\n"
        "class LLM:\n    def __init__(self, model_name, device_map):\n"
        "        self.pipe = pipeline('text-generation', model=model_name, device_map=device_map, trust_remote_code=True)\n")
    (pkg / "config.py").write_text("class StoryConfig:\n    def __init__(self, taxonomy=None): pass\n"
                                   "class FileSystemConfig: pass\nclass FileConfig: pass\n")
    (pkg / "utils.py").write_text("def load_taxonomy(path):\n    return {}\n")
    (isg / "seeds.txt").write_text("1.file_extension\n3.count_rows_1\n1.count_rows_0\n7.get_title\n")
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32, torch.__version__ = "bf16", "f16", "f32", "0"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        fs_dir = tmp_path / "fs"
        assert ig.main(["generate", "--isg", str(isg), "--fs-dir", str(fs_dir)]) == 0
        assert sorted(p.name for p in fs_dir.glob("*.json")) == ["1.json", "7.json", "regeneration_manifest.json"]
        man = json.loads((fs_dir / "regeneration_manifest.json").read_text())
        assert man["n_done"] == 2 and list(man["failed"]) == ["3"] and man["dtype"] == "bfloat16"
        calls = sys.modules["simulator.llm"].CALLS
        assert calls[0]["trust_remote_code"] is False and calls[0]["torch_dtype"] == "bf16"
        assert ig.main(["generate", "--isg", str(isg), "--fs-dir", str(fs_dir)]) == 0  # only seed 3 retried
        assert list(json.loads((fs_dir / "regeneration_manifest.json").read_text())["failed"]) == ["3"]
    finally:
        for name in [m for m in sys.modules if m == "simulator" or m.startswith("simulator.")]:
            del sys.modules[name]
