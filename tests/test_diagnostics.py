# tests/test_diagnostics.py
import json
import os

import numpy as np
import pandas as pd
import pytest

from simulator.audit import audit_export
from simulator.bundle import check_answer
from simulator.diagnostics import (add_canary, blind_baseline, canary_probe, card_markdown, contamination_gap,
                                   diagnose_episode, feature_shortcuts, gap_test, law_alternatives, make_canary,
                                   make_twin, pair_tasks, public_features, quality_card, run_episodes, summarize)
from simulator.diagnostics.stats import binomial_p_greater
from simulator.doe import run_design
from simulator.engine import SimulatorEngine
from simulator.envs import ExperimentEnv
from simulator.export.tasks import collect_bundle_tasks

LAW_BASE = {"experiment_type": "experiments.law_discovery.logic.HiddenLawExperiment", "n_points": 25,
            "generate_plots": False, "bundle": {"figure": {"show_latent": False}, "qa": {"value_queries": 0}},
            "observation": {"operators": [{"type": "gaussian", "snr_db": 35}]}}


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    out = tmp_path_factory.mktemp("suite")
    table = pd.DataFrame({"law": ["power_law", "hill", "exponential_decay", "logistic"], "counterfactual": [True] * 4})
    run_design(LAW_BASE, table, output_dir=str(out / "original"), replicates=2, base_seed=7)
    make_twin(str(out / "original"), str(out / "twin"), base_seed=8)
    return str(out / "original"), str(out / "twin")


# ------------------------------------------------------------------ statistics
def test_gap_test_paired_and_unpaired():
    a = np.ones(40)
    b = np.zeros(40)
    r = gap_test(a, b, paired=True, n_resamples=2000)
    assert r["gap"] == 1.0 and r["p_value"] < 0.001 and r["ci"] == (1.0, 1.0)
    same = gap_test(a, a, paired=True, n_resamples=500)
    assert same["gap"] == 0.0 and same["p_value"] == 1.0
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=30), rng.normal(size=25)
    assert gap_test(x, y, n_resamples=2000)["p_value"] > 0.05
    assert gap_test(x + 2, y, n_resamples=2000)["p_value"] < 0.01
    with pytest.raises(ValueError):
        gap_test([1, 2], [1], paired=True)
    assert binomial_p_greater(10, 10, 0.5) == pytest.approx(0.5 ** 10)
    assert binomial_p_greater(0, 10, 0.3) == pytest.approx(1.0)


# ------------------------------------------------------------------ canaries
def test_canary_written_probed_and_not_a_leak(suite, tmp_path):
    canary = make_canary("my release")
    assert canary.startswith("sds-canary:my-release:")
    (tmp_path / "README.md").write_text("# data\n")
    written = add_canary(str(tmp_path), canary)
    assert len(written) == 2 and canary in (tmp_path / "README.md").read_text()
    add_canary(str(tmp_path), canary)
    assert (tmp_path / "README.md").read_text().count(canary) == 1
    seen = canary_probe(lambda p: canary, canary)
    assert seen["hex_matched"] >= 20 and seen["chance_p"] < 1e-20
    tail = canary_probe(lambda p: canary.rsplit(":", 1)[1][8:], canary)  # continuation only
    assert tail["matched_chars"] == seen["matched_chars"]
    unseen = canary_probe(lambda p: "no idea", canary, n_trials=3)
    assert unseen["matched_chars"] == 0 and unseen["chance_p"] == 1.0 and len(unseen["outputs"]) == 3
    with pytest.raises(ValueError):
        add_canary(str(tmp_path), "not-a-canary")
    assert audit_export(str(tmp_path), suite[0]).ok


# ------------------------------------------------------------------ twins and gaps
def test_twin_has_same_design_and_new_laws(suite):
    original, twin = suite
    a = pd.read_csv(os.path.join(original, "design_results.csv"))
    b = pd.read_csv(os.path.join(twin, "design_results.csv"))
    assert list(a["law"]) == list(b["law"]) and list(a["replicate"]) == list(b["replicate"])
    assert not set(a["seed"]) & set(b["seed"])
    assert (a["out.law_expression"] != b["out.law_expression"]).all()
    with open(os.path.join(original, "design_manifest.json")) as f:
        seed = json.load(f)["base_seed"]
    with pytest.raises(ValueError, match="different"):
        make_twin(original, os.path.join(os.path.dirname(original), "bad"), base_seed=seed)


def test_contamination_gap_detects_memorisation(suite):
    original, twin = suite
    ta, tb = collect_bundle_tasks(original), collect_bundle_tasks(twin)
    pairs = pair_tasks(ta, tb, original, twin)
    assert len(pairs) == len(ta) == len(tb)
    by_id_a = {t["id"]: t for t in ta}
    by_id_b = {t["id"]: t for t in tb}
    assert all(by_id_a[x]["template"] == by_id_b[y]["template"] for x, y in pairs)
    templates = {t["id"]: t["template"] for t in ta}
    memorised = {t["id"]: 1.0 for t in ta}  # recites the leaked key
    fresh = {t["id"]: float(check_answer(t, "nonsense" if t["answer_type"] != "number" else 0.0)) for t in tb}
    gap = contamination_gap(memorised, fresh, pairs, templates, n_resamples=2000)
    assert gap.gap > 0.8 and gap.p_value < 0.01 and "law_family" in gap.by_template
    assert "gap" in gap.summary()
    honest = contamination_gap(memorised, {t["id"]: 1.0 for t in tb}, pairs, n_resamples=500)
    assert honest.gap == 0.0 and honest.p_value == 1.0
    with pytest.raises(ValueError):
        contamination_gap({}, {}, pairs)


# ------------------------------------------------------------------ shortcuts and quality
def _choice(i, answer, choices, eid=None):
    return {"id": f"r{i}/q1", "experiment_id": eid or f"r{i}", "template": "pick", "answer_type": "choice",
            "answer": answer, "choices": choices, "tolerance": None, "prompt": f"Pick {i}", "question": f"Pick {i}",
            "fidelity": "reference", "answerable_from": "figure"}


def _number(i, answer, template="num"):
    return {"id": f"r{i}/{template}", "experiment_id": f"r{i}", "template": template, "answer_type": "number",
            "answer": answer, "tolerance": 0.1, "prompt": f"Value {i}?", "question": f"Value {i}?",
            "fidelity": "audit", "answerable_from": "data"}


def test_blind_baseline_finds_majority_and_position_bias():
    skewed = [_choice(i, "a", ["a", "b", "c", "d"]) for i in range(30)]
    report = blind_baseline(skewed)["pick"]
    assert report["accuracy"] == 1.0 and report["chance"] == 0.25 and report["p_value"] < 1e-10
    rng = np.random.default_rng(1)
    position = []
    for i in range(40):  # balanced labels, but the answer is always listed first
        answer = ["w", "x", "y", "z"][i % 4]
        position.append(_choice(i, answer, [answer] + [c for c in "wxyz" if c != answer]))
    report = blind_baseline(position)["pick"]
    assert report["position_accuracy"] == 1.0
    balanced = [_choice(i, c, list(rng.permutation(list("wxyz")))) for i, c in enumerate("wxyz" * 10)]
    assert blind_baseline(balanced)["pick"]["p_value"] > 0.01
    constant = [_number(i, 3.0) for i in range(10)]
    assert blind_baseline(constant)["num"]["accuracy"] == 1.0


def test_feature_shortcuts_permutation_test():
    rng = np.random.default_rng(2)
    tasks = [dict(_number(i, float(v)), tolerance=0.6) for i, v in enumerate(rng.uniform(0, 10, 40))]
    leaky = {t["experiment_id"]: {"figure_bytes": t["answer"] * 1000 + rng.normal(0, 1)} for t in tasks}
    report = feature_shortcuts(tasks, leaky, k=1, n_permutations=100)["num"]
    assert report["accuracy"] > 0.5 and report["p_value"] < 0.05 and report["gain"] > 0.3
    noise = {t["experiment_id"]: {"figure_bytes": rng.normal()} for t in tasks}
    assert feature_shortcuts(tasks, noise, k=1, n_permutations=100)["num"]["p_value"] > 0.05
    with pytest.raises(ValueError):
        feature_shortcuts(tasks, {t["experiment_id"]: {} for t in tasks})


def test_quality_card():
    tasks = [_choice(i, "a", ["a", "b"]) for i in range(10)] + [_number(i, 1.0 + i) for i in range(5)]
    tasks.append(dict(tasks[-1], id="dup"))
    card = quality_card(tasks, solver=lambda t: t["answer"])
    pick, num = card["templates"]["pick"], card["templates"]["num"]
    assert pick["majority_share"] == 1.0 and pick["position_counts"] == [10, 0] and pick["position_p"] < 0.01
    assert num["duplicates"] == 1 and num["solver_accuracy"] == 1.0 and num["distinct_ratio"] == pytest.approx(5 / 6)
    assert card["fidelity"] == {"reference": 10, "audit": 6}
    assert "| pick | 10 | choice |" in card_markdown(card)


def test_public_features_and_excluded_templates(tmp_path):
    cfg = dict(LAW_BASE, law="power_law", seed=3,
               bundle={"qa": {"value_queries": 0, "exclude": ["argmax", "peak_count"]}})
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(cfg)
    templates = [t["template"] for t in collect_bundle_tasks(run.experiment_dir)]
    assert "argmax" not in templates and "peak_count" not in templates and "law_formula" in templates
    ids = [t["id"].split("/")[1] for t in collect_bundle_tasks(run.experiment_dir)]
    assert ids == [f"q{i + 1}" for i in range(len(ids))]
    features = public_features(run.experiment_dir)
    row = features[run.experiment_id]
    assert row["figure_bytes"] > 1000 and row["caption_chars"] > 10 and "table_rows" not in row
    with pytest.raises(ValueError):
        SimulatorEngine(str(tmp_path), verbose=False).run_config(dict(cfg, bundle={"qa": {"exclude": "argmax"}}))


# ------------------------------------------------------------------ agent fallacies
SPEC = {"base_config": {"experiment_type": "experiments.law_discovery.logic.HiddenLawExperiment", "n_points": 30,
                        "counterfactual": True, "x_min": 0.5, "x_max": 5.0,
                        "observation": {"operators": [{"type": "gaussian", "snr_db": 40}]}},
        "hidden": {"law": {"choices": ["power_law"]}, "law_seed": [0, 1.0e12]},
        "controls": {"x_min": [0.1, 9.0], "x_max": [0.5, 10.0]},
        "target": "law_expression", "target_type": "expression", "domain": [0.5, 5.0], "budget": 3,
        "tolerance": 0.02}


def _agent(kind):
    from experiments.law_discovery.logic import textbook_counterpart

    def agent(obs, log):
        n = len(log.experiments)
        if kind == "guess":
            return {"type": "answer", "value": "2*x**1.5"}
        if kind == "repeat" and n < 2:
            return {"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}}
        if kind == "narrow" and n == 0:
            return {"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 1.0}}
        if kind in ("oracle", "textbook", "stubborn") and n == 0:
            return {"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}}
        if kind == "garbage":
            return {"type": "answer", "value": "import os"}
        truth = log.truth_hint
        if kind == "textbook":
            return {"type": "answer", "value": textbook_counterpart("power_law", truth, (0.5, 5.0))}
        if kind == "stubborn":
            return {"type": "answer", "value": "3*x"}
        return {"type": "answer", "value": truth}
    return agent


def _play(kind, n=4):
    """Scripted agents that may see the truth, to test the flags apart from fitting skill."""
    env = ExperimentEnv(SPEC)
    original_reset = env.reset

    def reset(seed=None):
        obs = original_reset(seed=seed)
        _play.truth = env.truth["law_expression"]
        return obs
    env.reset = reset

    def agent(obs, log):
        log.truth_hint = _play.truth
        return _agent(kind)(obs, log)
    logs = run_episodes(env, agent, seeds=range(n))
    return [diagnose_episode(log, law_alternatives(log)) for log in logs], logs


def test_flags_separate_the_fallacies():
    flags = {}
    for kind in ("oracle", "guess", "repeat", "narrow", "textbook", "stubborn", "garbage"):
        diagnoses, _ = _play(kind)
        flags[kind] = summarize(diagnoses)
    oracle = flags["oracle"]
    assert oracle["accuracy"] == 1.0 and all(v["rate"] == 0 for v in oracle["flags"].values())
    assert flags["guess"]["flags"]["answer_without_evidence"]["rate"] == 1.0
    assert flags["repeat"]["flags"]["repeated_experiment"]["rate"] == 1.0
    assert flags["narrow"]["flags"]["narrow_sampling"]["rate"] == 1.0
    tb = flags["textbook"]
    assert tb["flags"]["matches_textbook"]["rate"] >= 0.5 and tb["accuracy"] <= 0.5
    assert flags["stubborn"]["flags"]["contradicts_own_data"]["rate"] == 1.0
    assert flags["garbage"]["flags"]["invalid_answer"]["rate"] == 1.0


def test_episode_log_serialises_and_no_answer_flag():
    diagnoses, logs = _play("oracle", n=1)
    data = logs[0].to_dict()
    assert data["truth"]["law"] == "power_law" and data["steps"][-1]["action"]["type"] == "answer"
    env = ExperimentEnv(SPEC)
    logs = run_episodes(env, lambda obs, log: {"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5.0}},
                        seeds=[0])
    d = diagnose_episode(logs[0])
    assert d["flags"]["no_answer"] and d["n_experiments"] == 3 and not d["correct"]


# ------------------------------------------------------------------ review regressions
def test_clustered_gap_test_keeps_its_size():
    rng = np.random.default_rng(5)
    rejections = 0
    for trial in range(200):  # null true, three perfectly correlated questions per run
        run_effect = rng.choice([-1.0, 0.0, 1.0], size=36)
        d = np.repeat(run_effect, 3)
        clusters = np.repeat(np.arange(36), 3)
        r = gap_test(d, np.zeros_like(d), paired=True, n_resamples=400, seed=trial, clusters=list(clusters))
        rejections += r["p_value"] < 0.05
    assert rejections / 200 < 0.09
    assert r["n_clusters"] == 36
    with pytest.raises(ValueError):
        gap_test([1, 2], [1, 2], clusters=[0, 1])


def test_contradiction_flag_spares_correct_answers_and_numeric_invalid():
    from simulator.diagnostics.episodes import EpisodeLog
    xs = list(np.linspace(0.5, 5, 30))
    truth = "2*x**1.7"
    ys = list(2 * np.asarray(xs) ** 1.7)  # noise-free measurements
    step = {"action": {"type": "experiment", "inputs": {"x_min": 0.5, "x_max": 5}},
            "observation": {"experiment": 1, "measurements": {"y": {"values": ys, "x": xs}}}, "reward": None,
            "info": {}}
    near = {"action": {"type": "answer", "value": "1.01*2*x**1.7"}, "observation": {"answer": "..."},
            "reward": 1.0, "info": {"correct": True}}
    log = EpisodeLog(seed=0, task={}, steps=[step, near], truth={"law_expression": truth},
                     target="law_expression", target_type="expression", domain=[0.5, 5], tolerance=0.02, budget=3)
    d = diagnose_episode(log, {"textbook": truth})
    assert not d["flags"]["contradicts_own_data"] and "matches_textbook" not in d["flags"]
    numeric = EpisodeLog(seed=0, task={}, steps=[step, {"action": {"type": "answer", "value": "abc"},
                                                         "observation": {"answer": "abc"}, "reward": 0.0,
                                                         "info": {"estimate": None, "correct": False}}],
                         target="k", target_type="auto", tolerance=0.05, budget=3)
    assert diagnose_episode(numeric)["flags"]["invalid_answer"]


def test_canary_chance_skips_fixed_uuid_digits():
    canary = make_canary("x")
    guid = canary.rsplit(":", 1)[1]
    full = canary_probe(lambda p: canary, canary, reveal=8)
    # 28 hidden characters: 4 dashes, the fixed version digit, the variant digit (2 bits), 22 free digits
    assert full["hex_matched"] == 23 and full["bits"] == 22 * 4 + 2
    assert full["chance_p"] == pytest.approx(2.0 ** -90)
    assert guid[14] == "4"


def test_twin_of_pre_replicated_table_and_numpy_answers(tmp_path):
    from simulator.doe import add_replicates_and_seeds
    table = add_replicates_and_seeds(pd.DataFrame({"law": ["power_law", "hill"]}), replicates=3, base_seed=4)
    run_design(LAW_BASE, table, output_dir=str(tmp_path / "o"))
    twin = make_twin(str(tmp_path / "o"), str(tmp_path / "t"), base_seed=99)
    assert len(twin.table) == 6
    ta, tb = collect_bundle_tasks(str(tmp_path / "o")), collect_bundle_tasks(str(tmp_path / "t"))
    assert len(pair_tasks(ta, tb, str(tmp_path / "o"), str(tmp_path / "t"))) == len(ta)
    ints = [dict(_number(i, np.int64(2)), answer_type="integer", tolerance=None) for i in range(5)]
    assert blind_baseline(ints)["num"]["accuracy"] == 1.0
    card = quality_card([_number(0, None), _number(1, float("nan")), _number(2, 1.0)])
    assert card["templates"]["num"]["non_finite"] == 2
    assert binomial_p_greater(1300, 3000, 0.4) < 0.01


def test_feature_shortcuts_groups_exclude_replicates():
    rng = np.random.default_rng(3)
    tasks, features, groups = [], {}, {}
    for point in range(12):  # replicates share the answer and a near-identical feature
        answer = ["a", "b", "c"][point % 3]
        for rep in range(3):
            eid = f"p{point}r{rep}"
            tasks.append(_choice(len(tasks), answer, ["a", "b", "c"], eid=eid))
            features[eid] = {"figure_bytes": point + rng.normal(0, 0.01)}
            groups[eid] = point
    leaky = feature_shortcuts(tasks, features, k=2, n_permutations=100)["pick"]
    grouped = feature_shortcuts(tasks, features, k=2, n_permutations=100, groups=groups)["pick"]
    assert leaky["accuracy"] == 1.0 and grouped["accuracy"] < leaky["accuracy"]
