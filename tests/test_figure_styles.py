# tests/test_figure_styles.py - figure style and clutter as design factors
import glob
import hashlib
import json
import os

import pandas as pd
import pytest

from scidatasim.bundle import resolve_bundle_config
from scidatasim.config import load_config
from scidatasim.doe import cross_styles, run_design
from scidatasim.engine import SimulatorEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = load_config(os.path.join(ROOT, "examples", "example_10_law_discovery", "config.yaml"))


def test_presets_fill_only_what_the_config_leaves_open():
    dense = resolve_bundle_config({"figure": {"style": "dense"}})["figure"]
    assert dense["distractors"] == 2 and dense["annotations"] == 4 and dense["ticks"] == "dense"
    mine = resolve_bundle_config({"figure": {"style": "dense", "distractors": 0}})["figure"]
    assert mine["distractors"] == 0 and mine["annotations"] == 4
    assert resolve_bundle_config({"figure": {"style": "print"}})["figure"]["palette"] == "grayscale"
    for bad in ({"style": "fancy"}, {"palette": "rainbow"}, {"ticks": "many"}, {"distractors": 9},
                {"annotations": -1}, {"line_width": 0}, {"legend_position": "left"}, {"distractors": True}):
        with pytest.raises(ValueError):
            resolve_bundle_config({"figure": bad})


def _bundle(tmp_path, figure, seed=3):
    cfg = dict(BASE, seed=seed, bundle={**BASE["bundle"], "figure": {**BASE["bundle"]["figure"], **figure}})
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(cfg)
    return os.path.join(run.experiment_dir, "bundle")


def test_clutter_is_drawn_named_and_reproducible(tmp_path):
    a = _bundle(tmp_path / "a", {"distractors": 2, "annotations": 3, "palette": "grayscale"})
    b = _bundle(tmp_path / "b", {"distractors": 2, "annotations": 3, "palette": "grayscale"})
    table = pd.read_csv(os.path.join(a, "table.csv"))
    assert {"other_A", "other_B"} <= set(table.columns)
    caption = open(os.path.join(a, "caption.txt")).read()
    assert "Unrelated curves (other A, other B)" in caption
    manifest = json.load(open(os.path.join(a, "bundle.json")))
    assert manifest["figure"]["style"]["distractors"] == 2 and manifest["figure"]["style"]["palette"] == "grayscale"
    script = open(os.path.join(a, "plot_figure.py")).read()
    assert '"annotations"' in script and script.count('"text":') == 3
    digest = lambda d: hashlib.sha256(open(os.path.join(d, "figure.png"), "rb").read()).hexdigest()
    assert digest(a) == digest(b)  # same seed, same clutter, same picture
    plain = _bundle(tmp_path / "c", {})
    assert digest(plain) != digest(a)


def test_cross_styles_keeps_data_and_questions_and_changes_the_picture(tmp_path):
    design = pd.DataFrame({"law": ["power_law", "hill"], "counterfactual": [False, True]})
    styles = {"plain": {}, "busy": {"style": "dense"}, "grey": {"palette": "grayscale", "log_y": True}}
    table = cross_styles(design, styles, base_seed=21)
    assert len(table) == 6 and table.groupby("design_point")["seed"].nunique().eq(1).all()
    assert set(table["bundle.figure.variant"]) == set(styles)
    run = run_design(BASE, table, output_dir=str(tmp_path))
    out = run.table
    groups = {}
    for row in out.to_dict("records"):
        bundle = glob.glob(os.path.join(str(tmp_path), f"*{row['experiment_id']}*", "bundle"))[0]
        qa = json.load(open(os.path.join(bundle, "qa.json")))
        picture = hashlib.sha256(open(os.path.join(bundle, "figure.png"), "rb").read()).hexdigest()
        style = json.load(open(os.path.join(bundle, "bundle.json")))["figure"]["style"]
        assert style["variant"] == row["bundle.figure.variant"]
        groups.setdefault(row["design_point"], []).append((json.dumps(qa, sort_keys=True), picture))
    for entries in groups.values():
        assert len({q for q, _ in entries}) == 1  # identical questions and answers
        assert len({p for _, p in entries}) == 3  # three different pictures
    with pytest.raises(ValueError):
        cross_styles(design, {"bad": {"colour": "red"}})


def test_style_gap_pairs_the_same_items_across_styles(tmp_path):
    from scidatasim.diagnostics import style_gap
    from scidatasim.export.tasks import collect_bundle_tasks
    design = pd.DataFrame({"law": ["power_law", "hill", "arrhenius", "logistic"], "counterfactual": [False] * 4})
    run_design(BASE, cross_styles(design, {"plain": {}, "busy": {"style": "dense"}}, replicates=2, base_seed=5),
               output_dir=str(tmp_path))
    tasks = collect_bundle_tasks(str(tmp_path))
    assert {t["figure_variant"] for t in tasks} == {"plain", "busy"}
    # a stand-in solver that misreads every busy chart and gets every plain one right
    correct = {t["id"]: t["figure_variant"] == "plain" for t in tasks}
    gap = style_gap(tasks, correct, base="plain", n_resamples=2000)
    busy = gap["busy"]
    assert busy["n_pairs"] == len(tasks) // 2 and busy["accuracy"] == 0.0 and busy["base_accuracy"] == 1.0
    assert busy["difference"] == -1.0 and busy["p_value"] < 0.05
    same = style_gap(tasks, {t["id"]: True for t in tasks}, base="plain", n_resamples=500)
    assert same["busy"]["difference"] == 0.0
    with pytest.raises(ValueError):
        style_gap(tasks, correct, base="nonexistent")
