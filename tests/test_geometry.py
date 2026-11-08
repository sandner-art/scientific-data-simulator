# tests/test_geometry.py
"""Verification of the triangle geometry model and its diagram bundles."""
import json
import math
import os
import runpy

import numpy as np
import pytest

from experiments.geometry.logic import TriangleExperiment
from simulator.engine import SimulatorEngine
from simulator.provenance import file_sha256

ALL = ["circumcircle", "incircle", "centroid", "orthocenter", "medians", "altitudes", "euler_line"]


def triangle(**kw):
    cfg = {"seed": 0, "constructions": ALL}
    cfg.update(kw)
    model = TriangleExperiment(cfg)
    model.rng = np.random.default_rng(cfg["seed"])
    model.initialize(cfg)
    return model, model.get_results()


@pytest.mark.parametrize("seed", range(20))
def test_independent_identities(seed):
    model, r = triangle(seed=seed)
    a, b, c = r["side_lengths"]["data"]
    A, B, C = np.radians(r["angles_deg"]["data"])
    s = (a + b + c) / 2
    R, rr, area = r["circumradius"]["data"], r["inradius"]["data"], r["area"]["data"]
    assert area == pytest.approx(math.sqrt(s * (s - a) * (s - b) * (s - c)), rel=1e-9)  # Heron vs shoelace
    assert a / math.sin(A) == pytest.approx(2 * R, rel=1e-9)                               # law of sines
    assert b / math.sin(B) == pytest.approx(2 * R, rel=1e-9) and c / math.sin(C) == pytest.approx(2 * R, rel=1e-9)
    assert np.degrees(A + B + C) == pytest.approx(180.0, abs=1e-9)
    O, I, G, H = (r[k]["data"] for k in ("circumcenter", "incenter", "centroid", "orthocenter"))
    assert np.dot(O - I, O - I) == pytest.approx(R * (R - 2 * rr), rel=1e-7, abs=1e-9)    # Euler's theorem
    np.testing.assert_allclose(G - O, (H - O) / 3, atol=1e-9)                              # Euler line OG:GH = 1:2
    v = r["vertices"]["data"]
    for p in v:
        assert np.linalg.norm(p - O) == pytest.approx(R, rel=1e-9)
    assert min(r["angles_deg"]["data"]) >= 15.0


@pytest.mark.parametrize("vertices, kind", [([[0, 0], [4, 0], [0, 3]], "right"),
                                            ([[0, 0], [4, 0], [2, 3]], "acute"),
                                            ([[0, 0], [6, 0], [1, 1]], "obtuse")])
def test_given_vertices_and_triangle_type(vertices, kind):
    _, r = triangle(vertices=vertices)
    assert r["triangle_type"]["data"] == kind
    if kind == "right":
        assert r["area"]["data"] == pytest.approx(6.0) and r["circumradius"]["data"] == pytest.approx(2.5)
        assert r["inradius"]["data"] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [dict(vertices=[[0, 0], [1, 1], [2, 2]]), dict(vertices=[[0, 0], [1, 0]]),
                                 dict(constructions=["nine_point_circle"]), dict(min_angle_deg=70)])
def test_invalid_configurations(bad):
    with pytest.raises(ValueError):
        triangle(**bad)


def _bundle_run(tmp_path, **kw):
    cfg = {"experiment_type": "experiments.geometry.logic.TriangleExperiment", "seed": 7, "constructions": ALL,
           "bundle": True, "generate_plots": False, **kw}
    return SimulatorEngine(str(tmp_path), verbose=False).run_config(cfg)


def test_diagram_bundle(tmp_path):
    run = _bundle_run(tmp_path)
    bdir = os.path.join(run.experiment_dir, "bundle")
    files = run.record.bundle["files"]
    assert {"figure.png", "plot_figure.py", "caption.txt", "questions.json"} <= {
        n for n, f in files.items() if f["visibility"] == "public"}
    assert files["elements.csv"]["visibility"] == "private"  # holds the circle radii verbatim
    with open(os.path.join(bdir, "qa.json")) as f:
        qa = {q["template"]: q for q in json.load(f)}
    assert qa["area"]["answer"] == pytest.approx(run.results["area"]["data"])
    assert qa["triangle_type"]["answer"] == run.results["triangle_type"]["data"]
    O, G = run.results["circumcenter"]["data"], run.results["centroid"]["data"]
    assert qa["euler_distance"]["answer"] == pytest.approx(float(np.linalg.norm(O - G)))
    with open(os.path.join(bdir, "caption.txt")) as f:
        caption = f.read()
    assert "Euler line" in caption and str(round(run.results["area"]["data"], 2)) not in caption
    regenerated = tmp_path / "again.png"
    runpy.run_path(os.path.join(bdir, "plot_figure.py"))["main"](str(regenerated))
    assert file_sha256(str(regenerated)) == files["figure.png"]["sha256"]


def test_diagram_bundle_is_deterministic(tmp_path):
    a, b = _bundle_run(tmp_path / "a"), _bundle_run(tmp_path / "b")
    for name in ("figure.png", "elements.csv", "qa.json", "plot_figure.py"):
        assert a.record.bundle["files"][name]["sha256"] == b.record.bundle["files"][name]["sha256"]
    c = _bundle_run(tmp_path / "c", seed=8)
    assert c.record.bundle["files"]["figure.png"]["sha256"] != a.record.bundle["files"]["figure.png"]["sha256"]


def test_noisy_side_measurements(tmp_path):
    run = _bundle_run(tmp_path, observation={"targets": ["side_lengths"],
                                             "operators": [{"type": "gaussian", "sigma": 0.05}]})
    observed, exact = run.results["side_lengths_observed"]["data"], run.results["side_lengths"]["data"]
    assert observed.shape == (3,) and not np.array_equal(observed, exact)
    assert run.record.ground_truth["observation_fields"] == ["side_lengths_observed"]


def test_bare_diagram_without_axes(tmp_path):
    run = _bundle_run(tmp_path, bundle={"figure": {"axes": False, "format": "svg"}})
    assert os.path.exists(os.path.join(run.experiment_dir, "bundle", "figure.svg"))
