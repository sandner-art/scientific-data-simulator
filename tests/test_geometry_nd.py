# tests/test_geometry_nd.py
"""Verification of the polygon and n-dimensional simplex models and their diagram bundles.

Every check compares the model with an independent route to the same quantity:
ear-clipping triangulation (polygon area and centroid), the turning-angle theorem,
regular-polygon and regular-simplex closed forms, the Cayley-Menger determinant
(simplex volume, facet volumes and circumradius), facet-hyperplane distances
(inradius) and the triangle model (d = 2).
"""
from __future__ import annotations

import json
import math
import os
import runpy

import numpy as np
import pytest

from experiments.geometry.logic import TriangleExperiment
from experiments.geometry.polygons import PolygonExperiment
from experiments.geometry.simplex import SimplexExperiment, edge_pairs
from simulator.checker import check_model
from simulator.engine import SimulatorEngine
from simulator.provenance import file_sha256


def build(cls, **cfg):
    cfg.setdefault("seed", 0)
    model = cls(cfg)
    model.rng = np.random.default_rng(cfg["seed"])
    model.initialize(cfg)
    return model, model.get_results()


def data(results, name):
    return results[name]["data"]


# ============================================================================ polygons
def _cross(u, w):
    return float(u[0] * w[1] - u[1] * w[0])


def _in_triangle(p, a, b, c):
    """p strictly inside or on the boundary of the counter-clockwise triangle abc."""
    return _cross(b - a, p - a) >= -1e-12 and _cross(c - b, p - b) >= -1e-12 and _cross(a - c, p - c) >= -1e-12


def ear_clip(v):
    """Triangulate a simple polygon by ear clipping; returns positive-area triangles."""
    pts = [np.asarray(p, dtype=float) for p in v]
    x, y = np.array(v)[:, 0], np.array(v)[:, 1]
    if np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        pts = pts[::-1]  # make counter-clockwise
    idx = list(range(len(pts)))
    triangles = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i, j, l = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i], pts[j], pts[l]
            if _cross(b - a, c - b) <= 0:
                continue  # reflex corner: not an ear
            if any(_in_triangle(pts[m], a, b, c) for m in idx if m not in (i, j, l)):
                continue
            triangles.append((a, b, c))
            idx.pop(k)
            break
        else:  # pragma: no cover - a simple polygon always has an ear
            raise AssertionError("no ear found")
    triangles.append(tuple(pts[m] for m in idx))
    return triangles


POLYGON_CASES = [(kind, seed) for kind in ("regular", "convex", "simple") for seed in range(8)]


@pytest.mark.parametrize("kind, seed", POLYGON_CASES)
def test_polygon_area_centroid_and_angles(kind, seed):
    n = 3 + seed  # 3..10 sides
    _, r = build(PolygonExperiment, kind=kind, n_sides=n, seed=seed)
    v = data(r, "vertices")
    tris = ear_clip(v)
    assert len(tris) == n - 2
    areas = np.array([0.5 * _cross(b - a, c - a) for a, b, c in tris])
    assert np.all(areas > 0)
    assert data(r, "area") == pytest.approx(areas.sum(), rel=1e-10)
    centroid = sum(w * (a + b + c) / 3 for w, (a, b, c) in zip(areas, tris)) / areas.sum()
    np.testing.assert_allclose(data(r, "centroid"), centroid, atol=1e-9)
    # side lengths, perimeter
    sides = [math.hypot(*(v[(i + 1) % n] - v[i])) for i in range(n)]
    np.testing.assert_allclose(data(r, "side_lengths"), sides, rtol=1e-12)
    assert data(r, "perimeter") == pytest.approx(sum(sides))
    # interior angles: sum (n-2)*180, each from the law of cosines at the corner
    angles = data(r, "interior_angles_deg")
    assert angles.sum() == pytest.approx((n - 2) * 180.0, abs=1e-9)
    assert data(r, "angle_sum_deg") == pytest.approx((n - 2) * 180.0, abs=1e-9)
    for i in range(n):
        p, q = v[i - 1] - v[i], v[(i + 1) % n] - v[i]
        inner = math.degrees(math.acos(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))))
        reflex = _cross(v[i] - v[i - 1], v[(i + 1) % n] - v[i]) < 0  # right turn in a CCW polygon
        assert angles[i] == pytest.approx(360.0 - inner if reflex else inner, abs=1e-9)
    # convexity: all turns to the same side
    turns = [_cross(v[i] - v[i - 1], v[(i + 1) % n] - v[i]) for i in range(n)]
    assert data(r, "n_reflex") == sum(t < 0 for t in turns)
    assert data(r, "convexity") == ("convex" if min(turns) > 0 else "non-convex")
    assert data(r, "orientation") == "counterclockwise"
    assert data(r, "n_diagonals") == n * (n - 3) // 2
    if kind in ("regular", "convex"):
        assert data(r, "convexity") == "convex"
    if kind != "regular":
        assert np.all(np.abs(angles - 180.0) >= 5.0) and angles.min() >= 15.0


@pytest.mark.parametrize("n", [3, 4, 5, 6, 12, 26])
def test_regular_polygon_closed_forms(n):
    R = 2.5
    _, r = build(PolygonExperiment, kind="regular", n_sides=n, radius=R, center=[1.0, -2.0], rotation_deg=17.0)
    assert data(r, "area") == pytest.approx(0.5 * n * R ** 2 * math.sin(2 * math.pi / n), rel=1e-12)
    assert data(r, "perimeter") == pytest.approx(2 * n * R * math.sin(math.pi / n), rel=1e-12)
    np.testing.assert_allclose(data(r, "interior_angles_deg"), (n - 2) * 180.0 / n, atol=1e-9)
    np.testing.assert_allclose(data(r, "centroid"), [1.0, -2.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(data(r, "vertices") - [1.0, -2.0], axis=1), R, rtol=1e-12)


@pytest.mark.parametrize("vertices, convexity, n_reflex, area, orientation", [
    ([[0, 0], [2, 0], [2, 2], [0, 2]], "convex", 0, 4.0, "counterclockwise"),
    ([[0, 0], [0, 2], [2, 2], [2, 0]], "convex", 0, 4.0, "clockwise"),
    ([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], "non-convex", 1, 3.0, "counterclockwise"),  # L-shape
    ([[0, 0], [2, 1], [4, 0], [2, 3]], "non-convex", 1, 4.0, "counterclockwise"),                  # dart
    ([[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3]], "non-convex", 2, 7.0, "counterclockwise"),  # U
])
def test_known_polygons(vertices, convexity, n_reflex, area, orientation):
    _, r = build(PolygonExperiment, vertices=vertices)
    assert data(r, "convexity") == convexity and data(r, "n_reflex") == n_reflex
    assert data(r, "area") == pytest.approx(area) and data(r, "orientation") == orientation
    n = len(vertices)
    assert data(r, "interior_angles_deg").sum() == pytest.approx((n - 2) * 180.0)
    if n == 4 and convexity == "convex":
        np.testing.assert_allclose(data(r, "interior_angles_deg"), 90.0)
        np.testing.assert_allclose(data(r, "centroid"), [1.0, 1.0])


@pytest.mark.parametrize("bad", [
    dict(vertices=[[0, 0], [2, 2], [2, 0], [0, 2]]),          # bow-tie
    dict(vertices=[[0, 0], [1, 0], [1, 0], [0, 1]]),          # repeated vertex
    dict(vertices=[[0, 0], [1, 0], [2, 0]]),                  # collinear
    dict(vertices=[[0, 0], [2, 0], [1, 0], [1, 1]]),          # side folds back
    dict(vertices=[[0, 0], [1, 0]]),
    dict(kind="star"), dict(n_sides=2), dict(n_sides=27), dict(radius=0), dict(jitter=1.0),
    dict(min_radius_fraction=1.0),
])
def test_polygon_invalid_configurations(bad):
    with pytest.raises(ValueError):
        build(PolygonExperiment, **bad)


def test_polygon_reproducibility():
    for kind in ("convex", "simple"):
        a, b = build(PolygonExperiment, kind=kind, seed=5)[1], build(PolygonExperiment, kind=kind, seed=5)[1]
        c = build(PolygonExperiment, kind=kind, seed=6)[1]
        assert np.array_equal(data(a, "vertices"), data(b, "vertices"))
        assert not np.array_equal(data(a, "vertices"), data(c, "vertices"))


def test_polygon_questions():
    model, r = build(PolygonExperiment, kind="simple", n_sides=7, seed=3)
    qa = {q["template"]: q for q in model.questions(r, np.random.default_rng(0))}
    assert qa["angle_sum"]["answer"] == 900 and qa["n_diagonals"]["answer"] == 14
    assert qa["convexity"]["answer"] == data(r, "convexity") and qa["n_reflex"]["answer"] == data(r, "n_reflex")
    assert all(q["answerable_from"] == "figure" for q in qa.values())
    _, reg = build(PolygonExperiment, kind="regular", n_sides=5)
    assert "longest_side" not in {q["template"] for q in model.questions(reg, np.random.default_rng(0))}


# ============================================================================ simplices
def cayley_menger(points):
    """Squared-distance matrix D and bordered Cayley-Menger determinant of the points."""
    m = len(points)
    D = np.array([[float(np.sum((p - q) ** 2)) for q in points] for p in points])
    cm = np.ones((m + 1, m + 1))
    cm[0, 0] = 0.0
    cm[1:, 1:] = D
    return D, float(np.linalg.det(cm))


def cm_volume(points):
    k = len(points) - 1
    _, det = cayley_menger(points)
    return math.sqrt((-1) ** (k + 1) * det / (2 ** k * math.factorial(k) ** 2))


SIMPLEX_CASES = [(d, seed) for d in range(2, 7) for seed in range(5)]


@pytest.mark.parametrize("d, seed", SIMPLEX_CASES)
def test_simplex_independent_identities(d, seed):
    _, r = build(SimplexExperiment, dims=d, seed=seed)
    v = data(r, "vertices")
    V = data(r, "volume")
    assert V == pytest.approx(cm_volume(v), rel=1e-8)                           # det vs Cayley-Menger
    # circumcentre equidistant; circumradius from R^2 = -det(D) / (2 det(CM))
    O, R = data(r, "circumcenter"), data(r, "circumradius")
    np.testing.assert_allclose(np.linalg.norm(v - O, axis=1), R, rtol=1e-9)
    D, det_cm = cayley_menger(v)
    assert R ** 2 == pytest.approx(-np.linalg.det(D) / (2 * det_cm), rel=1e-7)
    # facet volumes (Gram) vs Cayley-Menger; incentre at distance r from every facet hyperplane
    facets = data(r, "facet_volumes")
    I, rr = data(r, "incenter"), data(r, "inradius")
    for i in range(d + 1):
        F = np.delete(v, i, axis=0)
        assert facets[i] == pytest.approx(cm_volume(F), rel=1e-7)
        normal = np.linalg.svd(F[1:] - F[0])[2][-1]
        assert abs(float(np.dot(normal, I - F[0]))) == pytest.approx(rr, rel=1e-8)
        # incentre lies on the same side of facet i as vertex i
        assert np.dot(normal, I - F[0]) * np.dot(normal, v[i] - F[0]) > 0
    assert rr == pytest.approx(d * V / facets.sum())
    # edges in lexicographic pair order
    pairs = edge_pairs(d)
    assert pairs[0] == (0, 1) and pairs[-1] == (d - 1, d) and len(pairs) == math.comb(d + 1, 2)
    np.testing.assert_allclose(data(r, "edge_lengths"), [np.linalg.norm(v[i] - v[j]) for i, j in pairs])
    # f-vector: binomials, Euler characteristic of the boundary sphere, total count
    f = data(r, "f_vector")
    assert list(f) == [math.comb(d + 1, k + 1) for k in range(d)] and data(r, "n_edges") == f[1]
    assert sum((-1) ** k * f[k] for k in range(d)) == 1 - (-1) ** d
    assert f.sum() == 2 ** (d + 1) - 2
    # rounded coordinates (quoted verbatim in the questions) and a well-shaped simplex
    np.testing.assert_array_equal(v, np.round(v, 2))
    assert SimplexExperiment.quality(v) >= 0.2
    # projection: orthonormal pair; projected vertices consistent
    P = data(r, "projection")
    np.testing.assert_allclose(P.T @ P, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(data(r, "projected_vertices"), v @ P)


@pytest.mark.parametrize("d", range(2, 7))
def test_regular_simplex_closed_forms(d):
    a = 3.0
    _, r = build(SimplexExperiment, dims=d, regular=True, edge_length=a)
    np.testing.assert_allclose(data(r, "edge_lengths"), a, rtol=1e-12)
    assert data(r, "volume") == pytest.approx(a ** d / math.factorial(d) * math.sqrt((d + 1) / 2 ** d), rel=1e-12)
    assert data(r, "circumradius") == pytest.approx(a * math.sqrt(d / (2 * (d + 1))), rel=1e-12)
    assert data(r, "inradius") == pytest.approx(a / math.sqrt(2 * d * (d + 1)), rel=1e-12)
    for name in ("circumcenter", "incenter", "centroid"):
        np.testing.assert_allclose(data(r, name), 0.0, atol=1e-12)
    assert SimplexExperiment.quality(data(r, "vertices")) == pytest.approx(1.0)


def test_tetrahedron_triple_product():
    _, r = build(SimplexExperiment, dims=3, seed=11)
    v = data(r, "vertices")
    a, b, c = v[1] - v[0], v[2] - v[0], v[3] - v[0]
    assert data(r, "volume") == pytest.approx(abs(float(np.dot(a, np.cross(b, c)))) / 6.0, rel=1e-12)
    assert len(data(r, "edge_lengths")) == 6 and list(data(r, "f_vector")) == [4, 6, 4]


@pytest.mark.parametrize("seed", range(5))
def test_simplex_d2_agrees_with_triangle_model(seed):
    _, s = build(SimplexExperiment, dims=2, seed=seed)
    v = data(s, "vertices")
    _, t = build(TriangleExperiment, vertices=v.tolist())
    assert data(s, "volume") == pytest.approx(data(t, "area"), rel=1e-12)
    for name in ("circumradius", "inradius"):
        assert data(s, name) == pytest.approx(data(t, name), rel=1e-10)
    for name in ("circumcenter", "incenter", "centroid"):
        np.testing.assert_allclose(data(s, name), data(t, name), atol=1e-10)
    a, b, c = data(t, "side_lengths")  # a = BC = P1P2, b = CA = P0P2, c = AB = P0P1
    np.testing.assert_allclose(data(s, "edge_lengths"), [c, b, a], rtol=1e-12)
    np.testing.assert_allclose(data(s, "facet_volumes"), [a, b, c], rtol=1e-12)
    np.testing.assert_array_equal(data(s, "projection"), np.eye(2))


def test_simplex_given_vertices_and_coordinate_projection():
    unit = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    model, r = build(SimplexExperiment, vertices=unit, projection="coordinates")
    assert model.dims == 4 and data(r, "volume") == pytest.approx(1 / 24)
    assert data(r, "circumradius") == pytest.approx(1.0)  # circumcentre (1/2, 1/2, 1/2, 1/2)
    assert data(r, "inradius") == pytest.approx(1 / (4 + 2.0))  # r = 1 / (d + sqrt(d)) for the corner simplex
    np.testing.assert_allclose(data(r, "projected_vertices"), np.array(unit, dtype=float)[:, :2])


@pytest.mark.parametrize("bad", [dict(dims=1), dict(dims=7), dict(vertices=[[0, 0], [1, 0], [2, 0]]),
                                 dict(vertices=[[0, 0], [1, 0]]), dict(vertices=[[0, 0], [1, 0], [0, 1]], dims=3),
                                 dict(projection="perspective"), dict(edge_length=-1.0, regular=True),
                                 dict(min_quality=1.0)])
def test_simplex_invalid_configurations(bad):
    with pytest.raises(ValueError):
        build(SimplexExperiment, **bad)


def test_simplex_reproducibility():
    a, b = build(SimplexExperiment, dims=5, seed=4)[1], build(SimplexExperiment, dims=5, seed=4)[1]
    c = build(SimplexExperiment, dims=5, seed=9)[1]
    for name in ("vertices", "projection"):
        assert np.array_equal(data(a, name), data(b, name))
        assert not np.array_equal(data(a, name), data(c, name))
    r1, r2 = build(SimplexExperiment, dims=4, regular=True, seed=1)[1], build(SimplexExperiment, dims=4, regular=True, seed=2)[1]
    assert np.array_equal(data(r1, "vertices"), data(r2, "vertices"))  # regular coordinates are deterministic


# ============================================================================ bundles and checker
def _bundle_run(tmp_path, experiment_type, **kw):
    cfg = {"experiment_type": experiment_type, "seed": 7, "bundle": True, "generate_plots": False, **kw}
    return SimulatorEngine(str(tmp_path), verbose=False).run_config(cfg)


def _check_bundle(tmp_path, run):
    bdir = os.path.join(run.experiment_dir, "bundle")
    files = run.record.bundle["files"]
    assert {"figure.png", "plot_figure.py", "caption.txt", "questions.json"} <= {
        n for n, f in files.items() if f["visibility"] == "public"}
    assert files["elements.csv"]["visibility"] == "private"
    regenerated = tmp_path / "again.png"
    runpy.run_path(os.path.join(bdir, "plot_figure.py"))["main"](str(regenerated))
    assert file_sha256(str(regenerated)) == files["figure.png"]["sha256"]
    with open(os.path.join(bdir, "qa.json")) as f:
        qa = {q["template"]: q for q in json.load(f)}
    with open(os.path.join(bdir, "questions.json")) as f:
        assert all("answer" not in q and "tolerance" not in q for q in json.load(f))
    with open(os.path.join(bdir, "caption.txt")) as f:
        caption = f.read()
    return qa, caption


def test_polygon_bundle(tmp_path):
    run = _bundle_run(tmp_path, "experiments.geometry.polygons.PolygonExperiment", kind="simple", n_sides=8)
    qa, caption = _check_bundle(tmp_path, run)
    res = run.results
    assert qa["area"]["answer"] == pytest.approx(res["area"]["data"])
    assert qa["perimeter"]["answer"] == pytest.approx(res["perimeter"]["data"])
    assert qa["angle_sum"]["answer"] == 1080
    assert qa["convexity"]["answer"] == res["convexity"]["data"]
    assert qa["n_reflex"]["answer"] == res["n_reflex"]["data"]
    vertex = "ABCDEFGH".index(qa["angle"]["question"].split("vertex ")[1][0])
    assert qa["angle"]["answer"] == pytest.approx(res["interior_angles_deg"]["data"][vertex])
    assert "polygon ABCDEFGH" in caption and "convex" not in caption
    assert str(round(res["area"]["data"], 2)) not in caption


def test_simplex_bundle(tmp_path):
    run = _bundle_run(tmp_path, "experiments.geometry.simplex.SimplexExperiment", dims=4)
    qa, caption = _check_bundle(tmp_path, run)
    res = run.results
    assert "projection" in caption and "4-dimensional simplex" in caption
    assert qa["n_edges"]["answer"] == 10 and qa["n_edges"]["answerable_from"] == "figure"
    assert qa["volume"]["answer"] == pytest.approx(res["volume"]["data"])
    assert qa["circumradius"]["answer"] == pytest.approx(res["circumradius"]["data"])
    assert qa["inradius"]["answer"] == pytest.approx(res["inradius"]["data"])
    k = int(qa["k_faces"]["question"].split("How many ")[1].split("-")[0])
    assert k in (2, 3) and qa["k_faces"]["answer"] == res["f_vector"]["data"][k] == math.comb(5, k + 1)
    for name in ("volume", "circumradius", "inradius", "k_faces"):
        assert qa[name]["answerable_from"] == "data"
    v = res["vertices"]["data"]
    for p in v:  # the data questions quote the exact (rounded) coordinates
        assert "(" + ", ".join(f"{x:.10g}" for x in p) + ")" in qa["volume"]["question"]
    if "longest_edge" in qa:
        assert qa["longest_edge"]["answer"] == res["longest_edge"]["data"]
    with open(os.path.join(run.experiment_dir, "bundle", "elements.csv")) as f:
        kinds = [line.split(",")[0] for line in f.read().splitlines()[1:]]
    assert kinds.count("segment") == 10 and kinds.count("point") == 5


def test_simplex_d2_bundle_is_planar(tmp_path):
    run = _bundle_run(tmp_path, "experiments.geometry.simplex.SimplexExperiment", dims=2, bundle={"qa": {"enabled": True}})
    with open(os.path.join(run.experiment_dir, "bundle", "qa.json")) as f:
        qa = json.load(f)
    assert {q["answerable_from"] for q in qa if q["template"] in ("volume", "circumradius")} == {"figure"}


@pytest.mark.parametrize("model, config", [
    ("experiments.geometry.polygons.PolygonExperiment", {"kind": "simple", "n_sides": 7}),
    ("experiments.geometry.polygons.PolygonExperiment", {"kind": "regular", "n_sides": 5}),
    ("experiments.geometry.simplex.SimplexExperiment", {"dims": 4}),
])
def test_models_pass_checker(model, config):
    report = check_model(model, config)
    assert report.accepted, report.feedback()
    assert report.get("lint").status == "pass" and report.get("determinism").status == "pass"
