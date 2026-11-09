# experiments/geometry/polygons.py
"""Plane polygons with exact coordinates: regular, random convex and random simple.

Configuration::

    kind: convex              # regular | convex | simple
    n_sides: 6                # 3..26 (vertices are labelled A, B, C, ...)
    radius: 5.0               # regular: circumradius; random kinds: radius of the drawing circle
    center: [0.0, 0.0]
    rotation_deg: 90.0        # regular only: polar angle of vertex A
    jitter: 0.6               # convex/simple: angular jitter of each vertex within its sector, in [0, 1)
    min_radius_fraction: 0.35 # simple only: vertex radii are drawn from [fraction * radius, radius]
    min_angle_deg: 15.0       # random kinds: reject polygons with an interior angle below this
                              #   (or a reflex angle above 360 - min_angle_deg)
    straight_margin_deg: 5.0  # random kinds: reject interior angles within this margin of 180 degrees
    vertices: null            # or [[x, y], ...]: a given simple polygon (overrides kind)
    show_centroid: false      # draw the area centroid in the diagram
    length_units: cm

Random polygons are drawn with ``self.rng``. ``convex``: one vertex per equal angular
sector of a circle (jittered), then a random axis stretch (factor 0.6-1) and
rotation - an affine image of a polygon inscribed in a circle, hence strictly convex.
``simple``: the same jittered angles with random radii, i.e. a polygon that is
star-shaped about its centre, hence simple and usually non-convex. Every polygon
(given or drawn) is checked to be simple (no two edges intersect).

Outputs: ``vertices`` (parameter, n x 2), ``side_lengths`` [AB, BC, ..., last-to-A] and
``interior_angles_deg`` [A, B, ...] (latent, 1-D), and labels ``area`` (shoelace),
``perimeter``, ``centroid`` (area centroid), ``angle_sum_deg``, ``convexity``
("convex" or "non-convex"), ``n_reflex`` (interior angles above 180 degrees),
``n_diagonals``, ``orientation`` ("counterclockwise"/"clockwise" vertex order) and
``longest_side``.
"""
from __future__ import annotations

import math
import string
from typing import Any, Dict, List, Optional

import numpy as np

from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType

KINDS = ("regular", "convex", "simple")
LETTERS = string.ascii_uppercase
MAX_SIDES = len(LETTERS)


def _cross(u, w) -> float:
    """z-component of the cross product of two plane vectors (np.cross is 3-D only in NumPy 2)."""
    return float(u[0] * w[1] - u[1] * w[0])


def _signed_area(v: np.ndarray) -> float:
    x, y = v[:, 0], v[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _segments_intersect(p1, p2, q1, q2, eps: float) -> bool:
    """Closed segments p1p2 and q1q2 intersect (touching counts)."""
    d1, d2 = _cross(p2 - p1, q1 - p1), _cross(p2 - p1, q2 - p1)
    d3, d4 = _cross(q2 - q1, p1 - q1), _cross(q2 - q1, p2 - q1)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
            ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
        return True

    def on_segment(a, b, p, d):
        return abs(d) <= eps and min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps \
            and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    return (on_segment(p1, p2, q1, d1) or on_segment(p1, p2, q2, d2)
            or on_segment(q1, q2, p1, d3) or on_segment(q1, q2, p2, d4))


def is_simple(v: np.ndarray) -> bool:
    """No repeated vertices, no zero-length sides and no two non-adjacent sides touching."""
    n = len(v)
    scale = float(np.max(np.ptp(v, axis=0))) or 1.0
    eps = 1e-12 * scale ** 2
    sides = np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1)
    if np.any(sides <= 1e-12 * scale):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # adjacent sides share a vertex
            if _segments_intersect(v[i], v[(i + 1) % n], v[j], v[(j + 1) % n], eps):
                return False
    # adjacent sides folding back onto each other (interior angle 0 or 360)
    for i in range(n):
        u, w = v[i] - v[i - 1], v[(i + 1) % n] - v[i]
        if abs(_cross(u, w)) <= eps and float(np.dot(u, w)) < 0:
            return False
    return True


class PolygonExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = ("Closed-form plane geometry (shoelace area, exact turning angles); checked against "
                     "independent identities (ear-clipping triangulation area and centroid, interior angle sum "
                     "(n-2)*180, regular-polygon closed forms, convexity of known shapes).")

    @classmethod
    def parameter_units(cls, config):
        """``radius`` is in the configured ``length_units`` (default cm)."""
        return {"radius": config.get("length_units", "cm")}

    def __init__(self, config: Dict[str, Any]):
        self.vertices_config = config.get("vertices")
        self.kind = str(config.get("kind", "convex"))
        self.n_sides = int(config.get("n_sides", 6))
        self.radius = float(config.get("radius", 5.0))
        self.center = np.asarray(config.get("center", [0.0, 0.0]), dtype=float)
        self.rotation = float(config.get("rotation_deg", 90.0))
        self.jitter = float(config.get("jitter", 0.6))
        self.min_radius_fraction = float(config.get("min_radius_fraction", 0.35))
        self.min_angle = float(config.get("min_angle_deg", 15.0))
        self.straight_margin = float(config.get("straight_margin_deg", 5.0))
        self.show_centroid = bool(config.get("show_centroid", False))
        self.units = config.get("length_units", "cm")
        if self.kind not in KINDS:
            raise ValueError(f"Unknown polygon kind {self.kind!r}; choose from {KINDS}")
        if self.vertices_config is None and not 3 <= self.n_sides <= MAX_SIDES:
            raise ValueError(f"n_sides must be in [3, {MAX_SIDES}]")
        if self.radius <= 0:
            raise ValueError("radius must be positive")
        if self.center.shape != (2,):
            raise ValueError("center must be an [x, y] pair")
        if not 0.0 <= self.jitter < 1.0:
            raise ValueError("jitter must be in [0, 1)")
        if not 0.0 < self.min_radius_fraction < 1.0:
            raise ValueError("min_radius_fraction must be in (0, 1)")
        if not 0.0 <= self.min_angle < 60.0:
            raise ValueError("min_angle_deg must be in [0, 60)")
        if not 0.0 <= self.straight_margin < 30.0:
            raise ValueError("straight_margin_deg must be in [0, 30)")
        self.vertices: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ construction
    def _regular(self) -> np.ndarray:
        n = self.n_sides
        theta = math.radians(self.rotation) + 2.0 * math.pi * np.arange(n) / n
        return self.center + self.radius * np.column_stack([np.cos(theta), np.sin(theta)])

    def _acceptable(self, v: np.ndarray) -> bool:
        angles = self.interior_angles(v)
        return (bool(np.all(angles >= self.min_angle)) and bool(np.all(angles <= 360.0 - self.min_angle))
                and bool(np.all(np.abs(angles - 180.0) >= self.straight_margin)) and is_simple(v))

    def _draw_vertices(self) -> np.ndarray:
        if self.vertices_config is not None:
            v = np.asarray(self.vertices_config, dtype=float)
            if v.ndim != 2 or v.shape[1] != 2 or not 3 <= len(v) <= MAX_SIDES:
                raise ValueError(f"vertices must be 3 to {MAX_SIDES} [x, y] pairs")
            if not is_simple(v):
                raise ValueError("vertices do not form a simple polygon (repeated vertices or crossing sides)")
            if abs(_signed_area(v)) <= 1e-12 * float(np.max(np.ptp(v, axis=0))) ** 2:
                raise ValueError("polygon has zero area")
            return v
        if self.kind == "regular":
            return self._regular()
        rng = self.get_rng()
        n = self.n_sides
        for _ in range(10000):
            offset = rng.uniform(0.0, 2.0 * math.pi)
            theta = offset + 2.0 * math.pi * (np.arange(n) + 0.5 + self.jitter * (rng.uniform(size=n) - 0.5)) / n
            if self.kind == "convex":
                stretch = rng.uniform(0.6, 1.0)
                phi = rng.uniform(0.0, 2.0 * math.pi)
                p = np.column_stack([stretch * np.cos(theta), np.sin(theta)]) * self.radius
                rot = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
                v = self.center + p @ rot.T
            else:
                r = self.radius * rng.uniform(self.min_radius_fraction, 1.0, size=n)
                v = self.center + np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            if self._acceptable(v):
                return v
        raise RuntimeError("could not draw a polygon satisfying min_angle_deg / straight_margin_deg")

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def side_labels(n: int) -> List[str]:
        return [LETTERS[i] + LETTERS[(i + 1) % n] for i in range(n)]

    @staticmethod
    def interior_angles(v: np.ndarray) -> np.ndarray:
        """Interior angle at each vertex (degrees), for either vertex orientation."""
        orient = 1.0 if _signed_area(v) > 0 else -1.0
        out = []
        n = len(v)
        for i in range(n):
            e1, e2 = v[i] - v[i - 1], v[(i + 1) % n] - v[i]
            turn = math.degrees(math.atan2(_cross(e1, e2), float(np.dot(e1, e2))))  # signed exterior angle
            out.append(180.0 - orient * turn)
        return np.array(out)

    def compute(self, v: np.ndarray) -> Dict[str, Any]:
        n = len(v)
        signed = _signed_area(v)
        area = abs(signed)
        sides = np.linalg.norm(np.roll(v, -1, axis=0) - v, axis=1)
        angles = self.interior_angles(v)
        x, y = v[:, 0], v[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        c = x * yn - xn * y
        centroid = np.array([np.sum((x + xn) * c), np.sum((y + yn) * c)]) / (6.0 * signed)
        n_reflex = int(np.sum(angles > 180.0))
        labels = self.side_labels(n)
        return dict(n=n, area=float(area), sides=sides, angles=angles, perimeter=float(sides.sum()),
                    centroid=centroid, angle_sum=float(angles.sum()), n_reflex=n_reflex,
                    convexity="convex" if n_reflex == 0 else "non-convex", n_diagonals=n * (n - 3) // 2,
                    orientation="counterclockwise" if signed > 0 else "clockwise",
                    longest_side=labels[int(np.argmax(sides))], side_labels=labels)

    # ------------------------------------------------------------------ ExperimentLogic
    def initialize(self, config):
        self.vertices = self._draw_vertices()
        return {}

    def run_step(self, state, step):
        return state

    def get_results(self):
        if self.vertices is None:
            self.vertices = self._draw_vertices()
        g = self.compute(self.vertices)
        u = self.units
        area_units = f"{u}**2" if u else None

        def d(name, dtype, role, units=None):
            return DataDescriptor(name, dtype, units=units, group="geometry", role=role)
        return {
            "vertices": {"data": self.vertices.copy(), "descriptor": d("vertices", DataType.NDARRAY, "parameter", u)},
            "side_lengths": {"data": g["sides"], "descriptor": d("side_lengths", DataType.NDARRAY, "latent", u)},
            "interior_angles_deg": {"data": g["angles"],
                                    "descriptor": d("interior_angles_deg", DataType.NDARRAY, "latent", "degree")},
            "area": {"data": g["area"], "descriptor": d("area", DataType.FLOAT, "label", area_units)},
            "perimeter": {"data": g["perimeter"], "descriptor": d("perimeter", DataType.FLOAT, "label", u)},
            "centroid": {"data": g["centroid"], "descriptor": d("centroid", DataType.NDARRAY, "label", u)},
            "angle_sum_deg": {"data": g["angle_sum"], "descriptor": d("angle_sum_deg", DataType.FLOAT, "label", "degree")},
            "convexity": {"data": g["convexity"], "descriptor": d("convexity", DataType.STRING, "label")},
            "n_reflex": {"data": g["n_reflex"], "descriptor": d("n_reflex", DataType.INT, "label")},
            "n_diagonals": {"data": g["n_diagonals"], "descriptor": d("n_diagonals", DataType.INT, "label")},
            "orientation": {"data": g["orientation"], "descriptor": d("orientation", DataType.STRING, "label")},
            "longest_side": {"data": g["longest_side"], "descriptor": d("longest_side", DataType.STRING, "label")},
        }

    # ------------------------------------------------------------------ bundle hooks
    def _name(self, n: int) -> str:
        return "polygon " + LETTERS[:n]

    def figure_elements(self, results):
        """Drawable elements of the diagram (all coordinates exact)."""
        v = results["vertices"]["data"]
        elements = [{"kind": "polygon", "label": self._name(len(v)), "points": v.tolist()}]
        for name, p in zip(LETTERS, v):
            elements.append({"kind": "point", "label": name, "points": [p.tolist()]})
        if self.show_centroid:
            elements.append({"kind": "point", "label": "centroid", "points": [self.compute(v)["centroid"].tolist()]})
        return elements

    def figure_caption(self, results):
        n = len(results["vertices"]["data"])
        extra = "; the area centroid is marked" if self.show_centroid else ""
        # Neutral wording: "convex"/"regular" would give away the convexity and angle answers.
        return (f"Diagram of a simple polygon {LETTERS[:n]} with {n} vertices, labelled in order along its "
                f"boundary{extra}; coordinates in {self.units}.")

    def questions(self, results, rng):
        """Exact questions about the polygon (ids are assigned by the bundle)."""
        v = results["vertices"]["data"]
        g = self.compute(v)
        n, u, name = g["n"], self.units, self._name(len(v))
        items = [
            dict(template="area", question=f"What is the area of {name} (in {u}^2)?", answer=g["area"],
                 answer_type="number", tolerance=0.03 * g["area"], tolerance_rule="3 % of the true value"),
            dict(template="perimeter", question=f"What is the perimeter of {name} (in {u})?",
                 answer=g["perimeter"], answer_type="number", tolerance=0.03 * g["perimeter"],
                 tolerance_rule="3 % of the true value"),
            dict(template="angle_sum", question=f"What is the sum of the interior angles of {name} (in degrees)?",
                 answer=int(round(g["angle_sum"])), answer_type="integer", tolerance=None, tolerance_rule="exact"),
            dict(template="n_diagonals", question=f"How many diagonals does {name} have?",
                 answer=g["n_diagonals"], answer_type="integer", tolerance=None, tolerance_rule="exact"),
        ]
        vertex = int(rng.integers(n))
        items.append(dict(template="angle", question=f"What is the interior angle of {name} at vertex "
                                                     f"{LETTERS[vertex]} (in degrees)?",
                          answer=float(g["angles"][vertex]), answer_type="number", tolerance=2.0,
                          tolerance_rule="2 degrees"))
        # Convexity is only asked when no angle is so close to 180 degrees that a drawing cannot settle it.
        if bool(np.all(np.abs(g["angles"] - 180.0) >= 3.0)):
            items.append(dict(template="convexity", question=f"Is {name} convex or non-convex?",
                              answer=g["convexity"], answer_type="choice", choices=["convex", "non-convex"],
                              tolerance=None, tolerance_rule="exact"))
            items.append(dict(template="n_reflex",
                              question=f"How many interior angles of {name} are reflex (greater than 180 degrees)?",
                              answer=g["n_reflex"], answer_type="integer", tolerance=None, tolerance_rule="exact"))
        ranked = np.sort(g["sides"])
        if ranked[-1] > 1.05 * ranked[-2]:
            labels = g["side_labels"]
            items.append(dict(template="longest_side",
                              question=f"Which side of {name} is the longest: {', '.join(labels)}?",
                              answer=g["longest_side"], answer_type="choice", choices=labels, tolerance=None,
                              tolerance_rule="exact"))
        for item in items:
            item.setdefault("series", None)
            item["answerable_from"] = "figure"
        return items
