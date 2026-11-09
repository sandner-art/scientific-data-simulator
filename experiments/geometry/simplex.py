# experiments/geometry/simplex.py
"""An n-dimensional simplex (d = 2..6) with exact coordinates and measures.

Configuration::

    dims: 3                  # d, 2..6 (inferred from ``vertices`` when they are given)
    vertices: null           # or d + 1 points in R^d
    regular: false           # true: regular simplex with edge length ``edge_length``,
                             #   centred at the origin (deterministic coordinates)
    edge_length: 5.0         # regular only
    extent: 10.0             # random: vertices uniform in [0, extent]^d ...
    decimals: 2              # ... rounded to this many decimals (null: no rounding), so the
                             #   coordinates quoted in questions are the exact inputs
    min_quality: 0.2         # random: reject draws with (V / V_regular(longest edge))^(1/d) below this
    projection: random       # random | coordinates  (2-D diagram of a d > 2 simplex)
    length_units: cm

Vertices are labelled P0..Pd. Edges are listed in lexicographic vertex-pair order
P0P1, P0P2, ..., P0Pd, P1P2, ..., P(d-1)Pd (``edge_pairs``); facet i is the facet
opposite vertex Pi.

Outputs: ``vertices`` (parameter, (d+1) x d), ``edge_lengths`` (latent, 1-D),
labels ``volume`` (|det(P1-P0, ..., Pd-P0)| / d!), ``facet_volumes`` ((d-1)-volumes from
the Gram determinant), ``circumcenter``/``circumradius`` (linear solve),
``incenter``/``inradius`` (r = d V / sum of facet volumes), ``centroid``,
``f_vector`` (number of k-faces C(d+1, k+1) for k = 0..d-1), ``n_edges``,
``longest_edge``; and auxiliary ``projection`` (d x 2 orthonormal basis) and
``projected_vertices`` (the diagram coordinates).

Diagram: for d = 2 the triangle itself; for d > 2 an orthographic projection onto a
plane spanned by a seeded random orthonormal pair (``projection: random``, drawn with
``self.rng`` after the vertices) or onto the first two coordinate axes
(``projection: coordinates``), with every edge drawn as a segment, the outline of the
projection as a polygon and labelled vertices. Lengths and angles in such a figure
are not true lengths, so the metric questions of a d > 2 simplex are marked
``answerable_from: "data"`` and quote the vertex coordinates (or the edge length of a
regular simplex) in the question text.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType

MIN_DIMS, MAX_DIMS = 2, 6
PROJECTIONS = ("random", "coordinates")


def _cross(u, w) -> float:
    """z-component of the cross product of two plane vectors (np.cross is 3-D only in NumPy 2)."""
    return float(u[0] * w[1] - u[1] * w[0])


def edge_pairs(d: int) -> List[Tuple[int, int]]:
    """Vertex pairs of the edges, in the documented (lexicographic) order."""
    return list(itertools.combinations(range(d + 1), 2))


def regular_simplex(d: int, a: float) -> np.ndarray:
    """Regular d-simplex with edge length ``a``, centred at the origin.

    The vertices a/sqrt(2) * e_i (i = 0..d) of R^(d+1) are centred and expressed in the
    orthonormal Helmert basis of the hyperplane sum(x) = 0.
    """
    points = np.eye(d + 1) * (a / math.sqrt(2.0))
    points = points - points.mean(axis=0)
    basis = np.zeros((d, d + 1))
    for k in range(1, d + 1):
        basis[k - 1, :k] = 1.0
        basis[k - 1, k] = -float(k)
        basis[k - 1] /= math.sqrt(k * (k + 1.0))
    return points @ basis.T


def _convex_hull(points: np.ndarray) -> List[int]:
    """Indices of the 2-D convex hull, counter-clockwise (Andrew's monotone chain)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    scale = float(np.max(np.ptp(points, axis=0))) or 1.0
    eps = 1e-12 * scale ** 2

    def chain(indices):
        out: List[int] = []
        for i in indices:
            while len(out) >= 2 and _cross(points[out[-1]] - points[out[-2]], points[i] - points[out[-2]]) <= eps:
                out.pop()
            out.append(i)
        return out
    lower, upper = chain(order), chain(order[::-1])
    return lower[:-1] + upper[:-1]


class SimplexExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = ("Closed-form n-dimensional Euclidean geometry (determinants, Gram determinants, a linear "
                     "solve for the circumcentre); checked against the Cayley-Menger determinant, equidistance "
                     "of the circumcentre, facet distances of the incentre, regular-simplex closed forms and the "
                     "triangle model for d = 2.")

    @classmethod
    def parameter_units(cls, config):
        """``edge_length`` and ``extent`` are in the configured ``length_units`` (default cm)."""
        u = config.get("length_units", "cm")
        return {"edge_length": u, "extent": u}

    def __init__(self, config: Dict[str, Any]):
        self.vertices_config = config.get("vertices")
        self.regular = bool(config.get("regular", False))
        self.edge_length = float(config.get("edge_length", 5.0))
        self.extent = float(config.get("extent", 10.0))
        decimals = config.get("decimals", 2)
        self.decimals = None if decimals is None else int(decimals)
        self.min_quality = float(config.get("min_quality", 0.2))
        self.projection_kind = str(config.get("projection", "random"))
        self.units = config.get("length_units", "cm")
        if self.vertices_config is not None:
            v = np.asarray(self.vertices_config, dtype=float)
            if v.ndim != 2 or v.shape[0] != v.shape[1] + 1:
                raise ValueError("vertices must be d + 1 points in R^d")
            d = v.shape[1]
            if "dims" in config and int(config["dims"]) != d:
                raise ValueError(f"dims={config['dims']} does not match the {d}-dimensional vertices")
            if self.regular:
                raise ValueError("give either vertices or regular: true, not both")
        else:
            d = int(config.get("dims", 3))
        self.dims = d
        if not MIN_DIMS <= d <= MAX_DIMS:
            raise ValueError(f"dims must be in [{MIN_DIMS}, {MAX_DIMS}]")
        if self.edge_length <= 0 or self.extent <= 0:
            raise ValueError("edge_length and extent must be positive")
        if not 0.0 <= self.min_quality < 1.0:
            raise ValueError("min_quality must be in [0, 1)")
        if self.decimals is not None and self.decimals < 0:
            raise ValueError("decimals must be >= 0 or null")
        if self.projection_kind not in PROJECTIONS:
            raise ValueError(f"Unknown projection {self.projection_kind!r}; choose from {PROJECTIONS}")
        self.vertices: Optional[np.ndarray] = None
        self.projection: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ construction
    @staticmethod
    def quality(v: np.ndarray) -> float:
        """(V / V_regular(longest edge))^(1/d): 1 for a regular simplex, 0 when degenerate."""
        d = v.shape[1]
        volume = abs(float(np.linalg.det(v[1:] - v[0]))) / math.factorial(d)
        longest = max(float(np.linalg.norm(v[i] - v[j])) for i, j in edge_pairs(d))
        regular = longest ** d / math.factorial(d) * math.sqrt((d + 1) / 2.0 ** d)
        return (volume / regular) ** (1.0 / d)

    def _draw_vertices(self) -> np.ndarray:
        d = self.dims
        if self.vertices_config is not None:
            v = np.asarray(self.vertices_config, dtype=float)
            if self.quality(v) < 1e-6:
                raise ValueError("vertices are affinely dependent (degenerate simplex)")
            return v
        if self.regular:
            return regular_simplex(d, self.edge_length)
        rng = self.get_rng()
        for _ in range(10000):
            v = rng.uniform(0.0, self.extent, size=(d + 1, d))
            if self.decimals is not None:
                v = np.round(v, self.decimals)
            if self.quality(v) >= max(self.min_quality, 1e-6):
                return v
        raise RuntimeError("could not draw a simplex satisfying min_quality")

    def _draw_projection(self) -> np.ndarray:
        d = self.dims
        if d == 2 or self.projection_kind == "coordinates":
            return np.eye(d)[:, :2].copy()
        q, r = np.linalg.qr(self.get_rng().normal(size=(d, 2)))
        return q * np.where(np.diag(r) < 0, -1.0, 1.0)  # unique sign convention

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def labels(d: int) -> List[str]:
        return [f"P{i}" for i in range(d + 1)]

    @classmethod
    def edge_labels(cls, d: int) -> List[str]:
        names = cls.labels(d)
        return [names[i] + names[j] for i, j in edge_pairs(d)]

    @staticmethod
    def facet_volume(points: np.ndarray) -> float:
        """(k-1)-volume of the simplex spanned by k points in R^d (Gram determinant)."""
        k = len(points) - 1
        e = points[1:] - points[0]
        gram = e @ e.T
        return math.sqrt(max(float(np.linalg.det(gram)), 0.0)) / math.factorial(k)

    def compute(self, v: np.ndarray) -> Dict[str, Any]:
        d = v.shape[1]
        pairs = edge_pairs(d)
        edges = np.array([np.linalg.norm(v[i] - v[j]) for i, j in pairs])
        volume = abs(float(np.linalg.det(v[1:] - v[0]))) / math.factorial(d)
        facets = np.array([self.facet_volume(np.delete(v, i, axis=0)) for i in range(d + 1)])
        # circumcentre: 2 (Pi - P0) . c = |Pi|^2 - |P0|^2
        a = 2.0 * (v[1:] - v[0])
        b = np.sum(v[1:] ** 2, axis=1) - float(np.sum(v[0] ** 2))
        circumcenter = np.linalg.solve(a, b)
        circumradius = float(np.linalg.norm(v[0] - circumcenter))
        total = float(facets.sum())
        inradius = d * volume / total
        incenter = facets @ v / total  # barycentric weights proportional to the opposite facet volumes
        f_vector = np.array([math.comb(d + 1, k + 1) for k in range(d)], dtype=int)
        labels = self.edge_labels(d)
        return dict(d=d, edges=edges, volume=volume, facets=facets, circumcenter=circumcenter,
                    circumradius=circumradius, incenter=incenter, inradius=float(inradius),
                    centroid=v.mean(axis=0), f_vector=f_vector, n_edges=len(pairs),
                    longest_edge=labels[int(np.argmax(edges))], edge_labels=labels)

    # ------------------------------------------------------------------ ExperimentLogic
    def initialize(self, config):
        self.vertices = self._draw_vertices()
        self.projection = self._draw_projection()
        return {}

    def run_step(self, state, step):
        return state

    def get_results(self):
        if self.vertices is None:
            self.initialize({})
        g = self.compute(self.vertices)
        d, u = self.dims, self.units

        def power(k):
            return None if not u else (u if k == 1 else f"{u}**{k}")

        def desc(name, dtype, role, units=None):
            return DataDescriptor(name, dtype, units=units, group="geometry", role=role)
        return {
            "vertices": {"data": self.vertices.copy(), "descriptor": desc("vertices", DataType.NDARRAY, "parameter", u)},
            "edge_lengths": {"data": g["edges"], "descriptor": desc("edge_lengths", DataType.NDARRAY, "latent", u)},
            "volume": {"data": g["volume"], "descriptor": desc("volume", DataType.FLOAT, "label", power(d))},
            "facet_volumes": {"data": g["facets"],
                              "descriptor": desc("facet_volumes", DataType.NDARRAY, "label", power(d - 1))},
            "circumradius": {"data": g["circumradius"], "descriptor": desc("circumradius", DataType.FLOAT, "label", u)},
            "circumcenter": {"data": g["circumcenter"],
                             "descriptor": desc("circumcenter", DataType.NDARRAY, "label", u)},
            "inradius": {"data": g["inradius"], "descriptor": desc("inradius", DataType.FLOAT, "label", u)},
            "incenter": {"data": g["incenter"], "descriptor": desc("incenter", DataType.NDARRAY, "label", u)},
            "centroid": {"data": g["centroid"], "descriptor": desc("centroid", DataType.NDARRAY, "label", u)},
            "f_vector": {"data": g["f_vector"], "descriptor": desc("f_vector", DataType.NDARRAY, "label")},
            "n_edges": {"data": g["n_edges"], "descriptor": desc("n_edges", DataType.INT, "label")},
            "longest_edge": {"data": g["longest_edge"], "descriptor": desc("longest_edge", DataType.STRING, "label")},
            "projection": {"data": self.projection.copy(),
                           "descriptor": desc("projection", DataType.NDARRAY, "auxiliary")},
            "projected_vertices": {"data": self.vertices @ self.projection,
                                   "descriptor": desc("projected_vertices", DataType.NDARRAY, "auxiliary", u)},
        }

    # ------------------------------------------------------------------ bundle hooks
    def figure_elements(self, results):
        """Projected outline (polygon), every edge (segments) and labelled vertices."""
        v = results["vertices"]["data"]
        d = v.shape[1]
        p = v @ results["projection"]["data"]
        names = self.labels(d)
        elements: List[Dict[str, Any]] = []
        if d == 2:
            elements.append({"kind": "polygon", "label": "triangle P0P1P2", "points": p.tolist()})
        else:
            hull = _convex_hull(p)
            if len(hull) >= 3:
                elements.append({"kind": "polygon", "label": "outline of the projection",
                                 "points": p[hull].tolist()})
        for i, j in edge_pairs(d):
            elements.append({"kind": "segment", "label": f"edge {names[i]}{names[j]}",
                             "points": [p[i].tolist(), p[j].tolist()]})
        for name, q in zip(names, p):
            elements.append({"kind": "point", "label": name, "points": [q.tolist()]})
        return elements

    def figure_caption(self, results):
        d = results["vertices"]["data"].shape[1]
        n_edges = math.comb(d + 1, 2)
        if d == 2:
            return (f"Diagram of a 2-dimensional simplex (triangle) P0P1P2 drawn in its own plane, with its "
                    f"3 edges; coordinates in {self.units}.")
        plane = ("a randomly oriented plane" if self.projection_kind == "random"
                 else "the plane of the first two coordinate axes")
        return (f"Orthographic projection onto {plane} of a {d}-dimensional simplex with vertices P0 to P{d}; "
                f"all {n_edges} edges are drawn (dashed) and the outline of the projection is solid. "
                f"Projected lengths and angles are not true lengths; plane coordinates in {self.units}.")

    @staticmethod
    def _num(x: float) -> str:
        return f"{float(x):.10g}"

    def _given(self, v: np.ndarray) -> str:
        """Self-contained statement of the simplex for questions that cannot be read from a projection."""
        d = v.shape[1]
        if self.regular:
            return f"A regular {d}-dimensional simplex has edge length {self._num(self.edge_length)} {self.units}."
        coords = "; ".join(f"P{i} = (" + ", ".join(self._num(x) for x in p) + ")" for i, p in enumerate(v))
        return f"A {d}-dimensional simplex has vertices (coordinates in {self.units}) {coords}."

    def questions(self, results, rng):
        """Exact questions; metric ones are answerable from the figure only for d = 2."""
        v = results["vertices"]["data"]
        g = self.compute(v)
        d, u = g["d"], self.units
        planar = d == 2
        source = "figure" if planar else "data"
        given = "" if planar else self._given(v) + " "
        rel = 0.03 if planar else 0.01  # figure reading vs. computation from the stated coordinates
        rule = f"{int(rel * 100)} % of the true value"
        what = "the triangle P0P1P2" if planar else "the simplex"
        measure = "area" if planar else f"{d}-dimensional volume"
        items = [
            dict(template="n_edges", question=f"How many edges does the {d}-dimensional simplex in the figure have?",
                 answer=g["n_edges"], answer_type="integer", tolerance=None, tolerance_rule="exact",
                 answerable_from="figure"),
            dict(template="volume", question=f"{given}What is the {measure} of {what} (in {u}^{d})?",
                 answer=g["volume"], answer_type="number", tolerance=rel * g["volume"], tolerance_rule=rule,
                 answerable_from=source),
            dict(template="circumradius",
                 question=f"{given}What is the radius of the circumscribed {'circle' if planar else 'hypersphere'} "
                          f"of {what} (in {u})?",
                 answer=g["circumradius"], answer_type="number", tolerance=rel * g["circumradius"],
                 tolerance_rule=rule, answerable_from=source),
            dict(template="inradius",
                 question=f"{given}What is the radius of the inscribed {'circle' if planar else 'hypersphere'} "
                          f"of {what} (in {u})?",
                 answer=g["inradius"], answer_type="number", tolerance=(0.05 if planar else rel) * g["inradius"],
                 tolerance_rule="5 % of the true value" if planar else rule, answerable_from=source),
        ]
        if d >= 3:
            k = int(rng.integers(2, d))  # 2 <= k <= d - 1
            items.append(dict(template="k_faces",
                              question=f"How many {k}-dimensional faces does a {d}-dimensional simplex have?",
                              answer=int(g["f_vector"][k]), answer_type="integer", tolerance=None,
                              tolerance_rule="exact", answerable_from="data"))
        ranked = np.sort(g["edges"])
        if ranked[-1] > (1.05 if planar else 1.01) * ranked[-2]:
            labels = g["edge_labels"]
            items.append(dict(template="longest_edge",
                              question=f"{given}Which edge of {what} is the longest: {', '.join(labels)}?",
                              answer=g["longest_edge"], answer_type="choice", choices=labels, tolerance=None,
                              tolerance_rule="exact", answerable_from=source))
        for item in items:
            item.setdefault("series", None)
        return items
