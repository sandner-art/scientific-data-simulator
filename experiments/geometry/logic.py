# experiments/geometry/logic.py
"""Triangle constructions with exact coordinates (plane Euclidean geometry).

A triangle ABC is either given (``vertices: [[x, y], [x, y], [x, y]]``) or drawn
with ``self.rng`` inside a square of side ``extent`` (rejecting triangles with an
angle below ``min_angle_deg``). The model computes exact side lengths, angles,
area, perimeter, circle radii and triangle centres, and describes a diagram of
the requested constructions:

    constructions: [circumcircle, incircle, centroid, orthocenter, medians, altitudes, euler_line]

Outputs: ``vertices`` (parameter, 3x2), ``side_lengths`` [a=BC, b=CA, c=AB] and
``angles_deg`` [A, B, C] (latent; 1-D, so observation operators can turn them
into noisy measurements), and labels: area, perimeter, circumradius, inradius,
centroid, circumcenter, incenter, orthocenter, triangle_type.

Bundles draw the diagram from :meth:`figure_elements` (code + element table +
image) and ask the questions from :meth:`questions` (area, angles, circle radii,
type of triangle, longest side, Euler-line distance), all with exact answers.
"""
import math

import numpy as np

from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType

CONSTRUCTIONS = ("circumcircle", "incircle", "centroid", "orthocenter", "medians", "altitudes", "euler_line")
NAMES = ("A", "B", "C")


def _cross(u, w):
    """z-component of the cross product of two plane vectors (np.cross is 3-D only in NumPy 2)."""
    return float(u[0] * w[1] - u[1] * w[0])


def _foot(p, a, b):
    """Foot of the perpendicular from p onto line ab."""
    d = b - a
    return a + d * float(np.dot(p - a, d) / np.dot(d, d))


class TriangleExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = ("Closed-form Euclidean geometry; checked against independent identities (Heron vs. shoelace, "
                     "law of sines, Euler's OI^2 = R(R - 2r), Euler line OG:GH = 1:2, angle sum).")
    @classmethod
    def parameter_units(cls, config):
        """``extent`` is in the configured ``length_units`` (default cm)."""
        return {"extent": config.get("length_units", "cm")}

    def __init__(self, config):
        self.vertices_config = config.get("vertices")
        self.extent = float(config.get("extent", 10.0))
        self.min_angle = float(config.get("min_angle_deg", 15.0))
        self.units = config.get("length_units", "cm")
        self.constructions = list(config.get("constructions", ["circumcircle", "incircle", "centroid"]))
        unknown = set(self.constructions) - set(CONSTRUCTIONS)
        if unknown:
            raise ValueError(f"Unknown constructions {sorted(unknown)}; choose from {CONSTRUCTIONS}")
        if not 0 < self.min_angle < 60:
            raise ValueError("min_angle_deg must be in (0, 60)")
        self.vertices = None

    # ------------------------------------------------------------------ geometry
    def _draw_vertices(self):
        if self.vertices_config is not None:
            v = np.asarray(self.vertices_config, dtype=float)
            if v.shape != (3, 2):
                raise ValueError("vertices must be three [x, y] pairs")
            if abs(_cross(v[1] - v[0], v[2] - v[0])) < 1e-12:
                raise ValueError("vertices are collinear")
            return v
        rng = self.get_rng()
        for _ in range(10000):
            v = rng.uniform(0.0, self.extent, size=(3, 2))
            if abs(_cross(v[1] - v[0], v[2] - v[0])) < 1e-9 * self.extent ** 2:
                continue
            if min(self._angles(v)) >= self.min_angle:
                return v
        raise RuntimeError("could not draw a triangle satisfying min_angle_deg")

    @staticmethod
    def _sides(v):
        a = np.linalg.norm(v[1] - v[2])  # opposite A
        b = np.linalg.norm(v[2] - v[0])  # opposite B
        c = np.linalg.norm(v[0] - v[1])  # opposite C
        return np.array([a, b, c])

    @classmethod
    def _angles(cls, v):
        out = []
        for i in range(3):
            p, q, r = v[i], v[(i + 1) % 3], v[(i + 2) % 3]
            u, w = q - p, r - p
            cosine = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
            out.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
        return np.array(out)

    def compute(self, v):
        a, b, c = sides = self._sides(v)
        angles = self._angles(v)
        area = 0.5 * abs(_cross(v[1] - v[0], v[2] - v[0]))  # shoelace
        perimeter = float(sides.sum())
        centroid = v.mean(axis=0)
        # circumcenter
        ax, ay = v[0]
        bx, by = v[1]
        cx, cy = v[2]
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d
        circumcenter = np.array([ux, uy])
        circumradius = float(np.linalg.norm(v[0] - circumcenter))
        incenter = (a * v[0] + b * v[1] + c * v[2]) / perimeter
        inradius = float(area / (perimeter / 2.0))
        orthocenter = v.sum(axis=0) - 2.0 * circumcenter  # H = A + B + C - 2O
        largest = float(angles.max())
        kind = "right" if abs(largest - 90.0) < 1e-6 else ("obtuse" if largest > 90.0 else "acute")
        return dict(sides=sides, angles=angles, area=float(area), perimeter=perimeter, centroid=centroid,
                    circumcenter=circumcenter, circumradius=circumradius, incenter=incenter, inradius=inradius,
                    orthocenter=orthocenter, triangle_type=kind)

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

        def d(name, dtype, role, units=None, group="geometry"):
            return DataDescriptor(name, dtype, units=units, group=group, role=role)
        return {
            "vertices": {"data": self.vertices.copy(), "descriptor": d("vertices", DataType.NDARRAY, "parameter", u)},
            "side_lengths": {"data": g["sides"], "descriptor": d("side_lengths", DataType.NDARRAY, "latent", u)},
            "angles_deg": {"data": g["angles"], "descriptor": d("angles_deg", DataType.NDARRAY, "latent", "degree")},
            "area": {"data": g["area"], "descriptor": d("area", DataType.FLOAT, "label", area_units)},
            "perimeter": {"data": g["perimeter"], "descriptor": d("perimeter", DataType.FLOAT, "label", u)},
            "circumradius": {"data": g["circumradius"], "descriptor": d("circumradius", DataType.FLOAT, "label", u)},
            "inradius": {"data": g["inradius"], "descriptor": d("inradius", DataType.FLOAT, "label", u)},
            "centroid": {"data": g["centroid"], "descriptor": d("centroid", DataType.NDARRAY, "label", u)},
            "circumcenter": {"data": g["circumcenter"], "descriptor": d("circumcenter", DataType.NDARRAY, "label", u)},
            "incenter": {"data": g["incenter"], "descriptor": d("incenter", DataType.NDARRAY, "label", u)},
            "orthocenter": {"data": g["orthocenter"], "descriptor": d("orthocenter", DataType.NDARRAY, "label", u)},
            "triangle_type": {"data": g["triangle_type"], "descriptor": d("triangle_type", DataType.STRING, "label")},
        }

    # ------------------------------------------------------------------ bundle hooks
    def figure_elements(self, results):
        """Drawable elements of the diagram (all coordinates exact)."""
        v = results["vertices"]["data"]
        g = self.compute(v)
        elements = [{"kind": "polygon", "label": "triangle ABC", "points": v.tolist()}]
        for name, p in zip(NAMES, v):
            elements.append({"kind": "point", "label": name, "points": [p.tolist()]})
        if "circumcircle" in self.constructions:
            elements.append({"kind": "circle", "label": "circumcircle", "points": [g["circumcenter"].tolist()],
                             "r": g["circumradius"]})
            elements.append({"kind": "point", "label": "O", "points": [g["circumcenter"].tolist()]})
        if "incircle" in self.constructions:
            elements.append({"kind": "circle", "label": "incircle", "points": [g["incenter"].tolist()],
                             "r": g["inradius"]})
            elements.append({"kind": "point", "label": "I", "points": [g["incenter"].tolist()]})
        if "centroid" in self.constructions or "medians" in self.constructions:
            elements.append({"kind": "point", "label": "G", "points": [g["centroid"].tolist()]})
        if "orthocenter" in self.constructions or "altitudes" in self.constructions:
            elements.append({"kind": "point", "label": "H", "points": [g["orthocenter"].tolist()]})
        for i in range(3):
            p, q, r = v[i], v[(i + 1) % 3], v[(i + 2) % 3]
            if "medians" in self.constructions:
                elements.append({"kind": "segment", "label": f"median from {NAMES[i]}",
                                 "points": [p.tolist(), ((q + r) / 2).tolist()]})
            if "altitudes" in self.constructions:
                elements.append({"kind": "segment", "label": f"altitude from {NAMES[i]}",
                                 "points": [p.tolist(), _foot(p, q, r).tolist()]})
        if "euler_line" in self.constructions:
            elements.append({"kind": "segment", "label": "Euler line",
                             "points": [g["circumcenter"].tolist(), g["orthocenter"].tolist()]})
        return elements

    def figure_caption(self, results):
        names = {"euler_line": "Euler line", "orthocenter": "orthocentre H", "centroid": "centroid G",
                 "circumcircle": "circumcircle (centre O)", "incircle": "incircle (centre I)"}
        shown = [names.get(c, c.replace("_", " ")) for c in self.constructions]
        return (f"Diagram of triangle ABC" + (f" with its {', '.join(shown)}" if shown else "")
                + f"; coordinates in {self.units}.")

    def questions(self, results, rng):
        """Exact questions about the triangle (ids are assigned by the bundle)."""
        g = self.compute(results["vertices"]["data"])
        u = self.units
        items = [
            dict(template="area", question=f"What is the area of triangle ABC (in {u}^2)?", answer=g["area"],
                 answer_type="number", tolerance=0.03 * g["area"], tolerance_rule="3 % of the true value"),
            dict(template="perimeter", question=f"What is the perimeter of triangle ABC (in {u})?",
                 answer=g["perimeter"], answer_type="number", tolerance=0.03 * g["perimeter"], tolerance_rule="3 % of the true value"),
            dict(template="triangle_type", question="Is triangle ABC acute, right or obtuse?",
                 answer=g["triangle_type"], answer_type="choice", choices=["acute", "right", "obtuse"],
                 tolerance=None),
        ]
        vertex = int(rng.integers(3))
        items.append(dict(template="angle", question=f"What is the interior angle at vertex {NAMES[vertex]} "
                                                     f"(in degrees)?",
                          answer=float(g["angles"][vertex]), answer_type="number", tolerance=2.0, tolerance_rule="2 degrees"))
        sides = {"BC": g["sides"][0], "CA": g["sides"][1], "AB": g["sides"][2]}
        ranked = sorted(sides.values())
        if ranked[-1] > 1.05 * ranked[-2]:
            items.append(dict(template="longest_side", question="Which side is the longest: AB, BC or CA?",
                              answer=max(sides, key=sides.get), answer_type="choice", choices=["AB", "BC", "CA"],
                              tolerance=None))
        if "circumcircle" in self.constructions:
            items.append(dict(template="circumradius", question=f"What is the radius of the circumcircle (in {u})?",
                              answer=g["circumradius"], answer_type="number", tolerance=0.03 * g["circumradius"], tolerance_rule="3 % of the true value"))
        if "incircle" in self.constructions:
            items.append(dict(template="inradius", question=f"What is the radius of the incircle (in {u})?",
                              answer=g["inradius"], answer_type="number", tolerance=0.05 * g["inradius"], tolerance_rule="5 % of the true value"))
        if "euler_line" in self.constructions or {"centroid", "circumcircle"} <= set(self.constructions):
            og = float(np.linalg.norm(g["centroid"] - g["circumcenter"]))
            items.append(dict(template="euler_distance",
                              question=f"What is the distance between the circumcenter O and the centroid G "
                                       f"(in {u})?",
                              answer=og, answer_type="number", tolerance=max(0.05 * og, 0.01 * g["circumradius"]),
                              tolerance_rule="5 % of the true value (at least 1 % of the circumradius)"))
        for item in items:
            item.setdefault("series", None)
            item["answerable_from"] = "figure"
        return items
