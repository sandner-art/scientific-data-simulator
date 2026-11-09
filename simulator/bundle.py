# simulator/bundle.py
"""Multimodal run bundles.

One run -> one aligned bundle, all derived from the same arrays and seed::

    bundle/
      figure.png         rendered by plot_figure.py               (public)
      plot_figure.py     standalone script: table -> figure        (public)
      table.csv          exactly the data drawn in the figure      (private*; time series)
      elements.csv       exactly the elements drawn                (private*; diagrams)
      caption.txt        what the figure shows; no hidden answers  (public)
      questions.json     questions without answers                 (public)
      qa.json            questions + verifiable answers            (private)
      graph.json/.dot    mechanism graph (node-link / Graphviz)    (private)
      description.md     mechanism, parameters, noise, fidelity    (private)
      bundle.json        manifest: files, sha256, visibility       (private)

"Public" files can be handed to a system under test; "private" files hold the
answers. (*) The exact data table answers figure questions verbatim, so it is
private unless ``bundle.public_table: true`` (training data). The figure is produced *by executing* plot_figure.py, so code, image
and table cannot disagree. Answers are computed from the exact (latent) arrays;
each question says whether it can be answered from the figure, or only by
inference (e.g. a rate constant, or a latent curve that is not drawn).

Configuration (``bundle:`` in the experiment config; ``bundle: true`` uses defaults)::

    bundle:
      figure:
        format: png            # png | svg | pdf
        dpi: 120
        width: 6.4             # inches
        height: 4.8
        show_latent: true      # draw exact curves (false: observations only -> harder)
        show_observed: true
        legend: true           # (diagrams: point labels)
        grid: true
        title: true
        axes: true             # diagrams: coordinate axes (false: bare diagram)
      qa:
        enabled: true
        value_queries: 2       # random "value at x" questions per series
        parameters: null       # parameter-inference questions; default: model.inferable_parameters
        exclude: []            # templates to leave out (e.g. ones a quality card shows are degenerate)
      graph: true
      public_table: false      # true: mark table.csv / elements.csv public (training data)
"""
from __future__ import annotations

import copy
import json
import os
import runpy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_handler import results_to_dataframe
from .provenance import file_sha256
from .utils import Role

PUBLIC = "public"
PRIVATE = "private"

#: Question fields withheld from public files. The numeric tolerance is private
#: because it is derived from the answer (5 % of a parameter) or from the data
#: (3 % of a series' range) and would leak them; public files carry the textual
#: ``tolerance_rule`` instead.
PRIVATE_QUESTION_KEYS = ("answer", "tolerance")


def public_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """A question as it may be shown to a system under test."""
    return {k: v for k, v in item.items() if k not in PRIVATE_QUESTION_KEYS}

DEFAULTS: Dict[str, Any] = {
    "figure": {"format": "png", "dpi": 120, "width": 6.4, "height": 4.8, "show_latent": True,
               "show_observed": True, "legend": True, "grid": True, "title": True, "axes": True},
    "qa": {"enabled": True, "value_queries": 2, "parameters": None, "exclude": []},
    "graph": True,
    # The table/elements file holds the exact values drawn (curve values, circle
    # radii) and so answers figure questions verbatim. Private by default (benchmark
    # use); set true for training data, where the aligned table is wanted.
    "public_table": False,
}

# matplotlib's default categorical palette (tab10), fixed so bundles do not
# depend on the user's rcParams.
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# ---------------------------------------------------------------------------- config
def resolve_bundle_config(spec: Any) -> Optional[Dict[str, Any]]:
    """Merge ``bundle:`` with defaults; ``None``/``False`` disables bundles."""
    if spec in (None, False):
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError("'bundle' must be true/false or a mapping")
    unknown = set(spec) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown bundle keys: {sorted(unknown)}")
    cfg = copy.deepcopy(DEFAULTS)
    for section in ("figure", "qa"):
        sub = spec.get(section) or {}
        if not isinstance(sub, dict):
            raise ValueError(f"bundle.{section} must be a mapping")
        unknown = set(sub) - set(DEFAULTS[section])
        if unknown:
            raise ValueError(f"Unknown bundle.{section} keys: {sorted(unknown)}")
        cfg[section].update(sub)
    if "graph" in spec:
        cfg["graph"] = bool(spec["graph"])
    if "public_table" in spec:
        cfg["public_table"] = bool(spec["public_table"])
    fig = cfg["figure"]
    if fig["format"] not in ("png", "svg", "pdf"):
        raise ValueError("bundle.figure.format must be png, svg or pdf")
    if not (fig["show_latent"] or fig["show_observed"]):
        raise ValueError("bundle.figure: show_latent and show_observed cannot both be false")
    for key in ("dpi", "width", "height"):
        if not (isinstance(fig[key], (int, float)) and fig[key] > 0):
            raise ValueError(f"bundle.figure.{key} must be a positive number")
    if int(cfg["qa"]["value_queries"]) < 0:
        raise ValueError("bundle.qa.value_queries must be >= 0")
    exclude = cfg["qa"].get("exclude") or []
    if isinstance(exclude, str) or not all(isinstance(t, str) for t in exclude):
        raise ValueError("bundle.qa.exclude must be a list of template names")
    cfg["qa"]["exclude"] = list(exclude)
    return cfg


# ---------------------------------------------------------------------------- helpers
def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _base_name(name: str, suffix: str) -> str:
    return name[: -len(suffix)] if suffix and name.endswith(suffix) else name


def _x_name(results: Dict[str, Dict[str, Any]], descriptor) -> Optional[str]:
    x = descriptor.x_axis or "time"
    return x if x in results else None


def _is_series(info: Dict[str, Any]) -> bool:
    data = info["data"]
    return isinstance(data, np.ndarray) and data.ndim == 1 and np.issubdtype(data.dtype, np.number)


def _latent_series(results) -> List[Tuple[str, str]]:
    """(name, coordinate name) for every 1-D latent output with a coordinate."""
    out = []
    for name, info in results.items():
        d = info["descriptor"]
        if d.role is Role.LATENT and _is_series(info):
            x = _x_name(results, d)
            if x is not None and len(results[x]["data"]) == len(info["data"]):
                out.append((name, x))
    return out


def _plotted_series(results, fig_cfg, suffix) -> List[Dict[str, Any]]:
    """Series drawn in the figure: time-series lines/markers, filtered by role."""
    series = []
    for name, info in results.items():
        d = info["descriptor"]
        if d.group != "time_series" or d.plot_type not in ("line", "scatter") or not _is_series(info):
            continue
        if d.role is Role.OBSERVATION and not fig_cfg["show_observed"]:
            continue
        if d.role is not Role.OBSERVATION and not fig_cfg["show_latent"]:
            continue
        x = _x_name(results, d)
        if x is None:
            continue
        series.append({"name": name, "x": x, "style": "scatter" if d.plot_type == "scatter" else "line",
                       "base": _base_name(name, suffix), "role": d.role.value if d.role else None,
                       "units": d.units})
    colours: Dict[str, str] = {}
    for s in series:
        colours.setdefault(s["base"], PALETTE[len(colours) % len(PALETTE)])
        s["color"] = colours[s["base"]]
    return series


# ---------------------------------------------------------------------------- figure
_SCRIPT = '''#!/usr/bin/env python3
"""Regenerate {output} from {table}.

Generated by Scientific Data Simulator (the run is identified in bundle.json).
Needs pandas and matplotlib only; run it from anywhere: python plot_figure.py
"""
import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
SPEC = json.loads(r"""{spec}""")


def main(output=None):
    table = pd.read_csv(HERE / SPEC["table"])
    fig = Figure(figsize=(SPEC["width"], SPEC["height"]), dpi=SPEC["dpi"])
    ax = fig.add_subplot()
    for s in SPEC["series"]:
        d = table[[s["x"], s["y"]]].dropna()
        if s["style"] == "line":
            ax.plot(d[s["x"]], d[s["y"]], color=s["color"], linewidth=1.5, label=s["label"])
        else:
            ax.plot(d[s["x"]], d[s["y"]], color=s["color"], linestyle="none", marker="o",
                    markersize=3, alpha=0.8, label=s["label"])
    ax.set_xlabel(SPEC["xlabel"])
    ax.set_ylabel(SPEC["ylabel"])
    if SPEC["title"]:
        ax.set_title(SPEC["title"])
    if SPEC["grid"]:
        ax.grid(True, alpha=0.3)
    if SPEC["legend"]:
        ax.legend(fontsize="small")
    fig.savefig(output or HERE / SPEC["output"], metadata=SPEC["metadata"])


if __name__ == "__main__":
    main()
'''

_DIAGRAM_SCRIPT = '''#!/usr/bin/env python3
"""Regenerate {output} (a geometric diagram) from {table}.

Generated by Scientific Data Simulator (the run is identified in bundle.json).
Needs pandas and matplotlib only; run it from anywhere: python plot_figure.py
"""
import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon

HERE = Path(__file__).resolve().parent
SPEC = json.loads(r"""{spec}""")


def main(output=None):
    table = pd.read_csv(HERE / SPEC["table"])
    fig = Figure(figsize=(SPEC["width"], SPEC["height"]), dpi=SPEC["dpi"])
    ax = fig.add_subplot()
    xs, ys = [], []
    for row in table.itertuples():
        pts = json.loads(row.points)
        xs += [p[0] for p in pts]
        ys += [p[1] for p in pts]
        if row.kind == "polygon":
            ax.add_patch(Polygon(pts, closed=True, fill=False, linewidth=1.8, color="#1f77b4"))
        elif row.kind == "segment":
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color="#7f7f7f", linewidth=1.0,
                    linestyle="--")
        elif row.kind == "circle":
            ax.add_patch(Circle(pts[0], row.r, fill=False, linewidth=1.2, color="#2ca02c"))
            xs += [pts[0][0] - row.r, pts[0][0] + row.r]
            ys += [pts[0][1] - row.r, pts[0][1] + row.r]
        elif row.kind == "point":
            ax.plot([pts[0][0]], [pts[0][1]], "o", color="#d62728", markersize=3)
            if SPEC["labels"]:
                ax.annotate(row.label, pts[0], textcoords="offset points", xytext=(4, 4), fontsize=10)
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    if SPEC["axes"]:
        ax.set_xlabel(SPEC["xlabel"])
        ax.set_ylabel(SPEC["ylabel"])
        if SPEC["grid"]:
            ax.grid(True, alpha=0.3)
    else:
        ax.set_axis_off()
    if SPEC["title"]:
        ax.set_title(SPEC["title"])
    fig.savefig(output or HERE / SPEC["output"], metadata=SPEC["metadata"])


if __name__ == "__main__":
    main()
'''

_METADATA = {"png": {"Software": None}, "svg": {"Date": None, "Creator": None},
             "pdf": {"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None}}


def _axis_label(name: str, units: Optional[str]) -> str:
    return f"{name} [{units}]" if units else name


def _base_coordinate(series, suffix) -> str:
    for s in series:
        if s["role"] != Role.OBSERVATION.value:
            return s["x"]
    return _base_name(series[0]["x"], suffix)


def _write_figure(bundle_dir, results, series, fig_cfg, experiment_id, model_name, suffix) -> Dict[str, Any]:
    columns = {s["name"] for s in series} | {s["x"] for s in series}
    subset = {n: results[n] for n in columns}
    if "observation_index" in results:
        subset["observation_index"] = results["observation_index"]  # aligns sampled rows
    table = results_to_dataframe(subset)
    table = table[[c for c in table.columns if c in columns]]
    table = table[sorted(table.columns, key=lambda c: (c not in {s["x"] for s in series}, c))]
    table.to_csv(os.path.join(bundle_dir, "table.csv"), index=False)

    x_units = results[series[0]["x"]]["descriptor"].units
    x_label = _base_coordinate(series, suffix)
    y_units = {s["units"] for s in series}
    spec = {
        "table": "table.csv", "output": f"figure.{fig_cfg['format']}",
        "width": fig_cfg["width"], "height": fig_cfg["height"], "dpi": fig_cfg["dpi"],
        "xlabel": _axis_label(x_label, x_units),
        "ylabel": _axis_label("value", y_units.pop()) if len(y_units) == 1 else "value",
        "title": model_name if fig_cfg["title"] else None,
        "grid": fig_cfg["grid"], "legend": fig_cfg["legend"],
        "metadata": _METADATA[fig_cfg["format"]],
        "series": [{"x": s["x"], "y": s["name"], "style": s["style"], "color": s["color"],
                    "label": s["name"]} for s in series],
    }
    script = _SCRIPT.format(output=spec["output"], table=spec["table"], spec=json.dumps(spec, indent=2))
    script_path = os.path.join(bundle_dir, "plot_figure.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    # Render by executing the shipped script: code, table and image stay consistent.
    runpy.run_path(script_path, run_name="__bundle__")["main"]()
    return spec


def _write_diagram(bundle_dir, elements, fig_cfg, model_name, length_units) -> Dict[str, Any]:
    rows = []
    for e in elements:
        if e.get("kind") not in ("polygon", "segment", "circle", "point"):
            raise ValueError(f"Unknown figure element kind {e.get('kind')!r}")
        rows.append({"kind": e["kind"], "label": e.get("label", ""), "points": json.dumps(e["points"]),
                     "r": float(e.get("r", float("nan")))})
    pd.DataFrame(rows, columns=["kind", "label", "points", "r"]).to_csv(
        os.path.join(bundle_dir, "elements.csv"), index=False)
    spec = {
        "table": "elements.csv", "output": f"figure.{fig_cfg['format']}",
        "width": fig_cfg["width"], "height": fig_cfg["height"], "dpi": fig_cfg["dpi"],
        "xlabel": _axis_label("x", length_units), "ylabel": _axis_label("y", length_units),
        "title": model_name if fig_cfg["title"] else None, "grid": fig_cfg["grid"],
        "axes": fig_cfg["axes"], "labels": fig_cfg["legend"],
        "metadata": _METADATA[fig_cfg["format"]],
    }
    script = _DIAGRAM_SCRIPT.format(output=spec["output"], table=spec["table"], spec=json.dumps(spec, indent=2))
    script_path = os.path.join(bundle_dir, "plot_figure.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    runpy.run_path(script_path, run_name="__bundle__")["main"]()
    return spec


# ---------------------------------------------------------------------------- text
def _observation_phrase(realized: Optional[Dict[str, Any]]) -> Optional[str]:
    if not realized:
        return None
    parts = []
    sampling = realized.get("sampling") or {}
    if sampling.get("method") == "every":
        parts.append(f"sampled every {sampling['every']} steps")
    elif sampling.get("method") in ("uniform", "random"):
        parts.append(f"sampled at {sampling['n']} {'random' if sampling['method'] == 'random' else 'evenly spaced'} points")
    elif sampling.get("method") == "indices":
        parts.append("sampled at selected points")
    ops = next(iter(realized.get("targets", {}).values()), {}).get("operators", [])
    words = {"gaussian": "additive Gaussian noise", "multiplicative": "multiplicative noise",
             "poisson": "Poisson counting noise", "quantize": "quantisation", "outliers": "occasional outliers",
             "missing": "missing values", "clip": "clipping"}
    noise = [words.get(op["type"], op["type"]) for op in ops]
    if noise:
        parts.append("with " + ", ".join(noise))
    return " ".join(parts) if parts else None


def _caption(series, results, realized, model_name, suffix) -> str:
    latent = sorted({s["name"] for s in series if s["role"] != Role.OBSERVATION.value})
    observed = sorted({s["base"] for s in series if s["role"] == Role.OBSERVATION.value})
    x = _base_coordinate(series, suffix)
    xs = np.asarray(results[x]["data"] if x in results else results[series[0]["x"]]["data"], dtype=float)
    text = f"Time-series chart from the {model_name} model, {x} from {_fmt(np.nanmin(xs))} to {_fmt(np.nanmax(xs))}."
    if latent:
        text += f" Lines show the exact model output for {', '.join(latent)}."
    if observed:
        phrase = _observation_phrase(realized)
        text += f" Markers show observations of {', '.join(observed)}" + (f", {phrase}." if phrase else ".")
    return text


def _description(record, gt, fidelity, model_name, caption) -> str:
    lines = [f"# Run {record.experiment_id}", "", f"Model: `{gt.get('experiment_type')}` ({model_name})",
             f"Seed: {gt.get('seed')}", ""]
    if fidelity:
        lines += [f"Fidelity: **{fidelity['level']}** - {fidelity.get('note') or ''}", ""]
    lines += ["## Figure", "", caption, "", "## Parameters (ground truth)", "", "| parameter | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted((gt.get("parameters") or {}).items())]
    shift = gt.get("law_shift")
    if shift:
        lines += ["", "## Law shift", "", shift.get("description") or ""]
        lines += [f"- `{k}`: {c['original']!r} -> {c['shifted']!r}" for k, c in shift["changes"].items()]
    obs = gt.get("observation")
    if obs:
        lines += ["", "## Observation model", ""]
        for target, info in obs.get("targets", {}).items():
            ops = "; ".join(", ".join(f"{k}={v}" for k, v in op.items() if k != "indices") for op in info["operators"])
            lines.append(f"- `{target}` -> `{info['observed_as']}`: {ops or 'no value operators'}")
        if obs.get("sampling"):
            lines.append(f"- sampling: {obs['sampling']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- questions
def _tolerances(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Figure-reading tolerances: 3 % of the y range, max(2 % of x range, 1.5 grid steps)."""
    y_range = float(np.nanmax(y) - np.nanmin(y))
    y_tol = 0.03 * y_range if y_range > 0 else max(0.03 * abs(float(np.nanmax(y))), 1e-9)
    x_range = float(x[-1] - x[0]) if len(x) > 1 else 0.0
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
    return y_tol, max(0.02 * abs(x_range), 1.5 * abs(dx), 1e-9)


def _peak_count(y: np.ndarray) -> int:
    from scipy.signal import find_peaks
    y_range = float(np.nanmax(y) - np.nanmin(y))
    if y_range <= 0:
        return 0
    peaks, _ = find_peaks(y, prominence=0.1 * y_range)
    return int(len(peaks))


def generate_questions(results, plotted: List[Dict[str, Any]], gt: Dict[str, Any],
                       inferable: Tuple[str, ...], qa_cfg: Dict[str, Any],
                       rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Questions with verifiable answers computed from the exact (latent) arrays."""
    visible_latent = {s["name"] for s in plotted if s["role"] != Role.OBSERVATION.value}
    visible_observed = {s["base"] for s in plotted if s["role"] == Role.OBSERVATION.value}
    items: List[Dict[str, Any]] = []

    def add(**item):
        item["id"] = f"q{len(items) + 1}"
        items.append(item)

    series_list = _latent_series(results)
    for name, x_name in series_list:
        y = np.asarray(results[name]["data"], dtype=float)
        x = np.asarray(results[x_name]["data"], dtype=float)
        if not np.all(np.isfinite(y)) or len(y) < 2:
            continue
        source = ("figure" if name in visible_latent
                  else "inference" if name in visible_observed else "data")
        y_tol, x_tol = _tolerances(x, y)
        units = results[name]["descriptor"].units
        unit_txt = f" (in {units})" if units else ""
        i_max = int(np.argmax(y))
        add(template="argmax", series=name, answerable_from=source, answer_type="number",
            question=f"At what {x_name} does {name} reach its maximum?", answer=float(x[i_max]), tolerance=x_tol,
            tolerance_rule="2 % of the x range (at least 1.5 grid steps)")
        add(template="max", series=name, answerable_from=source, answer_type="number",
            question=f"What is the maximum value of {name}{unit_txt}?", answer=float(y[i_max]), tolerance=y_tol, tolerance_rule="3 % of the series' range")
        add(template="min", series=name, answerable_from=source, answer_type="number",
            question=f"What is the minimum value of {name}{unit_txt}?", answer=float(np.min(y)), tolerance=y_tol, tolerance_rule="3 % of the series' range")
        add(template="final", series=name, answerable_from=source, answer_type="number",
            question=f"What is the value of {name} at {x_name} = {_fmt(x[-1])}?", answer=float(y[-1]), tolerance=y_tol, tolerance_rule="3 % of the series' range")
        for _ in range(int(qa_cfg["value_queries"])):
            i = int(rng.integers(0, len(x)))
            add(template="value_at", series=name, answerable_from=source, answer_type="number",
                question=f"What is the value of {name} at {x_name} = {_fmt(x[i])}?", answer=float(y[i]),
                tolerance=y_tol, tolerance_rule="3 % of the series' range")
        add(template="peak_count", series=name, answerable_from=source, answer_type="integer",
            question=f"How many distinct peaks (local maxima rising at least 10 % of its range) does {name} have?",
            answer=_peak_count(y), tolerance=None, tolerance_rule="exact")

    # comparisons between latent series sharing units and coordinate
    for i, (a, xa) in enumerate(series_list):
        for b, xb in series_list[i + 1:]:
            if xa != xb or results[a]["descriptor"].units != results[b]["descriptor"].units:
                continue
            ya, yb = np.max(results[a]["data"]), np.max(results[b]["data"])
            if np.isclose(ya, yb, rtol=0.05):
                continue  # too close to call reliably from a figure
            source = "figure" if {a, b} <= visible_latent else "inference"
            add(template="compare_max", series=[a, b], answerable_from=source, answer_type="choice",
                question=f"Which reaches a higher maximum: {a} or {b}?", choices=[a, b],
                answer=a if ya > yb else b, tolerance=None, tolerance_rule="exact")

    params = gt.get("parameters") or {}
    wanted = qa_cfg["parameters"] if qa_cfg["parameters"] is not None else list(inferable)
    for p in wanted:
        value = params.get(p)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        add(template="parameter", series=None, parameter=p, answerable_from="inference", answer_type="number",
            question=f"Estimate the model parameter {p} from the data.", answer=float(value),
            tolerance=max(0.05 * abs(float(value)), 1e-12), tolerance_rule="5 % of the true value")
    return items


def check_answer(item: Dict[str, Any], response: Any) -> bool:
    """Score a response against a QA item: numeric within tolerance, exact integer,
    choice, or expression (same function on the item's domain, see
    :func:`simulator.expressions.matches`)."""
    kind = item["answer_type"]
    try:
        if kind == "number":
            tol = item.get("tolerance") or 1e-9 * max(1.0, abs(item["answer"]))
            return abs(float(response) - float(item["answer"])) <= tol
        if kind == "integer":
            return int(response) == int(item["answer"])
        if kind == "choice":
            return str(response).strip().lower() == str(item["answer"]).strip().lower()
        if kind == "expression":
            from .expressions import matches
            return matches(str(response), str(item["answer"]), tuple(item["domain"]),
                           item.get("variable", "x"), rtol=float(item.get("tolerance") or 0.01))
    except (TypeError, ValueError):
        return False
    raise ValueError(f"Unknown answer_type {kind!r}")


# ---------------------------------------------------------------------------- graph
def _graph_dot(graph: Dict[str, Any]) -> str:
    lines = ["digraph mechanism {", "  rankdir=LR;"]
    for node in graph.get("nodes", []):
        lines.append(f'  "{node["id"]}" [shape=box, label="{node["id"]}\\n({node.get("kind", "")})"];')
    for e in graph.get("edges", []):
        colour = "darkgreen" if e.get("sign") == "+" else "firebrick" if e.get("sign") == "-" else "black"
        label = f'{e.get("sign", "")} {e.get("parameter", "")}={e.get("value", "")}'.strip()
        lines.append(f'  "{e["source"]}" -> "{e["target"]}" [label="{label}", color={colour}];')
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- main entry
def build_bundle(experiment_dir: str, record, results: Dict[str, Dict[str, Any]], logic, model_config,
                 bundle_cfg: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """Write ``<experiment_dir>/bundle/`` and return its manifest."""
    bundle_dir = os.path.join(experiment_dir, "bundle")
    os.makedirs(bundle_dir, exist_ok=True)
    gt = record.ground_truth or {}
    suffix = (gt.get("observation") or {}).get("suffix", "_observed")
    model_name = type(logic).__name__
    fidelity = record.provenance.get("fidelity")
    files: Dict[str, str] = {}

    elements = logic.figure_elements(results) if callable(getattr(logic, "figure_elements", None)) else None
    plotted = [] if elements else _plotted_series(results, bundle_cfg["figure"], suffix)
    figure_spec = None
    if elements:
        units = next((i["descriptor"].units for i in results.values()
                      if i["descriptor"].role is not None and i["descriptor"].role.value == "parameter"), None)
        figure_spec = _write_diagram(bundle_dir, elements, bundle_cfg["figure"], model_name, units)
        files.update({figure_spec["output"]: PUBLIC, "plot_figure.py": PUBLIC,
                      "elements.csv": PUBLIC if bundle_cfg["public_table"] else PRIVATE})
        caption_hook = getattr(logic, "figure_caption", None)
        caption = caption_hook(results) if callable(caption_hook) else f"Diagram from the {model_name} model."
    elif plotted:
        figure_spec = _write_figure(bundle_dir, results, plotted, bundle_cfg["figure"],
                                    record.experiment_id, model_name, suffix)
        files.update({figure_spec["output"]: PUBLIC, "plot_figure.py": PUBLIC,
                      "table.csv": PUBLIC if bundle_cfg["public_table"] else PRIVATE})
        caption = _caption(plotted, results, gt.get("observation"), model_name, suffix)
    else:
        caption = f"Run of the {model_name} model (no time-series outputs to draw)."
    with open(os.path.join(bundle_dir, "caption.txt"), "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    files["caption.txt"] = PUBLIC

    questions: List[Dict[str, Any]] = []
    if bundle_cfg["qa"]["enabled"]:
        questions = generate_questions(results, plotted, gt, tuple(getattr(logic, "inferable_parameters", ())),
                                       bundle_cfg["qa"], rng)
        model_questions = getattr(logic, "questions", None)
        if callable(model_questions):
            for item in model_questions(results, rng) or []:
                missing = {"question", "answer", "answer_type"} - set(item)
                if missing or item["answer_type"] not in ("number", "integer", "choice", "expression") \
                        or (item["answer_type"] == "expression" and "domain" not in item):
                    raise ValueError(f"{model_name}.questions() returned an invalid item: {item}")
                item = dict(item)
                item.setdefault("template", "model")
                item.setdefault("series", None)
                item.setdefault("tolerance", None)
                item.setdefault("answerable_from", "data")
                item["id"] = f"q{len(questions) + 1}"
                questions.append(item)
        excluded = set(bundle_cfg["qa"].get("exclude") or [])
        unknown = excluded - {q.get("template") for q in questions}
        if unknown:
            import warnings
            warnings.warn(f"bundle.qa.exclude: no question of template(s) {sorted(unknown)} in this run "
                          "(misspelled?)", stacklevel=2)
        if excluded:  # filtered after generation, so the random draws (and other questions) do not change
            questions = [q for q in questions if q.get("template") not in excluded]
            for i, q in enumerate(questions):
                q["id"] = f"q{i + 1}"
        for q in questions:
            q.setdefault("tolerance_rule", "exact" if q.get("tolerance") is None else "set by the model")
        public = [public_question(q) for q in questions]
        with open(os.path.join(bundle_dir, "qa.json"), "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2)
        with open(os.path.join(bundle_dir, "questions.json"), "w", encoding="utf-8") as f:
            json.dump(public, f, indent=2)
        files.update({"qa.json": PRIVATE, "questions.json": PUBLIC})

    graph = logic.mechanism_graph(model_config) if bundle_cfg["graph"] else None
    if graph:
        with open(os.path.join(bundle_dir, "graph.json"), "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, default=float)
        with open(os.path.join(bundle_dir, "graph.dot"), "w", encoding="utf-8") as f:
            f.write(_graph_dot(graph))
        files.update({"graph.json": PRIVATE, "graph.dot": PRIVATE})

    with open(os.path.join(bundle_dir, "description.md"), "w", encoding="utf-8") as f:
        f.write(_description(record, gt, fidelity, model_name, caption))
    files["description.md"] = PRIVATE

    manifest = {
        "experiment_id": record.experiment_id,
        "seed": gt.get("seed"),
        "model": gt.get("experiment_type"),
        "fidelity": (fidelity or {}).get("level"),
        "caption": caption,
        "n_questions": len(questions),
        "figure": ({k: figure_spec[k] for k in ("output", "width", "height", "dpi")}
                   if figure_spec else None),
        "files": {name: {"path": f"bundle/{name}", "visibility": vis,
                         "sha256": file_sha256(os.path.join(bundle_dir, name))}
                  for name, vis in sorted(files.items())},
    }
    with open(os.path.join(bundle_dir, "bundle.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
