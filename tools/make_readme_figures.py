"""Figures for the README (docs/images/), made with the simulator itself so they can be regenerated.

    python tools/make_readme_figures.py            # all figures, about a minute
    python tools/make_readme_figures.py --only fields styles

Needs the dev profile (plotting, Parquet) and Mesa for the agent-based figures.
Colours: the first three slots of a validated categorical palette (blue, orange, aqua: colour-blind
safe for all pairs); aqua is low-contrast on white, so its series always carry a direct label.
"""
import argparse
import glob
import os
import sys
import tempfile

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.environ.get("SDS_FIGURES_OUT") or os.path.join(ROOT, "docs", "images")  # override: tests

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#9cc3ef", "#2a78d6", "#123a6b"])

matplotlib.rcParams.update({
    "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": "#b8b7b1",
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "legend.frameon": False, "lines.linewidth": 2.0, "savefig.facecolor": SURFACE,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", metadata={"Software": None})
    print(f"{path} ({os.path.getsize(path) // 1024} KB)")


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", color=INK, fontsize=9,
                va="center", ha="left")
    ax.plot([x], [y], "o", color=color, markersize=4)


# ---------------------------------------------------------------------------- 1 scorer loophole
def loophole():
    from scidatasim.diagnostics import polynomial_formula
    from scidatasim.engine import simulate
    from scidatasim.experiments.law_discovery.logic import FORMULA_EXTENSION, fit_law, law_expression
    from scidatasim.expressions import evaluate
    sim = simulate({"experiment_type": "scidatasim.experiments.law_discovery.logic.HiddenLawExperiment",
                    "law": "michaelis_menten", "seed": 11, "n_points": 40, "generate_plots": False,
                    "observation": {"operators": [{"type": "gaussian", "snr_db": 30}]}})
    x, y = sim.results["x"]["data"], sim.results["y_observed"]["data"]
    lo, hi = sim.logic.domain
    end = hi + FORMULA_EXTENSION * (hi - lo)
    grid = np.linspace(lo, end, 300)
    truth = evaluate(sim.results["law_expression"]["data"], x=grid)
    poly = evaluate(polynomial_formula(x, y), x=grid)
    law = evaluate(law_expression("michaelis_menten", fit_law("michaelis_menten", x, y)), x=grid)
    fig = Figure(figsize=(7.2, 3.6))
    ax = fig.add_subplot()
    ax.axvspan(hi, end, color="#eeede9", zorder=0, lw=0)
    ax.plot(x, y, "o", color=BLUE, markersize=4, alpha=0.85, zorder=3)
    ax.plot(grid, truth, color=INK2, lw=1.2, zorder=2)
    ax.plot(grid, law, color=ORANGE, zorder=4)
    ax.plot(grid, poly, color=AQUA, ls=(0, (5, 2)), zorder=4)
    top = max(truth.max(), y.max()) * 1.25
    ax.set_ylim(min(0, poly.min(), y.min()) - 0.05 * top, top)
    ax.set_xlim(lo, end + 2.9)
    ax.text(end - 0.12, top * 0.04, "scored beyond the data", color=INK2, fontsize=8.5, va="bottom", ha="right")
    ax.text(x[3], y.max() * 0.25, "observations (30 dB)", color=BLUE, fontsize=8.5)
    label_end(ax, end, law[-1], "fitted law: passes", ORANGE, dy=7)
    label_end(ax, end, truth[-1], "true law", INK2, dy=-8)
    exits = np.flatnonzero(poly > top)
    i = exits[0] - 1 if exits.size else len(grid) - 1
    ax.annotate("fitted polynomial: fails", (grid[i], poly[i]), xytext=(-165, -26), textcoords="offset points",
                color=INK, fontsize=9, va="center",
                arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8, "shrinkB": 3})
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Formula questions are scored beyond the data, where curve fitting without the law fails")
    save(fig, "scorer_loophole.png")


# ---------------------------------------------------------------------------- 2 series operators
def _series():
    from scidatasim.utils import DataDescriptor, DataType, Role
    x = np.linspace(0, 10, 120)
    y = 3 + 1.5 * np.sin(0.9 * x) + 0.15 * x
    return {"x": {"data": x, "descriptor": DataDescriptor("x", DataType.NDARRAY, role=Role.COORDINATE)},
            "y": {"data": y, "descriptor": DataDescriptor("y", DataType.NDARRAY, role=Role.LATENT, x_axis="x")}}


def series_operators():
    from scidatasim.observation import apply_observation
    panels = [
        ("Independent noise", [{"type": "gaussian", "sigma": 0.35}], None),
        ("Correlated noise (AR(1), rho 0.9)", [{"type": "ar1", "sigma": 0.35, "rho": 0.9}], None),
        ("Drift and a calibration error", [{"type": "calibration", "gain": 0.9, "offset": 0.3},
                                           {"type": "drift", "rate": 0.12}, {"type": "gaussian", "sigma": 0.1}], None),
        ("Detection limit (reported as the limit)", [{"type": "gaussian", "sigma": 0.15},
                                                     {"type": "censor", "low": 2.5, "replace": "limit"}], 2.5),
        ("High values missing (MNAR, 30 %)", [{"type": "gaussian", "sigma": 0.15},
                                              {"type": "missing", "rate": 0.3, "mechanism": "mnar", "strength": 3}],
         None),
        ("Outliers (5 %) and quantisation", [{"type": "gaussian", "sigma": 0.15},
                                             {"type": "outliers", "rate": 0.05, "scale": 1.2},
                                             {"type": "quantize", "step": 0.25}], None),
    ]
    fig = Figure(figsize=(10.5, 5.2))
    axes = fig.subplots(2, 3, sharex=True, sharey=True)
    for ax, (title, ops, limit) in zip(axes.ravel(), panels):
        results = _series()
        added, _ = apply_observation(results, {"operators": ops}, np.random.default_rng(7))
        ax.plot(results["x"]["data"], results["y"]["data"], color=INK2, lw=1.2)
        ax.plot(results["x"]["data"], added["y_observed"]["data"], "o", color=BLUE, markersize=3.2, alpha=0.9)
        if limit is not None:
            ax.axhline(limit, color=ORANGE, lw=1.2, ls=(0, (4, 2)))
            ax.text(9.9, limit - 0.15, "limit", color=INK2, fontsize=8, ha="right", va="top")
        ax.set_title(title, fontsize=9.5)
    axes[0, 0].text(6.6, 1.0, "grey: exact\nblue: observed", color=INK2, fontsize=8)
    fig.suptitle("The measurement model: the same exact series, observed six ways", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "series_operators.png")


# ---------------------------------------------------------------------------- 3 field operators
def field_operators():
    from scidatasim.observation import apply_observation
    from scidatasim.utils import DataDescriptor, DataType, Role
    n = 96
    yy, xx = np.mgrid[0:n, 0:n] / n
    exact = 0.15 + 0.5 * (np.sin(18 * xx) > 0.6) * (yy < 0.35)  # bars of a resolution target
    for cx, cy, r in ((0.3, 0.68, 0.10), (0.68, 0.62, 0.06), (0.78, 0.82, 0.03)):
        exact += 0.85 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))
    panels = [
        ("Exact field", []),
        ("Point-spread blur (sigma 2 px)", [{"type": "psf_blur", "sigma": 2.0}]),
        ("Camera: photon and read noise, 8 bit", [{"type": "detector", "scale": 60, "read_noise": 4, "bits": 8}]),
        ("Vignetting and misregistration", [{"type": "vignetting", "strength": 0.45},
                                           {"type": "shift", "dx": 6.5, "dy": -4.0}]),
        ("Binned 4 x 4", [{"type": "bin", "factor": 4}]),
        ("Dead and hot pixels", [{"type": "missing", "rate": 0.02}, {"type": "outliers", "rate": 0.01, "scale": 8}]),
    ]
    fig = Figure(figsize=(8.2, 5.9))
    axes = fig.subplots(2, 3)
    cmap = SEQUENTIAL.copy()
    cmap.set_bad("#e34948")
    for ax, (title, ops) in zip(axes.ravel(), panels):
        results = {"u": {"data": exact.copy(), "descriptor": DataDescriptor("u", DataType.NDARRAY, role=Role.LATENT)}}
        data = exact if not ops else apply_observation(results, {"targets": ["u"], "operators": ops},
                                                       np.random.default_rng(3))[0]["u_observed"]["data"]
        ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=1.1, interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.suptitle("Fields are observed like a camera sees them (dead pixels in red)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "field_operators.png")


# ---------------------------------------------------------------------------- 4 figure styles
def figure_styles():
    import pandas as pd
    from PIL import Image
    from scidatasim.config import load_config
    from scidatasim.doe import cross_styles, run_design
    base = load_config(os.path.join(ROOT, "examples", "example_10_law_discovery", "config.yaml"))
    styles = {"default": {}, "minimal": {"style": "minimal"}, "dense (clutter)": {"style": "dense"},
              "print (grayscale)": {"style": "print"}}
    with tempfile.TemporaryDirectory() as tmp:
        table = cross_styles(pd.DataFrame({"law": ["logistic"], "counterfactual": [True]}), styles, base_seed=4)
        out = run_design(base, table, output_dir=tmp)
        fig = Figure(figsize=(10.5, 7.4))
        axes = fig.subplots(2, 2)
        for ax, row in zip(axes.ravel(), out.table.to_dict("records")):
            path = glob.glob(os.path.join(tmp, f"*{row['experiment_id']}*", "bundle", "figure.png"))[0]
            ax.imshow(np.asarray(Image.open(path).convert("RGB")))
            ax.set_title(row["bundle.figure.variant"], fontsize=10)
            ax.axis("off")
    fig.suptitle("The same run - same data, same questions - drawn in four styles (cross_styles)", x=0.01,
                 ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "figure_styles.png")


# ---------------------------------------------------------------------------- 5 agent-based
def agent_based():
    from scidatasim.engine import simulate
    pack = "scidatasim.experiments.social.logic."
    seg0 = simulate({"experiment_type": pack + "SegregationExperiment", "seed": 5, "n_steps": 0,
                     "homophily": 0.45, "generate_plots": False})
    seg = simulate({"experiment_type": pack + "SegregationExperiment", "seed": 5, "n_steps": 40,
                    "homophily": 0.45, "generate_plots": False})
    epi = simulate({"experiment_type": pack + "EpidemicTrialExperiment", "seed": 3, "n_steps": 160,
                    "effect": 0.5, "beta": 0.04, "generate_plots": False})
    fig = Figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    groups = ListedColormap([SURFACE, BLUE, ORANGE])
    for i, (sim, title) in enumerate(((seg0, "Schelling: start"), (seg, "after 40 steps (homophily 0.45)"))):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(sim.results["grid"]["data"] + 1, cmap=groups, vmin=0, vmax=2, interpolation="nearest")
        share = sim.results["segregation"]["data"][-1]
        ax.set_title(f"{title}\nsimilar neighbours {share:.0%}", fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    ax = fig.add_subplot(gs[0, 2])
    r = epi.results
    steps = r["step"]["data"]
    for arm, color, text in (("treated", ORANGE, "treated (effect 0.5)"), ("control", BLUE, "control")):
        sus = r[f"susceptible_{arm}"]["data"]
        infected = 1 - sus / sus[0]
        ax.plot(steps, infected, color=color)
        label_end(ax, steps[-1], infected[-1], text, color)
    ax.set_xlim(0, steps[-1] * 1.35)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("step")
    ax.set_title("Epidemic with a randomised trial: share infected", fontsize=9.5)
    fig.suptitle("Agent-based models with Mesa, each with a planted effect", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.text(0.01, -0.02, "Schelling grids: blue majority, orange minority, white empty.", color=INK2, fontsize=8.5)
    fig.tight_layout()
    save(fig, "agent_based.png")


# ---------------------------------------------------------------------------- 6 simulation study
def simulation_study():
    import pandas as pd
    sys.path.insert(0, ROOT)
    from examples.example_15_agent_based.run_experiment import (TRIAL, attack_rate_ratio, hazard_ratio,
                                                                 stratified_hazard_ratio)
    from scidatasim.ademp import SimulationStudy
    rows = [{"effect": e, "beta": b} for b in (0.02, 0.05) for e in (0.3, 0.6)]
    labels = [f"effect {e}, {'small' if b == 0.02 else 'large'} epidemic" for b in (0.02, 0.05) for e in (0.3, 0.6)]
    methods = {"attack-rate ratio": attack_rate_ratio, "hazard ratio": hazard_ratio,
               "stratified hazard ratio": stratified_hazard_ratio}
    result = SimulationStudy(aims="README figure", base_config=TRIAL, mechanisms=pd.DataFrame(rows),
                             estimands={"effect": "effect"}, methods=methods, n_sim=100, base_seed=15,
                             mechanism_labels=labels).run()
    perf = result.performance
    fig = Figure(figsize=(8.4, 3.8))
    ax = fig.add_subplot()
    colors = dict(zip(methods, (ORANGE, BLUE, AQUA)))
    offsets = dict(zip(methods, (-0.22, 0.0, 0.22)))
    for name in methods:
        part = perf[perf["method"] == name].sort_values("design_point")
        ypos = part["design_point"].to_numpy() + offsets[name]
        ax.errorbar(part["bias"], ypos, xerr=1.96 * part["bias_mcse"], fmt="o", color=colors[name],
                    markersize=6, elinewidth=1.4, capsize=0, label=name)
    ax.axvline(0, color=INK2, lw=1.0)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("bias of the estimated effect (95 % Monte Carlo interval)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, fontsize=8.5, handletextpad=0.3)
    ax.set_title("Which estimator recovers a planted treatment effect? (100 trials per row)", pad=26)
    row = perf[perf["design_point"] == 2]
    for name in methods:  # direct labels on the third row, where the estimators are far apart
        part = row[row["method"] == name]
        b, m = float(part["bias"].iloc[0]), float(part["bias_mcse"].iloc[0])
        ax.annotate(name, (b - 1.96 * m, 2 + offsets[name]), xytext=(-6, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=8, color=INK)
    ax.text(0.0, -0.62, " no bias", color=INK2, fontsize=8, va="bottom")
    save(fig, "simulation_study.png")


FIGURES = {"loophole": loophole, "series": series_operators, "fields": field_operators,
           "styles": figure_styles, "agents": agent_based, "study": simulation_study}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    args = parser.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name]()
