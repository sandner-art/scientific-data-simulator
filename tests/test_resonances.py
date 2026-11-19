"""Resonances: spectrum shapes, the experiment, the counting fit and the bump hunt."""
import numpy as np
import pytest
from scipy import stats

from scidatasim.baselines.counting import BinnedFit, asimov_z, gaussian_bins, global_significance, poisson_nll
from scidatasim.engine import simulate
from scidatasim.experiments.resonances.logic import (KNOWN, ResonanceExperiment, background_bins, bump_hunt,
                                                     efficiency, voigt_bins)

EXP = "scidatasim.experiments.resonances.logic.ResonanceExperiment"
POISSON = {"targets": ["counts"], "operators": [{"type": "poisson"}]}


def test_shapes_against_closed_forms():
    edges = np.linspace(80, 100, 201)
    assert voigt_bins(edges, 90.0, 0.0, 1.0) == pytest.approx(np.diff(stats.norm.cdf(edges, 90, 1)))
    bw = voigt_bins(edges, 91.1876, 2.4955, 1e-4)  # nearly a pure Breit-Wigner
    cauchy = np.diff(stats.cauchy.cdf(edges, 91.1876, 2.4955 / 2))
    assert np.max(np.abs(bw - cauchy)) < 2e-4
    b = background_bins(np.linspace(12, 70, 117), 3.0, 0.01)
    assert b.sum() == pytest.approx(1.0) and np.all(np.diff(b) < 0)
    assert efficiency([20.0], 20.0, 1.5)[0] == pytest.approx(0.5)
    assert asimov_z(10, 100) == pytest.approx(np.sqrt(2 * (110 * np.log(1.1) - 10)))


def test_experiment_signal_strength_and_questions():
    run = simulate(dict(experiment_type=EXP, seed=1, signal=True, z=5.0, mass=40.0, observation=POISSON))
    L = run.logic
    assert L.asimov() == pytest.approx(5.0, rel=1e-6)
    items = {q["template"]: q for q in L.questions(run.results, None)}
    assert items["resonance_mass"]["answer"] == 40.0 and not items["new_resonance"]["signal_undetectable"]
    assert 0.25 <= items["resonance_mass"]["tolerance"] < 2.0
    none = simulate(dict(experiment_type=EXP, seed=1, signal=False, observation=POISSON))
    q = {q["template"]: q for q in none.logic.questions(none.results, None)}
    assert q["resonance_mass"]["answer"] == 0.0 and q["new_resonance"]["answer"] == "no"
    assert q["resonance_mass"]["tolerance_rule"] == items["resonance_mass"]["tolerance_rule"]
    weak = simulate(dict(experiment_type=EXP, seed=1, signal=True, z=1.0, observation=POISSON)).logic
    assert {q["template"]: q for q in weak.questions({}, None)}["new_resonance"]["signal_undetectable"]
    ups = ResonanceExperiment({"window": [8.0, 12.0], "known": ["Upsilon(1S)", "Upsilon(2S)", "Upsilon(3S)", "Z"]})
    assert [k["name"] for k in ups.known] == ["Upsilon(1S)", "Upsilon(2S)", "Upsilon(3S)"]
    assert KNOWN["Z"][0] == pytest.approx(91.1876)
    with pytest.raises(ValueError):
        ResonanceExperiment({"known": ["X(3872)"]})
    with pytest.raises(ValueError):
        ResonanceExperiment({"signal": "maybe"})


def test_binned_fit_recovers_background_and_signal():
    rng = np.random.default_rng(0)
    edges = np.linspace(12, 70, 117)
    truth = 2e5 * background_bins(edges, 3.0, 0.01)
    fit = BinnedFit(edges, degree=3)
    counts = rng.poisson(truth)
    b = fit.fit(counts)
    assert b["converged"] and np.sum((counts - b["expected"]) ** 2 / b["expected"]) / 112 < 1.5
    sig = 300 * gaussian_bins(edges, 45.0, 0.45)
    counts = rng.poisson(truth + sig)
    scan = fit.scan(counts, lambda m: 0.01 * m, masses=fit.centres[10:-10])
    assert scan["best_mass"] == pytest.approx(45.0, abs=0.75) and scan["best_yield"] == pytest.approx(300, rel=0.35)
    g = global_significance(scan)
    assert g["z_global"] < g["z_local"] and g["p_global"] > g["p_local"]
    assert poisson_nll([0, 1], [1.0, 1.0]) == pytest.approx(2.0)


def test_bump_hunt_and_the_trigger():
    run = simulate(dict(experiment_type=EXP, seed=7, signal=True, z=6.5, mass=50.0, observation=POISSON))
    L = run.logic
    found = bump_hunt(L.edges, run.results["counts_observed"]["data"], L.resolution)
    q = {q["template"]: q for q in L.questions(run.results, None)}["resonance_mass"]
    assert found["detected"] and abs(found["mass"] - 50.0) <= q["tolerance"]
    trig = simulate(dict(experiment_type=EXP, seed=8, signal=False, trigger={"threshold": 20.0, "width": 1.5},
                         observation=POISSON))
    L = trig.logic
    counts = trig.results["counts_observed"]["data"]
    naive = bump_hunt(L.edges, counts, L.resolution)
    aware = bump_hunt(L.edges, counts, L.resolution, efficiency=L.eff)
    assert naive["detected"] and 18 < naive["mass"] < 26 and not aware["detected"]
    assert "trigger efficiency rises around 20" in L.questions(trig.results, None)[0]["question"]
