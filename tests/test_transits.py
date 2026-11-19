"""Transit light curves: geometry, limb darkening, the experiment, its labels and the baseline."""
import numpy as np
import pytest

from scidatasim.engine import simulate
from scidatasim.experiments.transits.logic import SNR_THRESHOLD, TransitExperiment
from scidatasim.experiments.transits.model import (Planet, a_over_rstar, blocked_fraction, overlap_area,
                                                   projected_separation, transit_search)

EXP = "scidatasim.experiments.transits.logic.TransitExperiment"
WHITE = {"targets": ["flux"], "operators": [{"type": "gaussian", "sigma": 0.001}]}


def test_overlap_area_closed_form_against_monte_carlo():
    assert overlap_area(1.0, 0.1, 0.0) == pytest.approx(np.pi * 0.01)
    assert overlap_area(1.0, 0.1, 1.2) == 0.0
    assert overlap_area(0.05, 0.1, 0.0) == pytest.approx(np.pi * 0.0025)
    rng = np.random.default_rng(0)
    for p, z in ((0.3, 0.9), (0.1, 1.0), (0.5, 0.6)):
        pts = rng.uniform(-1, 1, (1_000_000, 2)) * p + [z, 0.0]  # the box around the planet
        inside = (np.hypot(*pts.T) < 1) & (np.hypot(pts[:, 0] - z, pts[:, 1]) < p)
        assert overlap_area(1.0, p, z) == pytest.approx(inside.mean() * 4 * p * p, rel=0.005)


def test_limb_darkening_limits_and_convergence():
    p, u1, u2 = 0.01, 0.4, 0.25
    centre = blocked_fraction([0.0], p, u1, u2)[0]
    assert centre == pytest.approx(p ** 2 / (1 - u1 / 3 - u2 / 6), rel=2e-3)  # small planet, I(0) = 1
    assert blocked_fraction([0.0], 0.1)[0] == pytest.approx(0.01)  # uniform star: exact
    z = np.linspace(0, 1.08, 37)
    ref = blocked_fraction(z, 0.1, u1, u2, 6400)
    e1 = np.max(np.abs(blocked_fraction(z, 0.1, u1, u2, 200) - ref))
    e2 = np.max(np.abs(blocked_fraction(z, 0.1, u1, u2, 400) - ref))
    assert e2 < e1 / 2 and e2 < 2e-6
    assert blocked_fraction([0.95], 0.1, u1, u2)[0] < blocked_fraction([0.0], 0.1, u1, u2)[0]  # darker limb


def test_orbit_geometry_and_duration():
    assert a_over_rstar(365.25) == pytest.approx(215, rel=0.01)
    planet = Planet(3.0, 1.0, 0.1, a_over_rstar(3.0), 0.3)
    t = np.linspace(0.5, 1.5, 200001)
    inside = planet.in_transit(t)
    measured = t[inside].max() - t[inside].min()
    assert planet.duration == pytest.approx(measured, rel=1e-3)
    assert projected_separation([1.0 + 1.5], 3.0, 1.0, planet.a, 0.3)[0] > 1e8  # behind the star
    assert Planet(3.0, 1.0, 0.1, 10.0, 1.2).duration == 0.0
    assert planet.transits_in(np.arange(0, 10, 0.01)) == 4


def test_experiment_reproducible_and_answers():
    cfg = dict(experiment_type=EXP, planet=True, orbit={"period": 3.3, "t0": 1.0, "rp": 0.05, "b": 0.2},
               observation=WHITE)
    a, b = simulate(dict(cfg, seed=4)), simulate(dict(cfg, seed=4))
    assert np.array_equal(a.results["flux_observed"]["data"], b.results["flux_observed"]["data"])
    items = {q["template"]: q for q in a.logic.questions(a.results, np.random.default_rng(0))}
    assert items["transit_present"]["answer"] == "yes" and items["transit_period"]["answer"] == pytest.approx(3.3)
    assert items["transit_depth"]["answer"] == pytest.approx(1e6 * blocked_fraction(
        [0.2], 0.05, 0.4, 0.25)[0], rel=1e-3)
    gt = a.ground_truth["parameters"]
    assert gt["planet.n_transits"] >= 7 and gt["planet.snr"] > SNR_THRESHOLD
    found = transit_search(a.results["time"]["data"], a.results["flux_observed"]["data"], 1e-3)
    assert found["detected"] and abs(found["period"] - 3.3) <= items["transit_period"]["tolerance"]
    none = simulate(dict(experiment_type=EXP, planet=False, seed=4, observation=WHITE))
    items = {q["template"]: q for q in none.logic.questions(none.results, np.random.default_rng(0))}
    assert items["transit_period"]["answer"] == 0.0 and items["transit_depth"]["answer"] == 0.0
    assert items["transit_period"]["tolerance_rule"] == {q["template"]: q for q in a.logic.questions(
        a.results, np.random.default_rng(0))}["transit_period"]["tolerance_rule"]
    with pytest.raises(ValueError):
        TransitExperiment({"planet": "maybe"})
    with pytest.raises(ValueError):
        TransitExperiment({"orbit": {"radius": 1}})


def test_labels():
    tiny = TransitExperiment({"seed": 1, "planet": True, "orbit": {"rp": 0.003, "period": 3.0}})
    tiny.initialize({"seed": 1})
    single = TransitExperiment({"seed": 1, "planet": True, "orbit": {"rp": 0.05, "period": 25.0, "t0": 5.0}})
    single.initialize({"seed": 1})
    q_tiny = {q["template"]: q for q in tiny.questions(tiny.get_results(), None)}
    q_single = {q["template"]: q for q in single.questions(single.get_results(), None)}
    assert q_tiny["transit_present"]["signal_undetectable"] and not q_tiny["transit_period"]["period_ambiguous"]
    assert q_single["transit_period"]["period_ambiguous"] and not q_single["transit_present"]["signal_undetectable"]


def test_planted_into_a_real_light_curve(tmp_path):
    """real_noise: the transits multiplied into a measured light curve, by time."""
    rng = np.random.default_rng(5)
    t = np.arange(0, 30, 2 / 1440)
    real = 1.0 + 1e-3 * np.convolve(rng.normal(0, 1, t.size + 4), np.ones(5) / np.sqrt(5), "valid")[: t.size]
    np.savez(tmp_path / "lc.npz", time=t, flux=real)
    obs = {"targets": ["flux"], "operators": [{"type": "real_noise", "source": str(tmp_path / "lc.npz"),
                                               "time": "time", "column": "flux", "combine": "multiply"}]}
    run = simulate(dict(experiment_type=EXP, seed=2, planet=True, gap=None, span=20.0,
                        orbit={"period": 4.1, "t0": 0.7, "rp": 0.06, "b": 0.1}, observation=obs))
    y = run.results["flux_observed"]["data"]
    info = run.ground_truth["observation"]["targets"]["flux"]["operators"][0]
    assert info["align"] == "time" and np.isfinite(y).all()
    found = transit_search(run.results["time"]["data"], y, 1e-3)
    assert found["detected"] and found["period"] == pytest.approx(4.1, rel=0.01)
