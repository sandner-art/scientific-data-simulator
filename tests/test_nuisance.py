# tests/test_nuisance.py - nuisance models, their observation operators, and the packs' options
import numpy as np
import pytest

from scidatasim import nuisance as nu
from scidatasim.engine import simulate

TRANSITS = "scidatasim.experiments.transits.logic.TransitExperiment"
RESONANCES = "scidatasim.experiments.resonances.logic.ResonanceExperiment"


@pytest.mark.parametrize("side", ["low", "high"])
@pytest.mark.parametrize("n", [1.5, 3.0, 400.0])
def test_crystal_ball_normalised_and_cdf_consistent(side, n):
    x = np.linspace(-200, 200, 800001)
    f = nu.crystal_ball_pdf(x, 1.0, 0.5, 1.2, n, side)
    tail_beyond = 0.02 if n < 2 else 2e-3
    assert np.trapezoid(f, x) == pytest.approx(1.0, abs=tail_beyond)
    for q in (-1.0, 0.2, 0.9, 1.6):
        assert nu.crystal_ball_cdf(q, 1.0, 0.5, 1.2, n, side) == pytest.approx(
            np.trapezoid(f[x <= q], x[x <= q]) + (1 - np.trapezoid(f, x)) * (side == "low"), abs=1e-3)
    # the tail is on the requested side
    lo, hi = nu.crystal_ball_pdf([1.0 - 3.0, 1.0 + 3.0], 1.0, 0.5, 1.2, n, side)
    assert (lo > hi) if side == "low" else (hi > lo)


def test_crystal_ball_becomes_gaussian_and_tailed_bins_match_voigt():
    from scidatasim.experiments.resonances.logic import voigt_bins
    e = np.linspace(80, 100, 401)
    assert np.allclose(nu.tailed_bins(e, 91.19, 1.3, {"alpha": 8.0, "n": 3}, 2.4955), voigt_bins(e, 91.19, 2.4955, 1.3),
                       atol=1e-12)
    assert np.allclose(nu.tailed_bins(e, 90.0, 1.3, {"alpha": 8.0, "n": 3}), voigt_bins(e, 90.0, 0.0, 1.3), atol=1e-12)
    tailed = nu.tailed_bins(e, 90.0, 1.3, {"alpha": 1.0, "n": 3})
    centres = 0.5 * (e[1:] + e[:-1])
    assert np.sum(tailed * centres) / tailed.sum() < 90.0 - 0.1  # the radiative tail pulls the mean down
    with pytest.raises(ValueError):
        nu.crystal_ball_pdf(0.0, 0.0, 1.0, 1.0, 1.0)


def test_response_matrix_conserves_and_smears():
    c = np.linspace(1, 10, 91)
    R = nu.response_matrix(c, 0.1, {"alpha": 1, "n": 3})
    assert R.shape == (91, 91) and R[:, 45].sum() == pytest.approx(1.0, abs=2e-3)
    assert R[44, 45] > R[46, 45]  # low tail


def test_efficiencies():
    x = np.array([10.0, 20.0, 30.0])
    assert nu.logistic_efficiency(x, 20, 1)[1] == pytest.approx(0.5)
    assert nu.erf_efficiency(x, 20, 1)[1] == pytest.approx(0.5)
    m = np.array([12.0, 20.0, 26.0, 30.0, 40.0, 70.0])
    e1 = nu.two_body_efficiency(m, (17, 8), pair_pt=0.0, rapidity_sd=0.0, eta_max=10.0, turn_on=0.05)
    # at rest: both legs have pT = (m/2) sin(theta), so the efficiency is sqrt(1 - (2*17/m)^2) above 34 GeV
    expected = np.sqrt(np.clip(1 - (34.0 / m) ** 2, 0, None))
    assert np.allclose(e1, expected, atol=0.02)
    assert np.array_equal(nu.two_body_efficiency(m), nu.two_body_efficiency(m))  # fixed seed: reproducible
    e2 = nu.two_body_efficiency(m)
    assert np.all(np.diff(e2) > 0) and e2[-1] < 1


def test_quasi_periodic_flares_ramps_and_trimming():
    t = np.arange(0, 60, 0.01)
    rng = np.random.default_rng(0)
    q = nu.quasi_periodic(t, rng, 0.01, 5.0, 1e6)
    spec = np.abs(np.fft.rfft(q)) ** 2
    freqs = np.fft.rfftfreq(t.size, 0.01)
    top = freqs[np.argsort(spec)[-3:]]  # the power sits at the harmonics of 1/5 d (random amplitudes per harmonic)
    assert all(min(abs(f - k * 0.2) for k in (1, 2, 3)) < 0.02 for f in top)
    rms = [np.std(nu.quasi_periodic(t, np.random.default_rng(s), 0.01, 5.0, 2.0)) for s in range(20)]
    assert np.mean(rms) == pytest.approx(0.01, rel=0.15)
    f, events = nu.flares(t, np.random.default_rng(1), 0.5, (0.001, 0.02))
    assert 15 <= len(events) <= 50 and np.all(f >= 0) and f.max() <= 0.02 * len(events)
    tg = np.concatenate([t[t < 20], t[t >= 21]])
    r, ramps = nu.restart_ramps(tg, np.random.default_rng(2), 0.005, 0.2)
    assert [round(x["start"], 2) for x in ramps] == [0.0, 21.0]
    assert r[0] == pytest.approx(-0.005) and abs(r[np.searchsorted(tg, 5.0)]) < 1e-9
    keep = nu.trim_after_gaps(tg, 0.5)
    assert not keep[(tg < 0.5) | ((tg >= 21) & (tg < 21.5))].any() and keep[(tg > 1) & (tg < 19)].all()
    assert list(nu.gap_edges(tg, 0.5)) == pytest.approx([0.0, 21.0])


def test_operators_on_any_series():
    base = {"experiment_type": TRANSITS, "seed": 4, "planet": False}
    ops = [{"type": "quasi_periodic", "amplitude": 0.01, "period": 4.9, "combine": "multiply"},
           {"type": "flares", "rate": 0.5, "combine": "multiply"},
           {"type": "ramps", "amplitude": [0.002, 0.004], "sign": "random", "combine": "multiply"}]
    run = simulate({**base, "observation": {"targets": ["flux"], "operators": ops}})
    y = run.results["flux_observed"]["data"]
    assert 0.005 < np.nanstd(y) < 0.03
    again = simulate({**base, "observation": {"targets": ["flux"], "operators": ops}})
    assert np.array_equal(y, again.results["flux_observed"]["data"])
    spec = simulate({"experiment_type": RESONANCES, "seed": 4, "signal": False, "known": ["Z"], "window": [60, 120],
                     "observation": {"targets": ["counts"], "operators": [
                         {"type": "efficiency", "shape": "erf", "threshold": 70, "width": 2},
                         {"type": "response", "relative": 0.01, "tail": {"alpha": 1, "n": 3}}]}})
    obs, lat = spec.results["counts_observed"]["data"], spec.results["counts"]["data"]
    assert obs.sum() < lat.sum() and obs[0] < 0.01 * lat[0]
    with pytest.raises(ValueError):
        simulate({**base, "observation": {"targets": ["flux"], "operators": [{"type": "response", "sigma": 1, "relative": 1}]}})


def test_transit_pack_spots_flares_ramps():
    cfg = {"experiment_type": TRANSITS, "seed": 9, "planet": False,
           "spots": {"amplitude": 0.012, "period": 4.86}, "flares": {"rate": 1.0}, "ramps": {"amplitude": 0.004}}
    run = simulate(cfg)
    assert {"stellar_activity", "systematics"} <= set(run.results)
    assert 0.005 < np.std(run.results["stellar_activity"]["data"]) < 0.03
    sysm = run.results["systematics"]["data"]
    assert sysm[0] == pytest.approx(-0.004) and sysm.min() >= -0.008
    qs = run.logic.questions(run.results, np.random.default_rng(0))
    assert all(q.get("instrument_artifact") for q in qs)
    plain = simulate({"experiment_type": TRANSITS, "seed": 9, "planet": False})
    assert "systematics" not in plain.results
    assert not any("instrument_artifact" in q for q in plain.logic.questions(plain.results, np.random.default_rng(0)))
    with pytest.raises(ValueError):
        simulate({"experiment_type": TRANSITS, "seed": 1, "ramps": {"timescale": 0.2}})


def test_transit_search_trim_removes_ramp_false_alarm():
    from scidatasim.experiments.transits.model import transit_search
    run = simulate({"experiment_type": TRANSITS, "seed": 5, "planet": False, "ramps": {"amplitude": 0.003, "timescale": 0.15},
                    "observation": {"targets": ["flux"], "operators": [{"type": "gaussian", "sigma": 1.5e-4}]}})
    t, y = run.results["time"]["data"], run.results["flux_observed"]["data"]
    assert transit_search(t, y, 1.5e-4)["detected"]
    assert not transit_search(t, y, 1.5e-4, trim=1.0)["detected"]


def test_resonance_pack_trigger_shapes_and_tail():
    base = {"experiment_type": RESONANCES, "seed": 3, "signal": False}
    two = simulate({**base, "trigger": {"shape": "two_body"}})
    eff = two.results["trigger_efficiency"]["data"]
    assert np.all(np.diff(eff) >= -1e-9) and eff[0] < 0.1
    assert "two muons" in two.logic.questions(two.results, None)[0]["question"]
    lie = simulate({**base, "trigger": {"shape": "two_body", "disclosed_as": "logistic"}})
    shown, true = lie.results["trigger_efficiency"]["data"], lie.results["true_trigger_efficiency"]["data"]
    assert np.allclose(true, eff) and not np.allclose(shown, true) and np.max(np.abs(shown - true)) < 0.1
    assert "logistic" in lie.logic.questions(lie.results, None)[0]["question"]
    with pytest.raises(ValueError):
        simulate({**base, "trigger": {"shape": "cubic"}})
    ups = {**base, "known": ["Upsilon(1S)"], "window": [8.5, 11.5], "n_bins": 60}
    g = simulate(ups).results["counts"]["data"]
    t = simulate({**ups, "line_shape": {"tail": {"alpha": 1.0, "n": 3}}}).results["counts"]["data"]
    peak = int(np.argmax(g))
    assert t[peak - 4] > g[peak - 4] and t[peak + 4] <= g[peak + 4] + 1e-9


def test_bump_hunt_with_tailed_templates():
    from scidatasim.experiments.resonances.logic import bump_hunt
    tail = {"alpha": 1.0, "n": 3.0}
    run = simulate({"experiment_type": RESONANCES, "seed": 8, "signal": False, "known": ["Upsilon(1S)"],
                    "known_n": 2e5, "window": [8.5, 11.5], "n_bins": 60, "line_shape": {"tail": tail},
                    "observation": {"targets": ["counts"], "operators": [{"type": "poisson"}]}})
    e, c = run.logic.edges, run.results["counts_observed"]["data"]
    assert bump_hunt(e, c, 0.01, known=[9.4603])["detected"]
    assert not bump_hunt(e, c, 0.01, known=[9.4603], tail=tail)["detected"]


def test_line_shape_mixture_and_scale():
    from scidatasim.experiments.resonances.logic import line_shape_bins, voigt_bins
    e = np.linspace(8.5, 11.5, 61)
    one = line_shape_bins(e, 9.46, 0.0, 0.08, None, [[1.0, 1.0]])
    assert np.allclose(one, voigt_bins(e, 9.46, 0.0, 0.08), atol=1e-14)
    mix = line_shape_bins(e, 9.46, 0.0, 0.08, None, [[2.0, 1.0], [2.0, 2.0]])  # weights are normalised
    assert mix.sum() == pytest.approx(one.sum(), abs=1e-3) and mix.max() < one.max()
    with pytest.raises(ValueError):
        line_shape_bins(e, 9.46, 0.0, 0.08, None, [[1.0, -1.0]])
    base = {"experiment_type": RESONANCES, "seed": 1, "signal": False, "known": ["Upsilon(1S)"], "window": [8.5, 11.5],
            "n_bins": 60}
    c0 = simulate(base).results["counts"]["data"]
    c1 = simulate({**base, "line_shape": {"scale": -0.01}}).results["counts"]["data"]
    centres = 0.5 * (e[1:] + e[:-1])
    assert centres[np.argmax(c1)] < centres[np.argmax(c0)]
    with pytest.raises(ValueError):
        simulate({**base, "line_shape": {"scale": 0.5}})
    R1 = nu.response_matrix(centres, 0.05)
    R2 = nu.response_matrix(centres, 0.05, mixture=[[0.5, 1.0], [0.5, 3.0]])
    assert R2[30, 30] < R1[30, 30] and R2[:, 30].sum() == pytest.approx(1.0, abs=2e-3)


def test_bump_hunt_with_mixture_templates():
    from scidatasim.experiments.resonances.logic import bump_hunt
    mixture = [[0.5, 1.0], [0.5, 2.2]]
    run = simulate({"experiment_type": RESONANCES, "seed": 8, "signal": False, "known": ["Upsilon(1S)"], "resolution": 0.008,
                    "known_n": 2e5, "window": [8.5, 11.5], "n_bins": 60, "line_shape": {"mixture": mixture},
                    "observation": {"targets": ["counts"], "operators": [{"type": "poisson"}]}})
    e, c = run.logic.edges, run.results["counts_observed"]["data"]
    assert bump_hunt(e, c, 0.008, known=[9.4603])["detected"]
    assert not bump_hunt(e, c, 0.008, known=[9.4603], mixture=mixture)["detected"]


def test_momentum_dumps_and_quality_flags():
    t = np.arange(0, 27, 2 / 1440)
    x, times = nu.momentum_dumps(t, np.random.default_rng(0), 2.5, 3e-4, 0.1, first=1.0)
    assert np.allclose(times, np.arange(1.0, 27, 2.5)) and np.max(np.abs(x)) <= 3e-4 * 2
    run = simulate({"experiment_type": TRANSITS, "seed": 2, "planet": False, "dumps": {"amplitude": [1e-4, 4e-4]}})
    q = run.results["quality"]["data"]
    assert 9 <= np.sum(q == 32) <= 12 and set(np.unique(q)) <= {0, 32}
    assert all(qq.get("instrument_artifact") for qq in run.logic.questions(run.results, np.random.default_rng(0)))


def test_local_quadratic_detrend_and_red_noise():
    from scidatasim.experiments.transits.model import Planet, a_over_rstar, detrend, red_noise
    rng = np.random.default_rng(4)
    t = np.arange(0, 20, 2 / 1440)
    star = 1 + 0.02 * np.sin(2 * np.pi * t / 4.0)  # a spotted star: 2 % at 4 d
    planet = Planet(3.3, 1.1, 0.04, a_over_rstar(3.3), 0.1)
    y = star * planet.flux(t) * (1 + rng.normal(0, 3e-4, t.size))
    out_of_transit = planet.flux(t) == 1
    med = detrend(t, y, 0.5)
    lq = detrend(t, y, 0.75, "local_quadratic")
    assert np.std(lq[out_of_transit]) < np.std(med[out_of_transit])  # follows the curvature
    inside = planet.flux(t) < 1 - 0.8 * (1 - planet.flux(t).min())
    assert 1 - np.median(lq[inside]) > 0.7 * (1 - planet.flux(t).min())  # keeps most of the depth
    with pytest.raises(ValueError):
        detrend(t, y, 0.5, "spline")
    sw, sr = red_noise(t, 1 + rng.normal(0, 3e-4, t.size), (0.04, 0.2))
    assert sw == pytest.approx(3e-4, rel=0.05) and max(sr.values()) < 3e-5
    ar = np.empty(t.size)
    ar[0] = 0
    z = rng.normal(0, 1, t.size)
    for i in range(1, t.size):
        ar[i] = 0.98 * ar[i - 1] + np.sqrt(1 - 0.98 ** 2) * z[i]
    _, sr2 = red_noise(t, 1 + 3e-4 * ar, (0.04, 0.2))
    assert sr2[0.04] > 5e-5


def test_red_noise_search_and_second_baseline():
    from scidatasim.experiments.transits.model import transit_search, transit_search_robust
    clean = simulate({"experiment_type": TRANSITS, "seed": 6, "planet": True, "orbit": {"period": 3.3, "rp": 0.03},
                      "observation": {"targets": ["flux"], "operators": [{"type": "gaussian", "sigma": 4e-4}]}})
    t, y = clean.results["time"]["data"], clean.results["flux_observed"]["data"]
    r1, r2 = transit_search(t, y, 4e-4), transit_search_robust(t, y, 4e-4)
    assert r1["detected"] and r2["detected"] and abs(r2["period"] - 3.3) < 0.02
    assert r2["snr"] <= r2["snr_white"] + 1e-9 and set(r2["red_noise_ppm"]) == {0.04, 0.08, 0.12, 0.2}
    red = simulate({"experiment_type": TRANSITS, "seed": 7, "planet": False, "observation": {"targets": ["flux"], "operators": [
        {"type": "gaussian", "sigma": 4e-4}, {"type": "ar1", "sigma": 4e-4, "length": 0.05}]}})
    t, y = red.results["time"]["data"], red.results["flux_observed"]["data"]
    w, rr = transit_search(t, y, 4e-4), transit_search(t, y, 4e-4, noise="red")
    assert rr["snr"] < w["snr"]
    with pytest.raises(ValueError):
        transit_search(t, y, 4e-4, noise="pink")
