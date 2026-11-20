# tests/test_validation_tools.py - the open-data validations of the transit and resonance packs
import importlib.util
import os

import numpy as np
import pytest

from scidatasim.baselines.counting import global_significance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "tools", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_significance_is_finite_for_huge_excesses():
    scan = {"best_q0": 9e4, "q0": np.array([0.0, 2.0, 9e4, 1.0, 0.0])}
    g = global_significance(scan)
    assert np.isfinite(g["z_global"]) and g["z_global"] == pytest.approx(300.0, rel=0.01)
    small = global_significance({"best_q0": 16.0, "q0": np.array([0.0, 16.0, 0.0])})
    assert 0 < small["z_global"] < 4.0 and small["p_global"] > small["p_local"]


def _histogram(tmp_path, rng):
    from scidatasim.experiments.resonances.logic import background_bins, voigt_bins
    vc = _tool("validate_cms_dimuon")
    e = vc.EDGES
    mu = 3e6 * background_bins(e, 3.0, 0.01) + 1e5 * voigt_bins(e, 3.0969, 0.0, 0.031) \
        + 2e4 * voigt_bins(e, 9.4603, 0.0, 0.09)
    path = tmp_path / "h.csv"
    with open(path, "w") as f:
        f.write("low,high,counts\n")
        for lo, hi, c in zip(e[:-1], e[1:], rng.poisson(mu)):
            f.write(f"{lo:.2f},{hi:.2f},{c}\n")
    return vc, str(path)


def test_cms_peak_fit_recovers_mass_and_resolution(tmp_path):
    vc, path = _histogram(tmp_path, np.random.default_rng(1))
    edges, counts, prov = vc.load(path)
    assert edges.size == counts.size + 1 and prov["sha256"]
    row = vc.fit_peaks(edges, counts, 2.8, 3.4, ["J/psi"])[0]
    assert abs(row["mass"] - 3.0969) < 4 * row["mass_error"] + 1e-4
    assert row["resolution"] == pytest.approx(0.031, rel=0.05)
    e, c = vc.window(edges, counts, 12.0, 70.0, rebin=10)
    assert e.size == c.size + 1 and np.allclose(np.diff(e), 0.5)


def test_cms_histogram_from_nanoaod(tmp_path):
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward")
    vc = _tool("validate_cms_dimuon")
    # back-to-back opposite-charge muons of 45 GeV (mass 90), a same-sign pair, three muons, a 10 GeV pair, one muon
    n = [2, 2, 3, 2, 1]
    mu = ak.zip({"pt": ak.Array([[45, 45], [45, 45], [10, 10, 10], [5, 5], [3]]),
                 "eta": ak.Array([[0.0] * k for k in n]),
                 "phi": ak.Array([[0, np.pi], [0, np.pi], [0, 1, 2], [0, np.pi], [0]]),
                 "mass": ak.Array([[0.1057] * k for k in n]),
                 "charge": ak.Array([[1, -1], [1, 1], [1, -1, 1], [-1, 1], [1]])})
    root = str(tmp_path / "t.root")
    with uproot.recreate(root) as f:
        f.mktree("Events", {"Muon": mu.type.content}, field_name=lambda o, i: o + "_" + i, counter_name=lambda c: "n" + c)
        f["Events"].extend({"Muon": mu})
    edges, counts, _ = vc.load(vc.make_histogram(root, str(tmp_path / "h.csv")))
    assert counts.sum() == 2
    assert set(np.round(edges[np.nonzero(counts)[0]], 2)) == {10.0, 90.0}


def test_tess_helpers():
    vt = _tool("validate_tess")
    assert vt.phase_distance([10.0, 10.5, 11.9], 2.0, 0.0).tolist() == pytest.approx([0.0, 0.5, 0.1])
    tol = vt.tolerance(5.0, 0.1, 4)
    assert vt.classify(5.01, 5.0, tol) == "period"
    assert vt.classify(2.5, 5.0, tol) == "alias" and vt.classify(10.02, 5.0, tol) == "alias"
    assert vt.classify(3.3, 5.0, tol) is None
    assert vt.near_known(100100827, 0.9416) == "WASP-18 b" and vt.near_known(100100827, 0.4707) == "WASP-18 b"
    assert vt.near_known(100100827, 0.98) is None
    for tic, (host, planets) in vt.HOSTS.items():
        for p in planets:
            assert len(p) == 7 and p[1] > 0 and p[2] > 2.45e6 and p[3] > 0 and p[4] > 0


def test_tess_iterative_search_finds_two_planets_and_masking():
    from scidatasim.experiments.transits.model import Planet, a_over_rstar
    vt = _tool("validate_tess")
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 20.0, 10.0 / 1440)
    p1, p2 = Planet(3.1, 0.7, 0.05, a_over_rstar(3.1), 0.2), Planet(4.7, 1.9, 0.035, a_over_rstar(4.7), 0.1)
    y = p1.flux(t) * p2.flux(t) * (1 + rng.normal(0, 4e-4, t.size))
    rounds = vt.iterative_search(t, y, 4e-4, 4)
    periods = [r["period"] for r in rounds if r["detected"]]
    assert any(abs(p - 3.1) < 0.02 for p in periods) and any(abs(p - 4.7) < 0.03 for p in periods)
    tic = next(iter(vt.HOSTS))
    period, t0, _, dur_h = vt.HOSTS[tic][1][0][1:5]
    tt = np.arange(0.0, 5.0, 2.0 / 1440) + (t0 - vt.BTJD) + 100 * period
    lc = {"tic": tic, "time": tt, "flux": np.ones(tt.size)}
    bg = vt.masked_background(lc)
    gone = ~np.isfinite(bg)
    assert gone.any()
    near = np.minimum(vt.phase_distance(tt, period, t0 - vt.BTJD), vt.phase_distance(tt, period, t0 - vt.BTJD + period / 2))
    assert np.array_equal(gone, near < dur_h / 24)  # transits and secondary eclipses, one duration either side


def test_tess_activity_and_ramp_measures():
    vt = _tool("validate_tess")
    t = np.arange(0.0, 27.0, 2.0 / 1440)
    t = t[(t < 13) | (t >= 14)]
    y = 1 + 0.01 * np.sin(2 * np.pi * t / 4.86)
    lc = {"time": t, "flux": y.copy()}
    act = vt.activity(lc)
    assert act["period"] == pytest.approx(4.86, rel=0.03) and act["rms"] == pytest.approx(0.01 / np.sqrt(2), rel=0.05)
    ramp = np.where(t >= 14, -0.003 * np.exp(-(t - 14) / 0.2), 0.0)
    r = vt.ramp_size({"time": t, "flux": 1 + ramp})
    assert r["amplitude"] == pytest.approx(0.003 * np.exp(-0.05 / 0.2), rel=0.2) and 0.05 <= r["timescale"] < 0.5


def test_cms_tail_fit_recovers_tail(tmp_path):
    from scidatasim.nuisance import tailed_bins
    vc = _tool("validate_cms_dimuon")
    e = vc.EDGES
    from scidatasim.experiments.resonances.logic import background_bins
    mu = 2e6 * background_bins(e, 3.0, 0.01) + 3e5 * tailed_bins(e, 3.0969, 0.031, {"alpha": 1.3, "n": 4.0})
    path = tmp_path / "h.csv"
    with open(path, "w") as f:
        f.write("low,high,counts\n")
        for lo, hi, c in zip(e[:-1], e[1:], np.random.default_rng(2).poisson(mu)):
            f.write(f"{lo:.2f},{hi:.2f},{c}\n")
    edges, counts, _ = vc.load(str(path))
    r = vc.fit_tail(edges, counts, 2.8, 3.4, ["J/psi"])
    assert r["alpha"] == pytest.approx(1.3, rel=0.25) and r["chi2_ndf_tail"] < r["chi2_ndf_gaussian"]
    assert abs(r["peaks"][0]["offset_percent"]) < 0.05


def test_tess_null_arms_use_the_right_flux():
    vt = _tool("validate_tess")
    t = np.arange(8000) * vt.CADENCE
    bg = 1.0 - 0.01 * (vt.phase_distance(t, 1.3, 0.4) < 0.04)
    bg[3000:3500] = np.nan  # a gap in the flux, not in the row index
    for arm in ("real", "real, trimmed"):
        r = vt._null((1, bg, 1e-3, "x", arm, 100100827, 1.0 if "trimmed" in arm else 0.0))
        assert r["detected"] and abs(r["period"] - 1.3) < 0.01
    assert not vt._null((1, bg, 1e-3, "x", "white", 100100827))["detected"]


def test_tess_dump_profile_and_ar1_calibration():
    vt = _tool("validate_tess")
    from scidatasim.nuisance import momentum_dumps
    from scidatasim.experiments.transits.model import red_noise
    t = np.arange(0.0, 27.0, vt.CADENCE)
    x, times = momentum_dumps(t, np.random.default_rng(0), 2.5, 5e-4, 0.1, sign=-1.0, first=1.0)
    flat = 1 + x + np.random.default_rng(1).normal(0, 2e-4, t.size)
    prof = vt.dump_profile(t, flat, times)
    assert prof["peak_ppm"] == pytest.approx(-500 * np.exp(-0.025 / 0.1), rel=0.2) and prof["peak_significance"] > 10
    s = vt._ar1_sigma_for(1e-4, 0.05)
    rng = np.random.default_rng(3)
    rho = np.exp(-vt.CADENCE / 0.05)
    z = rng.normal(size=t.size)
    ar = np.empty(t.size)
    ar[0] = z[0]
    for i in range(1, t.size):
        ar[i] = rho * ar[i - 1] + np.sqrt(1 - rho ** 2) * z[i]
    assert red_noise(t, 1 + s * ar, (0.08,))[1][0.08] == pytest.approx(1e-4, rel=0.25)
