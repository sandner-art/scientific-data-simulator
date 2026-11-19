"""Detection baselines: Lomb-Scargle with Baluev's FAP, BLS, look-elsewhere corrections."""
import numpy as np
import pytest

from scidatasim.baselines import (bls, bls_null, empirical_p, gross_vitells, lomb_scargle, lomb_scargle_fap, p_to_z,
                                  sidak, z_to_p)
from scidatasim.baselines.bls import period_grid
from scidatasim.baselines.periodogram import frequency_grid


def _uneven(rng, n=300, span=100.0):
    return np.sort(rng.uniform(0, span, n))


def test_lomb_scargle_matches_astropy():
    astropy = pytest.importorskip("astropy.timeseries")
    rng = np.random.default_rng(0)
    t = _uneven(rng)
    dy = rng.uniform(0.5, 1.5, t.size)
    y = 0.4 * np.sin(2 * np.pi * t / 7.3) + rng.normal(0, dy) + 3.0
    f = frequency_grid(t)
    ours = lomb_scargle(t, y, dy, f)
    ref = astropy.LombScargle(t, y, dy, normalization="standard").power(f)
    assert np.allclose(ours["power"], ref, atol=1e-9)
    assert 1 / ours["best_frequency"] == pytest.approx(7.3, rel=0.02)
    ref_fap = astropy.LombScargle(t, y, dy, normalization="standard").false_alarm_probability(
        ref.max(), method="baluev", maximum_frequency=f.max())
    assert lomb_scargle_fap(ours["best_power"], f.max(), t, dy) == pytest.approx(ref_fap, rel=1e-3)


def test_lomb_scargle_fap_is_calibrated_on_noise():
    rng = np.random.default_rng(1)
    t = _uneven(rng, 200)
    f = frequency_grid(t)
    faps = []
    for _ in range(200):
        r = lomb_scargle(t, rng.normal(0, 1, t.size), None, f)
        faps.append(lomb_scargle_fap(r["best_power"], f.max(), t))
    faps = np.array(faps)
    assert np.mean(faps < 0.05) <= 0.09 and np.mean(faps < 0.5) >= 0.3  # an upper bound, not wildly loose


def _brute_box(t, y, w, period, nb_dur):
    """Exact best box of a given duration over all start phases (slow reference)."""
    phase = (t - t.min()) / period % 1.0
    best = 0.0
    W, wy = w.sum(), (w * y).sum()
    for start in np.linspace(0, 1, 2000, endpoint=False):
        inside = ((phase - start) % 1.0) < nb_dur
        win, winy = w[inside].sum(), (w * y)[inside].sum()
        if 0 < win < W:
            depth = (wy - winy) / (W - win) - winy / win
            best = max(best, depth ** 2 * win * (W - win) / W)
    return best


def test_bls_recovers_a_transit_and_matches_brute_force():
    rng = np.random.default_rng(2)
    t = np.arange(0, 27, 10 / 1440)
    y = 1.0 + rng.normal(0, 1e-3, t.size)
    period, dur, t0, depth = 3.7, 0.1, 1.3, 1.5e-3
    y[np.abs(((t - t0 + 0.5 * period) % period) - 0.5 * period) < dur / 2] -= depth
    r = bls(t, y, 1e-3, durations=(0.05, 0.1, 0.15), max_period=9.0, oversample=6)
    assert r["best_period"] == pytest.approx(period, rel=0.002)
    assert r["best_depth"] == pytest.approx(depth, rel=0.15) and r["best_duration"] == pytest.approx(dur, rel=0.25)
    assert abs((r["best_t0"] - t0 + 0.5 * period) % period - 0.5 * period) < 0.05
    w = np.full(t.size, 1e6)
    exact = _brute_box(t, y, w, r["best_period"], r["best_duration"] / r["best_period"])
    assert r["best_dchi2"] == pytest.approx(exact, rel=0.05)
    grid = period_grid(t, (0.05,), 0.5, 9.0)
    assert grid[0] >= 0.5 - 1e-9 and grid[-1] == pytest.approx(9.0) and np.all(np.diff(grid) > 0)


def test_bls_null_gives_an_empirical_fap():
    rng = np.random.default_rng(3)
    t = np.arange(0, 10, 30 / 1440)
    null = bls_null(t, 1e-3, n=30, rng=rng, durations=(0.1,), max_period=3.0)
    y = 1.0 + rng.normal(0, 1e-3, t.size)
    y[np.abs(((t - 0.7 + 1.0) % 2.0) - 1.0) < 0.05] -= 2e-3
    r = bls(t, y, 1e-3, durations=(0.1,), max_period=3.0)
    assert empirical_p(r["best_dchi2"], null) == pytest.approx(1 / 31)
    noise = bls(t, rng.normal(0, 1e-3, t.size), 1e-3, durations=(0.1,), max_period=3.0)["best_dchi2"]
    assert empirical_p(noise, null) > 0.01
    with pytest.raises(ValueError):
        bls_null(t, 1e-3, n=1, mode="other")


def test_significance_helpers():
    assert p_to_z(2.866515718791939e-07) == pytest.approx(5.0, rel=1e-6)
    assert z_to_p(3.0) == pytest.approx(1.3498980316300946e-03)
    assert sidak(0.01, 100) == pytest.approx(1 - 0.99 ** 100)
    rng = np.random.default_rng(4)
    scans = [np.convolve(rng.normal(0, 1, 2000), np.ones(20) / np.sqrt(20), "same") ** 2 for _ in range(50)]
    q = 25.0
    gv = gross_vitells(q, scans, level=0.5)
    observed = np.mean([s.max() >= q for s in scans])
    assert 0 < gv < 1 and observed <= max(gv * 3, 0.1)
    assert gross_vitells(q, 0.0) == pytest.approx(z_to_p(5.0) * 2)
