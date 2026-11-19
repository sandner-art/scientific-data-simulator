"""Fits of along-scan epoch astrometry: the five-parameter (single-star) solution, a periodogram
of circular orbits, a Keplerian refinement and the source excess noise.

These are the baselines a pre-registered test of DR4 states before the data arrive: what a
standard pipeline detects, with which false-alarm probability, and how far an orbit fit gets.
The fits are weighted least squares; the orbit fits are linear in the Thiele-Innes constants for a
fixed period, eccentricity and periastron time, so only those three are searched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats

from .model import DAYS_PER_YEAR, Epochs, Orbit, campbell

FIVE = ("ra_offset", "dec_offset", "parallax", "pmra", "pmdec")


def five_parameter_design(ep: Epochs) -> np.ndarray:
    """Columns: d w / d (ra_offset, dec_offset, parallax, pmra, pmdec)."""
    s, c = np.sin(ep.theta), np.cos(ep.theta)
    return np.column_stack([s, c, ep.f_al, ep.t * s, ep.t * c])


def orbit_design(ep: Epochs, period: float, eccentricity: float = 0.0, t_periastron: float = 0.0) -> np.ndarray:
    """Columns: d w / d (A, B, F, G) for a fixed period (days), eccentricity and periastron time."""
    from .model import kepler_eccentric_anomaly
    mean = 2 * np.pi * (ep.t * DAYS_PER_YEAR - t_periastron) / period
    ecc = kepler_eccentric_anomaly(mean, eccentricity)
    x = np.cos(ecc) - eccentricity
    y = np.sqrt(1.0 - eccentricity ** 2) * np.sin(ecc)
    s, c = np.sin(ep.theta), np.cos(ep.theta)
    # da = B X + G Y, dd = A X + F Y; w gets da sin(theta) + dd cos(theta)
    return np.column_stack([x * c, x * s, y * c, y * s])


@dataclass
class LinearFit:
    params: np.ndarray
    cov: np.ndarray
    chi2: float
    dof: int
    residual: np.ndarray


def weighted_lstsq(design: np.ndarray, w: np.ndarray, sigma: np.ndarray) -> LinearFit:
    sw = 1.0 / np.asarray(sigma, float)
    a = design * sw[:, None]
    b = np.asarray(w, float) * sw
    params, *_ = np.linalg.lstsq(a, b, rcond=None)
    cov = np.linalg.pinv(a.T @ a)
    residual = w - design @ params
    chi2 = float(np.sum((residual * sw) ** 2))
    return LinearFit(params, cov, chi2, int(len(w) - design.shape[1]), residual)


def fit_single_star(ep: Epochs, w: np.ndarray) -> Dict[str, float]:
    """The five-parameter solution, its standard errors, chi2 and the unit-weight error."""
    fit = weighted_lstsq(five_parameter_design(ep), w, ep.sigma)
    out = {name: float(v) for name, v in zip(FIVE, fit.params)}
    out.update({f"{name}_error": float(np.sqrt(fit.cov[i, i])) for i, name in enumerate(FIVE)})
    out.update(chi2=fit.chi2, dof=fit.dof, uwe=float(np.sqrt(fit.chi2 / max(fit.dof, 1))))
    return out


def excess_noise(ep: Epochs, w: np.ndarray, tol: float = 1e-6) -> float:
    """The source excess noise (mas): the extra per-transit noise that makes the five-parameter
    fit's chi2 equal its degrees of freedom (0 if it already does). AGIS adds it to the weights,
    so an orbit it does not model inflates it - which is why the orbit search below weights by the
    measurement errors alone unless told otherwise."""
    design = five_parameter_design(ep)

    def reduced(eps):
        fit = weighted_lstsq(design, w, np.sqrt(ep.sigma ** 2 + eps ** 2))
        return fit.chi2 / max(fit.dof, 1)

    if reduced(0.0) <= 1.0:
        return 0.0
    lo, hi = 0.0, float(np.std(w) + np.max(ep.sigma))
    while reduced(hi) > 1.0:
        hi *= 2.0
    while hi - lo > tol * max(hi, 1e-9):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if reduced(mid) > 1.0 else (lo, mid)
    return 0.5 * (lo + hi)


def frequency_grid(ep: Epochs, min_period: float = 10.0, max_period: Optional[float] = None,
                   oversample: float = 5.0) -> np.ndarray:
    """Trial periods (days), uniform in frequency, from ``min_period`` to ``max_period`` (default
    1.5 times the time span) with ``oversample`` steps per 1/span."""
    span = float((ep.t.max() - ep.t.min()) * DAYS_PER_YEAR)
    max_period = 1.5 * span if max_period is None else float(max_period)
    f = np.arange(1.0 / max_period, 1.0 / min_period, 1.0 / (oversample * span))
    return 1.0 / f[::-1]


def independent_frequencies(ep: Epochs, periods: np.ndarray) -> float:
    """Effective number of independent trial frequencies: span x frequency range (at least 1)."""
    span = float((ep.t.max() - ep.t.min()) * DAYS_PER_YEAR)
    return max(1.0, span * (1.0 / periods.min() - 1.0 / periods.max()))


def orbit_periodogram(ep: Epochs, w: np.ndarray, periods: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """Delta chi2 of a circular orbit added to the five-parameter model, per trial period.

    A circular orbit is linear in four constants for any phase, so each period is one linear fit.
    Without an orbit, Delta chi2 at a fixed period follows chi2 with 4 degrees of freedom."""
    periods = frequency_grid(ep) if periods is None else np.asarray(periods, float)
    five = five_parameter_design(ep)
    base = weighted_lstsq(five, w, ep.sigma).chi2
    sw = 1.0 / ep.sigma
    b = w * sw
    dchi2 = np.empty(periods.size)
    for k, period in enumerate(periods):
        a = np.column_stack([five, orbit_design(ep, period)]) * sw[:, None]
        params, *_ = np.linalg.lstsq(a, b, rcond=None)
        dchi2[k] = base - float(np.sum((b - a @ params) ** 2))
    return {"period": periods, "dchi2": dchi2, "chi2_single": np.float64(base)}


def false_alarm_probability(dchi2: float, n_independent: float, dof: int = 4) -> float:
    """FAP of the highest periodogram peak: 1 - (1 - p)^N with p the chi2 tail at one period.
    An approximation; ``tools/check_astrometry_detection.py`` measures the real rate on null runs."""
    p = float(stats.chi2.sf(dchi2, dof))
    return float(-np.expm1(n_independent * np.log1p(-min(p, 1 - 1e-16))))


def fit_keplerian(ep: Epochs, w: np.ndarray, period: float, n_e: int = 8, n_tp: int = 16,
                  refine: float = 0.02) -> Dict[str, object]:
    """Keplerian orbit near ``period``: a grid over period (+/- ``refine``), eccentricity and
    periastron time, linear in the Thiele-Innes constants. Returns the orbit, its chi2, a0 and
    its standard error (linearised) and the significance a0 / sigma(a0) as in DR4's NSS tables."""
    five = five_parameter_design(ep)
    best = None
    for p in period * (1.0 + np.linspace(-refine, refine, 9)):
        for e in np.linspace(0.0, 0.9, n_e):
            for tp in np.linspace(-p / 2, p / 2, n_tp, endpoint=False):
                fit = weighted_lstsq(np.column_stack([five, orbit_design(ep, p, e, tp)]), w, ep.sigma)
                if best is None or fit.chi2 < best[0].chi2:
                    best = (fit, p, e, tp)
    _, p, e, tp = best
    return _keplerian_solution(ep, w, p, e, tp)


def _keplerian_chi2(ep: Epochs, w: np.ndarray, five: np.ndarray, period: float, e: float, tp: float) -> float:
    return weighted_lstsq(np.column_stack([five, orbit_design(ep, period, e, tp)]), w, ep.sigma).chi2


def polish_keplerian(ep: Epochs, w: np.ndarray, period: float, eccentricity: float, t_periastron: float,
                     max_e: float = 0.95) -> Dict[str, object]:
    """Local optimisation (Nelder-Mead) of period, eccentricity and periastron time from a starting
    point; the Thiele-Innes constants and five parameters stay linear. Returns :func:`fit_keplerian`'s
    keys for the optimum."""
    from scipy.optimize import minimize
    five = five_parameter_design(ep)

    def cost(x):
        p, e, tp = np.exp(x[0]), x[1], x[2]
        if not 0.0 <= e < max_e:
            return np.inf
        return _keplerian_chi2(ep, w, five, p, e, tp)

    start = np.array([np.log(period), min(max(eccentricity, 0.0), max_e - 1e-3), t_periastron])
    simplex = np.vstack([start, start + [0.01, 0, 0], start + [0, 0.05, 0], start + [0, 0, 0.05 * period]])
    if simplex[2, 1] >= max_e:
        simplex[2, 1] = start[1] - 0.05
    res = minimize(cost, start, method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-4, "maxiter": 800, "initial_simplex": simplex})
    p, e, tp = float(np.exp(res.x[0])), float(res.x[1]), float(res.x[2])
    return _keplerian_solution(ep, w, p, e, tp)


def _keplerian_solution(ep: Epochs, w: np.ndarray, p: float, e: float, tp: float) -> Dict[str, object]:
    five = five_parameter_design(ep)
    fit = weighted_lstsq(np.column_stack([five, orbit_design(ep, p, e, tp)]), w, ep.sigma)
    a, b, f, g = (float(v) for v in fit.params[5:])
    a0 = campbell(a, b, f, g)["a0"]
    grad = np.zeros(4)
    for i in range(4):
        step = np.zeros(4)
        step[i] = 1e-6 * max(a0, 1e-3)
        grad[i] = (campbell(*(fit.params[5:] + step))["a0"] - campbell(*(fit.params[5:] - step))["a0"]) / (2 * step[i])
    sigma_a0 = float(np.sqrt(max(grad @ fit.cov[5:, 5:] @ grad, 0.0)))
    out: Dict[str, object] = {name: float(v) for name, v in zip(FIVE, fit.params[:5])}
    out["parallax_error"] = float(np.sqrt(fit.cov[2, 2]))
    out.update(orbit=Orbit(p, e, tp, a, b, f, g), chi2=fit.chi2, dof=fit.dof - 3, a0=a0, a0_error=sigma_a0,
               significance=a0 / sigma_a0 if sigma_a0 > 0 else float("inf"))
    return out


def candidate_periods(gram: Dict[str, np.ndarray], n_peaks: int = 3) -> np.ndarray:
    """Periods worth a Keplerian fit: the ``n_peaks`` highest local maxima of the periodogram and
    twice each of their periods (an eccentric orbit puts power at harmonics, P/2 above all, so the
    circular search's highest peak can sit at half the true period). Within 1 % counts as one."""
    d = gram["dchi2"]
    peaks = [k for k in range(d.size) if (k == 0 or d[k] >= d[k - 1]) and (k == d.size - 1 or d[k] >= d[k + 1])]
    peaks = sorted(peaks, key=lambda k: -d[k])
    out: list = []
    for k in peaks:
        if len(out) >= 2 * n_peaks:
            break
        for p in (float(gram["period"][k]), 2.0 * float(gram["period"][k])):
            if all(abs(p / q - 1.0) > 0.01 for q in out):
                out.append(p)
    return np.array(out)


def best_keplerian(ep: Epochs, w: np.ndarray, gram: Optional[Dict[str, np.ndarray]] = None,
                   n_peaks: int = 3) -> Dict[str, object]:
    """The Keplerian search: a grid fit and a local optimisation at each candidate period, the
    lowest chi2 kept."""
    gram = orbit_periodogram(ep, w) if gram is None else gram
    best = None
    for p in candidate_periods(gram, n_peaks):
        grid = fit_keplerian(ep, w, p)
        orb = grid["orbit"]
        sol = polish_keplerian(ep, w, orb.period, orb.eccentricity, orb.t_periastron)
        if sol["chi2"] > grid["chi2"]:
            sol = grid
        if best is None or sol["chi2"] < best["chi2"]:
            best = sol
    return best


def detect(ep: Epochs, w: np.ndarray, fap_threshold: float = 1e-3, periods: Optional[Sequence[float]] = None,
           keplerian: bool = True, n_peaks: int = 3) -> Dict[str, object]:
    """The baseline pipeline: five-parameter fit, circular-orbit periodogram and the FAP of its
    highest peak. If detected and ``keplerian``: Keplerian fits (grid, then local optimisation) at
    the candidate periods of :func:`candidate_periods`, the best by chi2 kept.

    ``period`` is the periodogram's peak; ``period_best`` and ``parallax`` are the final estimates
    (the Keplerian fit's when an orbit was detected and fitted, otherwise the peak and the
    five-parameter parallax: an orbit near one year, left unmodelled, biases the parallax)."""
    gram = orbit_periodogram(ep, w, periods)
    k = int(np.argmax(gram["dchi2"]))
    n_ind = independent_frequencies(ep, gram["period"])
    fap = false_alarm_probability(float(gram["dchi2"][k]), n_ind)
    single = fit_single_star(ep, w)
    out: Dict[str, object] = {"period": float(gram["period"][k]), "dchi2": float(gram["dchi2"][k]),
                              "fap": fap, "n_independent": n_ind, "detected": bool(fap < fap_threshold),
                              "single_star": single, "period_best": float(gram["period"][k]),
                              "parallax": single["parallax"], "parallax_error": single["parallax_error"]}
    if out["detected"] and keplerian:
        best = best_keplerian(ep, w, gram, n_peaks)
        out["keplerian"] = best
        out.update(period_best=best["orbit"].period, parallax=best["parallax"],
                   parallax_error=best.get("parallax_error", single["parallax_error"]))
    return out


def orbit_noncentrality(ep: Epochs, orbit: Orbit) -> float:
    """The Delta chi2 a noise-free orbit signal adds over the best five-parameter fit: the part of
    the orbit the five parameters cannot absorb, in units of the noise. With noise, the Delta chi2
    of the true orbit follows a non-central chi2 with this non-centrality."""
    oa, od = orbit.offsets(ep.t)
    signal = oa * np.sin(ep.theta) + od * np.cos(ep.theta)
    return weighted_lstsq(five_parameter_design(ep), signal, ep.sigma).chi2
