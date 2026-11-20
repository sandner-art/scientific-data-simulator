"""Transit light curves: a planet crossing a limb-darkened star on a circular orbit.

The flux blocked is integrated over the stellar disk in annuli: for intensity I(r) (quadratic limb
darkening, I = 1 - u1 (1 - mu) - u2 (1 - mu)^2, mu = sqrt(1 - r^2)) the planet disk of radius p at
projected distance z blocks sum_i I(r_i) [A(r_{i+1}) - A(r_i)], with A(r) the closed-form area
where the planet disk overlaps a circle of radius r. For a uniform star this is exact; with limb
darkening it converges with the number of annuli (second order; the tests check both).

Units: times in days, radii in stellar radii, ``a`` = a / R*, ``b`` = a cos(i) / R* (impact
parameter). Long-cadence exposures are integrated by supersampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

G_CGS = 6.674e-8
DAY = 86400.0


def overlap_area(r, p, z):
    """Area of the intersection of a circle of radius ``r`` (centred on the star) and the planet
    disk of radius ``p`` at distance ``z`` (arrays broadcast)."""
    r, p, z = np.broadcast_arrays(np.asarray(r, float), np.asarray(p, float), np.asarray(z, float))
    out = np.zeros(r.shape)
    inside = z <= np.abs(r - p)  # one circle within the other
    out = np.where(inside, np.pi * np.minimum(r, p) ** 2, out)
    part = (z < r + p) & ~inside
    if np.any(part):
        rr, pp, zz = r[part], p[part], z[part]
        k0 = np.arccos(np.clip((pp ** 2 + zz ** 2 - rr ** 2) / (2 * pp * zz), -1, 1))
        k1 = np.arccos(np.clip((rr ** 2 + zz ** 2 - pp ** 2) / (2 * rr * zz), -1, 1))
        root = np.sqrt(np.clip(4 * zz ** 2 * rr ** 2 - (rr ** 2 + zz ** 2 - pp ** 2) ** 2, 0, None))
        out[part] = pp ** 2 * k0 + rr ** 2 * k1 - 0.5 * root
    return out


def blocked_fraction(z, p: float, u1: float = 0.0, u2: float = 0.0, n_annuli: int = 400) -> np.ndarray:
    """Fraction of the star's flux the planet blocks at projected separations ``z``."""
    z = np.atleast_1d(np.asarray(z, float))
    out = np.zeros(z.shape)
    near = z < 1.0 + p
    if not np.any(near):
        return out
    if u1 == 0.0 and u2 == 0.0:  # uniform disk: exact
        out[near] = overlap_area(1.0, p, z[near]) / np.pi
        return out
    edges = np.linspace(0.0, 1.0, n_annuli + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    mu = np.sqrt(1.0 - mid ** 2)
    intensity = 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
    total = np.pi * (1.0 - u1 / 3.0 - u2 / 6.0)
    area = overlap_area(edges[None, :], p, z[near][:, None])  # (n_near, n_annuli + 1)
    out[near] = (np.diff(area, axis=1) @ intensity) / total
    return out


def projected_separation(t, period: float, t0: float, a: float, b: float) -> np.ndarray:
    """Planet-star distance on the sky (stellar radii) on a circular orbit; behind the star
    (secondary eclipse side) it is returned as a large number, so nothing is blocked."""
    phase = 2 * np.pi * (np.asarray(t, float) - t0) / period
    cos_i = b / a
    z = a * np.sqrt(np.sin(phase) ** 2 + (cos_i * np.cos(phase)) ** 2)
    return np.where(np.cos(phase) > 0, z, 1e9)


def a_over_rstar(period: float, density: float = 1.41) -> float:
    """a / R* from Kepler's third law for a stellar density in g/cm^3 (the Sun: 1.41)."""
    return float((G_CGS * density * (period * DAY) ** 2 / (3 * np.pi)) ** (1.0 / 3.0))


@dataclass
class Planet:
    period: float          # days
    t0: float              # mid-transit time, days
    rp: float              # Rp / R*
    a: float               # a / R*
    b: float               # impact parameter
    u1: float = 0.4
    u2: float = 0.25

    @property
    def duration(self) -> float:
        """Total duration (first to fourth contact), days; 0 if it does not transit."""
        k = ((1 + self.rp) ** 2 - self.b ** 2)
        if k <= 0:
            return 0.0
        sin_i = np.sqrt(1 - (self.b / self.a) ** 2)
        return float(self.period / np.pi * np.arcsin(min(1.0, np.sqrt(k) / (self.a * sin_i))))

    def flux(self, t, exposure: float = 0.0, supersample: int = 1, n_annuli: int = 400) -> np.ndarray:
        """Relative flux (1 out of transit). ``exposure`` (days) > 0 averages ``supersample``
        sub-exposures, as a long-cadence integration does."""
        t = np.asarray(t, float)
        if exposure > 0 and supersample > 1:
            offsets = (np.arange(supersample) + 0.5) / supersample * exposure - 0.5 * exposure
            tt = (t[:, None] + offsets[None, :]).ravel()
            z = projected_separation(tt, self.period, self.t0, self.a, self.b)
            return 1.0 - blocked_fraction(z, self.rp, self.u1, self.u2, n_annuli).reshape(t.size, supersample).mean(1)
        z = projected_separation(t, self.period, self.t0, self.a, self.b)
        return 1.0 - blocked_fraction(z, self.rp, self.u1, self.u2, n_annuli)

    def in_transit(self, t) -> np.ndarray:
        z = projected_separation(t, self.period, self.t0, self.a, self.b)
        return z < 1.0 + self.rp

    def transits_in(self, t) -> int:
        """Number of distinct transits with at least one point in ``t``."""
        t = np.asarray(t, float)
        t = t[np.isfinite(t)]
        epochs = np.round((t[self.in_transit(t)] - self.t0) / self.period)
        return int(np.unique(epochs).size)


def detrend(t, flux, window: float = 0.5) -> np.ndarray:
    """Divide by a running median over ``window`` days (removes stellar variability slower than
    the window; transits much shorter than it survive)."""
    t, flux = np.asarray(t, float), np.asarray(flux, float)
    out = np.full(flux.shape, np.nan)
    ok = np.isfinite(t) & np.isfinite(flux)
    tt, ff = t[ok], flux[ok]
    lo = np.searchsorted(tt, tt - window / 2)
    hi = np.searchsorted(tt, tt + window / 2)
    trend = np.array([np.median(ff[a:b]) for a, b in zip(lo, hi)])
    out[ok] = ff / trend
    return out


def transit_search(t, flux, error, durations=(0.04, 0.08, 0.12, 0.2), min_period: float = 0.5,
                   max_period: Optional[float] = None, snr_threshold: float = 7.1, window: float = 0.5,
                   trim: float = 0.0, min_gap: float = 0.5) -> dict:
    """The baseline: running-median detrending, box least squares, and a detection when the box's
    signal-to-noise depth / sigma_depth (= sqrt(Delta chi2)) exceeds ``snr_threshold`` (7.1, as in
    the Kepler pipeline). Returns the BLS best period, depth (ppm), duration, t0, SNR, detected.
    ``trim`` (days, default 0) drops the data that follow the start and every gap longer than
    ``min_gap``, where instruments settle (:func:`scidatasim.nuisance.trim_after_gaps`)."""
    from scidatasim.baselines.bls import bls
    t = np.asarray(t, float)
    if trim > 0:  # gaps are where the flux is missing, whether or not the time stamps are
        from scidatasim.nuisance import trim_after_gaps
        flux = np.asarray(flux, float)
        keep = np.isfinite(t) & np.isfinite(flux)
        keep[keep] = trim_after_gaps(t[keep], trim, min_gap)
        t, flux = t[keep], flux[keep]
        if np.ndim(error):
            error = np.asarray(error, float)[keep]
    flat = detrend(t, flux, window)
    span = float(np.nanmax(t) - np.nanmin(t))
    r = bls(t, flat, error, durations=durations, min_period=min_period,
            max_period=max_period or min(span / 2.0, 20.0))
    snr = float(np.sqrt(max(r["best_dchi2"], 0.0)))
    return {"period": r["best_period"], "depth_ppm": 1e6 * r["best_depth"], "duration": r["best_duration"],
            "t0": r["best_t0"], "snr": snr, "detected": bool(snr > snr_threshold and r["best_depth"] > 0)}
