"""Generalised Lomb-Scargle periodogram (Zechmeister and Kuerster 2009: a sinusoid plus a floating
mean, weighted by 1/sigma^2) and the false-alarm probability of its highest peak (Baluev 2008).

Power is the "standard" normalisation: the fraction of the constant model's chi2 the sinusoid
removes, in [0, 1]. ``lomb_scargle_fap`` follows Baluev's approximation as astropy implements it
(``method='baluev'``); the tests check both against astropy when it is installed (power to 1e-9,
the FAP to 1e-3 relative).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import gammaln


def frequency_grid(t, min_period: Optional[float] = None, max_period: Optional[float] = None,
                   oversample: float = 5.0, nyquist_factor: float = 5.0) -> np.ndarray:
    """Frequencies uniform in frequency, ``oversample`` steps per 1/span; from 1/max_period (default
    the span) to 1/min_period (default ``nyquist_factor`` times the mean pseudo-Nyquist frequency)."""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    span = float(t.max() - t.min())
    fmin = 1.0 / (max_period or span)
    fmax = 1.0 / min_period if min_period else nyquist_factor * 0.5 * t.size / span
    return np.arange(fmin, fmax, 1.0 / (oversample * span))


def lomb_scargle(t, y, dy=None, frequency=None, chunk: int = 2000) -> dict:
    """Power at each frequency; returns ``{"frequency", "power", "best_frequency", "best_power"}``.
    NaN points are ignored."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    dy = np.ones_like(y) if dy is None else np.broadcast_to(np.asarray(dy, float), y.shape)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    t, y, dy = t[ok], y[ok], dy[ok]
    frequency = frequency_grid(t) if frequency is None else np.asarray(frequency, float)
    w = 1.0 / dy ** 2
    w = w / w.sum()
    ybar = float(np.sum(w * y))
    yy = float(np.sum(w * (y - ybar) ** 2))
    power = np.empty(frequency.size)
    for k in range(0, frequency.size, chunk):
        f = frequency[k:k + chunk]
        arg = 2 * np.pi * np.outer(f, t)
        c, s = np.cos(arg), np.sin(arg)
        C, S = c @ w, s @ w
        YC = (c * (y - ybar)) @ w
        YS = (s * (y - ybar)) @ w
        CC = (c * c) @ w - C * C
        SS = (s * s) @ w - S * S
        CS = (c * s) @ w - C * S
        D = CC * SS - CS ** 2
        power[k:k + chunk] = (SS * YC ** 2 + CC * YS ** 2 - 2 * CS * YC * YS) / (yy * D)
    i = int(np.argmax(power))
    return {"frequency": frequency, "power": power, "best_frequency": float(frequency[i]),
            "best_power": float(power[i]), "n": int(t.size)}


def _weighted_var(t, dy):
    w = 1.0 / np.asarray(dy, float) ** 2
    w = w / w.sum()
    return float(np.sum(w * t ** 2) - np.sum(w * t) ** 2)


def lomb_scargle_fap(power, fmax: float, t, dy=None) -> np.ndarray:
    """Baluev (2008) false-alarm probability of a highest peak of standard-normalised ``power``,
    searched up to frequency ``fmax``: 1 - (1 - FAP_single) exp(-tau)."""
    t = np.asarray(t, float)
    dy = np.ones_like(t) if dy is None else np.broadcast_to(np.asarray(dy, float), t.shape)
    ok = np.isfinite(t) & np.isfinite(dy)
    t, dy = t[ok], dy[ok]
    n = t.size
    nh, nk = n - 1, n - 3
    z = np.asarray(power, float)
    fap_single = (1 - z) ** (0.5 * nk)
    gamma = np.sqrt(2.0 / nh) * np.exp(gammaln(nh / 2) - gammaln((nh - 1) / 2))
    w = fmax * np.sqrt(4 * np.pi * _weighted_var(t, dy))
    tau = gamma * w * (1 - z) ** (0.5 * (nk - 1)) * np.sqrt(0.5 * nh * z)
    return 1 - (1 - fap_single) * np.exp(-tau)
