"""Significance: p-values and Z, trials factors and the look-elsewhere effect.

A search over many trials (frequencies, masses, positions) reports its best local p-value; the
chance that noise alone gives a value that small *somewhere* is the global p-value. With N
independent trials it is Sidak's 1 - (1 - p)^N; for a continuous scan, Gross and Vitells (2010)
estimate it from the number of upcrossings of a low reference level, counted on the data or on
a few null simulations. ``empirical_p`` gives the global p-value directly from null simulations.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def p_to_z(p) -> np.ndarray:
    """One-sided Gaussian significance of a p-value (p = 2.87e-7 -> Z = 5)."""
    return stats.norm.isf(np.asarray(p, float))


def z_to_p(z) -> np.ndarray:
    return stats.norm.sf(np.asarray(z, float))


def sidak(p_local, n_trials: float) -> np.ndarray:
    """Global p-value of the smallest of ``n_trials`` independent local p-values."""
    p = np.clip(np.asarray(p_local, float), 0.0, 1.0)
    return -np.expm1(n_trials * np.log1p(-np.minimum(p, 1 - 1e-16)))


def trials_factor(p_local, p_global) -> np.ndarray:
    return np.asarray(p_global, float) / np.asarray(p_local, float)


def upcrossings(scan: Sequence[float], level: float) -> int:
    """Times a scan of a test statistic (e.g. Delta chi2 or -2 log likelihood ratio) crosses
    ``level`` upwards."""
    s = np.asarray(scan, float)
    return int(np.sum((s[:-1] < level) & (s[1:] >= level)))


def gross_vitells(q_max: float, scans_or_n0, level: float = 0.5, dof: int = 1) -> float:
    """Global p-value of a maximum ``q_max`` of a scanned chi2-like statistic (Gross and Vitells
    2010): P(q > q_max) + N(level) * exp(-(q_max - level) / 2), with N the mean number of
    upcrossings of ``level`` - counted on null scans (a list of arrays) or given directly. Valid
    for ``dof`` = 1 (and approximately for chi2 with more dof, scaled by its density ratio)."""
    if np.isscalar(scans_or_n0):
        n0 = float(scans_or_n0)
    else:
        n0 = float(np.mean([upcrossings(s, level) for s in scans_or_n0]))
    local = stats.chi2.sf(q_max, dof)
    if dof == 1:
        return float(min(1.0, local + n0 * np.exp(-(q_max - level) / 2)))
    ratio = (q_max / level) ** ((dof - 1) / 2) * np.exp(-(q_max - level) / 2)
    return float(min(1.0, local + n0 * ratio))


def empirical_p(value: float, null: Sequence[float], larger_is_extreme: bool = True) -> float:
    """p-value of ``value`` against null simulations of the same statistic, (k + 1) / (n + 1):
    never 0, so a few hundred nulls cannot claim more than they measured."""
    null = np.asarray(null, float)
    k = int(np.sum(null >= value)) if larger_is_extreme else int(np.sum(null <= value))
    return (k + 1) / (null.size + 1)
