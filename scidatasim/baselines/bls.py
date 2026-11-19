"""Box least squares (Kovacs, Zucker and Mazeh 2002): the transit search.

For each trial period the series is folded and binned in phase; a box of each trial duration is
slid over the bins (wrapping around phase 1). For a box with inverse-variance weight W_in inside
and W_out outside, the depth is the difference of the weighted means outside and inside, and the
fit improves chi2 by depth^2 W_in W_out / (W_in + W_out); that Delta chi2 is the statistic. Bins
are a fraction of the shortest duration (``oversample``), so a box edge is placed to within
duration / oversample.

The false-alarm probability of the highest peak has no closed form for BLS; :func:`bls_null`
gives the null distribution of the highest Delta chi2 on noise-only series with the same time
stamps and errors (``scidatasim.baselines.significance.empirical_p`` turns it into a p-value).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def bls(t, y, dy=None, periods: Optional[Sequence[float]] = None, durations: Sequence[float] = (0.04, 0.08, 0.12),
        oversample: int = 4, min_period: float = 0.5, max_period: Optional[float] = None,
        period_oversample: float = 3.0, dips_only: bool = True) -> dict:
    """Search ``periods`` (default :func:`period_grid` from ``min_period`` to ``max_period``) and
    ``durations`` (same units as ``t``). Returns per period the best Delta chi2 with its depth,
    duration and mid-transit time, and the overall best. A positive depth is a dip; with
    ``dips_only`` (default) boxes brighter than the rest are not counted (stellar spots and
    detrending residuals make bumps as readily as dips)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    dy = np.ones_like(y) if dy is None else np.broadcast_to(np.asarray(dy, float), y.shape)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    t, y, dy = t[ok], y[ok], dy[ok]
    durations = np.sort(np.asarray(durations, float))
    periods = period_grid(t, durations, min_period, max_period, period_oversample) if periods is None \
        else np.asarray(periods, float)
    w = 1.0 / dy ** 2
    W = float(w.sum())
    wy = w * y
    t0 = float(t.min())
    out = {k: np.empty(periods.size) for k in ("dchi2", "depth", "duration", "t0")}
    for k, period in enumerate(periods):
        nbins = max(int(np.ceil(oversample * period / durations[0])), 4)
        idx = np.minimum((((t - t0) / period) % 1.0 * nbins).astype(int), nbins - 1)
        bw = np.bincount(idx, weights=w, minlength=nbins)
        bwy = np.bincount(idx, weights=wy, minlength=nbins)
        cw = np.concatenate([[0.0], np.cumsum(np.concatenate([bw, bw]))])
        cwy = np.concatenate([[0.0], np.cumsum(np.concatenate([bwy, bwy]))])
        best = (-1.0, 0.0, 0.0, 0.0)
        for dur in durations:
            nb = max(1, int(round(dur / period * nbins)))
            if nb >= nbins:
                continue
            win = cw[nb:nb + nbins] - cw[:nbins]
            winy = cwy[nb:nb + nbins] - cwy[:nbins]
            wout = W - win
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = (float(wy.sum()) - winy) / wout - winy / win
                stat = np.where((win > 0) & (wout > 0) & ((depth > 0) | (not dips_only)),
                                depth ** 2 * win * wout / W, 0.0)
            j = int(np.argmax(stat))
            if stat[j] > best[0]:
                best = (float(stat[j]), float(depth[j]), nb * period / nbins,
                        t0 + (j + nb / 2) * period / nbins)
        out["dchi2"][k], out["depth"][k], out["duration"][k], out["t0"][k] = best
    i = int(np.argmax(out["dchi2"]))
    return {"period": periods, **out, "best_period": float(periods[i]), "best_dchi2": float(out["dchi2"][i]),
            "best_depth": float(out["depth"][i]), "best_duration": float(out["duration"][i]),
            "best_t0": float(out["t0"][i]), "n": int(t.size)}


def period_grid(t, durations, min_period: float = 0.5, max_period: Optional[float] = None,
                oversample: float = 3.0) -> np.ndarray:
    """Trial periods from ``min_period`` to ``max_period`` (default a third of the span, so at least
    three transits fit), spaced so that a step shifts the last transit by at most
    min(duration) / ``oversample`` over the span: frequency grows by a factor 1 + d / (oversample T)."""
    t = np.asarray(t, float)
    span = float(np.nanmax(t) - np.nanmin(t))
    max_period = max_period or span / 3.0
    if max_period <= min_period:
        raise ValueError(f"no periods to search: max_period {max_period:g} <= min_period {min_period:g}")
    step = np.log1p(float(np.min(durations)) / (oversample * span))
    n = int(np.floor(np.log(max_period / min_period) / step)) + 1
    return np.sort(max_period / np.exp(np.arange(n) * step))


def bls_null(t, dy, n: int = 200, rng: Optional[np.random.Generator] = None, y=None, mode: str = "gaussian",
             **search) -> np.ndarray:
    """Highest Delta chi2 of :func:`bls` on ``n`` noise-only series: Gaussian with the given errors
    (``mode="gaussian"``) or the measured values shuffled in time (``mode="shuffle"``, keeps the
    value distribution, destroys correlations). Same time stamps and search settings."""
    rng = np.random.default_rng() if rng is None else rng
    t = np.asarray(t, float)
    dy = np.broadcast_to(np.asarray(dy, float), t.shape)
    out = np.empty(n)
    for i in range(n):
        if mode == "gaussian":
            sim = rng.normal(0.0, dy)
        elif mode == "shuffle":
            sim = rng.permutation(np.asarray(y, float))
        else:
            raise ValueError("mode must be 'gaussian' or 'shuffle'")
        out[i] = bls(t, sim, dy, **search)["best_dchi2"]
    return out
