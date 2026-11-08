# simulator/diagnostics/stats.py
"""Small resampling tests shared by the diagnostics (NumPy only)."""
from __future__ import annotations

from typing import Any, Dict, Hashable, Optional, Sequence

import numpy as np


def gap_test(a: Sequence[float], b: Sequence[float], paired: bool = False, n_resamples: int = 10000,
             seed: Optional[int] = 0, confidence: float = 0.95,
             clusters: Optional[Sequence[Hashable]] = None) -> Dict[str, Any]:
    """Difference of means ``mean(a) - mean(b)`` with a permutation p-value and a bootstrap interval.

    ``paired=True`` needs equal lengths (item i of ``a`` is the twin of item i of ``b``) and uses
    a sign-flip permutation of the differences; otherwise group labels are permuted. The p-value
    is two-sided, with the +1 correction, so it is never 0.

    ``clusters`` (paired only) gives each pair a cluster label, e.g. the run its questions come
    from. Questions of one run share one fit and are not independent; signs are then flipped and
    the bootstrap resamples per cluster, which keeps the test valid under within-cluster
    correlation.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples need at least one score")
    rng = np.random.default_rng(seed)
    n_clusters = None
    if paired:
        if a.shape != b.shape:
            raise ValueError("paired samples must have the same length")
        d = a - b
        observed = float(d.mean())
        if clusters is None:
            units = d
        else:
            if len(clusters) != d.size:
                raise ValueError("one cluster label per pair")
            labels = {}
            index = np.array([labels.setdefault(c, len(labels)) for c in clusters])
            units = np.bincount(index, weights=d)  # per-cluster sums
            n_clusters = len(labels)
        signs = rng.choice((-1.0, 1.0), size=(n_resamples, units.size))
        null = (signs * units).sum(axis=1) / d.size
        picks = rng.integers(0, units.size, size=(n_resamples, units.size))
        if clusters is None:
            boot = units[picks].mean(axis=1)
        else:
            sizes = np.bincount(index).astype(float)
            boot = units[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    else:
        if clusters is not None:
            raise ValueError("clusters are supported for paired tests only")
        observed = float(a.mean() - b.mean())
        pooled = np.concatenate([a, b])
        null = np.empty(n_resamples)
        for i in range(n_resamples):
            perm = rng.permutation(pooled)
            null[i] = perm[:a.size].mean() - perm[a.size:].mean()
        boot = (a[rng.integers(0, a.size, size=(n_resamples, a.size))].mean(axis=1)
                - b[rng.integers(0, b.size, size=(n_resamples, b.size))].mean(axis=1))
    p = (1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (n_resamples + 1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()), "gap": observed, "ci": (float(lo), float(hi)),
            "p_value": float(p), "n_a": int(a.size), "n_b": int(b.size), "paired": paired,
            "n_clusters": n_clusters}


def binomial_p_greater(successes: int, n: int, p0: float) -> float:
    """One-sided exact binomial p-value P(X >= successes | n, p0)."""
    from scipy.stats import binom
    if n == 0 or successes <= 0:
        return 1.0
    p0 = min(max(p0, 0.0), 1.0)
    return float(binom.sf(successes - 1, n, p0))
