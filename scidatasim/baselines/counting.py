"""Counting experiments: binned Poisson likelihood fits, a signal scan and its significance.

The bump hunt of particle physics: a smooth background (``exp`` of a polynomial in the scaled
coordinate, times a known efficiency, plus known peaks with free yields) is fitted to binned counts
by maximum likelihood; a Gaussian signal of known width is then added at each trial position, its
yield (>= 0) and the background refitted. The test statistic q0 = 2 (NLL_b - NLL_s+b) gives the
local significance Z = sqrt(q0) (Wilks, one-sided); the global p-value over the scan comes from
the Gross-Vitells upcrossings of the scan itself, or from null toys.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import optimize, special, stats

from .significance import gross_vitells, upcrossings


def gaussian_bins(edges, mean: float, sigma: float) -> np.ndarray:
    """Fraction of a unit Gaussian in each bin."""
    return np.diff(stats.norm.cdf(np.asarray(edges, float), mean, sigma))


def poisson_nll(counts, expected) -> float:
    mu = np.clip(np.asarray(expected, float), 1e-300, None)
    n = np.asarray(counts, float)
    return float(np.sum(mu - n * np.log(mu) + special.gammaln(n + 1)))


class BinnedFit:
    """Background (+ signal) model for counts in bins ``edges``.

    Background: exp(poly_deg(x)) * bin width * efficiency, with x the bin centre's logarithm
    (``log_x``, default) or the centre itself, scaled to [-1, 1]; known peaks: Gaussians (``known``: list of (mean, sigma)) with free
    yields >= 0."""

    def __init__(self, edges, degree: int = 3, efficiency: Optional[Sequence[float]] = None,
                 known: Sequence = (), log_x: bool = True):
        self.edges = np.asarray(edges, float)
        self.centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        lo, hi = self.edges[0], self.edges[-1]
        if log_x and lo > 0:  # falling spectra are near power laws: polynomials in log m fit them
            self.x = 2 * (np.log(self.centres) - np.log(lo)) / (np.log(hi) - np.log(lo)) - 1
        else:
            self.x = 2 * (self.centres - lo) / (hi - lo) - 1
        self.widths = np.diff(self.edges)
        self.degree = int(degree)
        self.eff = np.ones(self.centres.size) if efficiency is None else np.asarray(efficiency, float)
        self.known = [gaussian_bins(self.edges, m, s) for m, s in known]

    def background(self, coeffs) -> np.ndarray:
        shape = np.exp(np.clip(np.polyval(coeffs[::-1], self.x), -700, 700)) * self.widths * self.eff
        return shape

    def expected(self, params, signal=None) -> np.ndarray:
        d = self.degree + 1
        mu = self.background(params[:d])
        for k, template in enumerate(self.known):
            mu = mu + params[d + k] * template * self.eff
        if signal is not None:
            mu = mu + params[-1] * signal * self.eff
        return mu

    def fit(self, counts, signal=None, start=None) -> Dict[str, object]:
        counts = np.asarray(counts, float)
        d, k = self.degree + 1, len(self.known)
        if start is None:  # the polynomial from a weighted fit of log counts, then the likelihood
            total = max(counts.sum(), 1.0)
            start = np.zeros(d + k + (signal is not None))
            good = (counts > 0) & (self.eff > 1e-3)
            if good.sum() > d:
                logs = np.log(counts[good] / (self.widths[good] * self.eff[good]))
                start[:d] = np.polynomial.polynomial.polyfit(self.x[good], logs, self.degree, w=np.sqrt(counts[good]))
            else:
                start[0] = np.log(total / max(np.sum(self.widths * self.eff), 1e-12))
            start[d:d + k] = 0.05 * total
        bounds = [(None, None)] * d + [(0, None)] * k + ([(0, None)] if signal is not None else [])

        def nll(p):
            return poisson_nll(counts, self.expected(p, signal))

        def grad(p):
            mu = np.clip(self.expected(p, signal), 1e-300, None)
            r = 1.0 - counts / mu
            g = np.empty(p.size)
            bkg = self.background(p[:d])
            for j in range(d):
                g[j] = np.sum(r * bkg * self.x ** j)
            for j, template in enumerate(self.known):
                g[d + j] = np.sum(r * template * self.eff)
            if signal is not None:
                g[-1] = np.sum(r * signal * self.eff)
            return g

        res = optimize.minimize(nll, start, jac=grad, method="L-BFGS-B", bounds=bounds,
                                options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-8})
        return {"params": res.x, "nll": float(res.fun), "expected": self.expected(res.x, signal),
                "converged": bool(res.success)}

    def scan(self, counts, sigma, masses=None) -> Dict[str, object]:
        """q0 and the fitted signal yield at each trial mass (Gaussian of width ``sigma``, a number or
        a function of the mass)."""
        counts = np.asarray(counts, float)
        masses = self.centres if masses is None else np.asarray(masses, float)
        b_only = self.fit(counts)
        q0 = np.zeros(masses.size)
        yields = np.zeros(masses.size)
        for i, m in enumerate(masses):
            s = gaussian_bins(self.edges, m, sigma(m) if callable(sigma) else sigma)
            start = np.concatenate([b_only["params"], [0.0]])
            sb = self.fit(counts, s, start=start)
            yields[i] = sb["params"][-1]
            q0[i] = max(0.0, 2 * (b_only["nll"] - sb["nll"])) if yields[i] > 0 else 0.0
        j = int(np.argmax(q0))
        return {"mass": masses, "q0": q0, "yield": yields, "background": b_only,
                "best_mass": float(masses[j]), "best_q0": float(q0[j]), "best_yield": float(yields[j]),
                "z_local": float(np.sqrt(q0[j]))}


def global_significance(scan: Dict[str, object], null_scans: Optional[Sequence[Sequence[float]]] = None,
                        level: float = 0.5) -> Dict[str, float]:
    """Global p-value of the scan's maximum by Gross-Vitells: upcrossings of ``level`` counted on
    ``null_scans`` if given, else on the scan itself (the usual shortcut; it counts the signal's own
    upcrossing too, so it errs conservative)."""
    q = float(scan["best_q0"])
    n0 = float(np.mean([upcrossings(s, level) for s in null_scans])) if null_scans else \
        float(upcrossings(scan["q0"], level))
    p_local = 0.5 * stats.chi2.sf(q, 1)  # one-sided: signal yield >= 0
    p_global = min(1.0, p_local + n0 * np.exp(-(q - level) / 2))
    return {"p_local": float(p_local), "z_local": float(np.sqrt(q)), "upcrossings": n0,
            "p_global": float(p_global), "z_global": float(stats.norm.isf(p_global)) if p_global < 1 else 0.0,
            "gross_vitells_two_sided": gross_vitells(q, n0, level)}


def asimov_z(s, b) -> float:
    """Median expected significance of a signal ``s`` over background ``b`` (Cowan et al. 2011)."""
    s, b = float(s), float(b)
    if s <= 0 or b <= 0:
        return 0.0
    return float(np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s)))
