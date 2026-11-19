"""Distances in a flat (or curved) w0wa-CDM universe, and fits of supernova Hubble diagrams.

E(z)^2 = Om (1+z)^3 + Ok (1+z)^2 + Ode f(z),  f(z) = (1+z)^(3 (1 + w0 + wa)) exp(-3 wa z / (1+z))
(Chevallier-Polarski-Linder), Ode = 1 - Om - Ok. Comoving distance D_C = (c/H0) int_0^z dz'/E(z'),
integrated by Gauss-Legendre on a log(1+z) grid; luminosity distance D_L = (1 + z_hel) D_M(z_cmb),
the Pantheon+ convention; distance modulus mu = 5 log10(D_L / 10 pc).

Supernova magnitudes m = mu + M are fitted with M (degenerate with H0) marginalised analytically,
with a diagonal or full covariance - the standard SN-only likelihood.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import optimize

C_KM_S = 299792.458


def dark_energy(z, w0: float = -1.0, wa: float = 0.0) -> np.ndarray:
    z = np.asarray(z, float)
    return (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))


def e_of_z(z, om: float, w0: float = -1.0, wa: float = 0.0, ok: float = 0.0) -> np.ndarray:
    z = np.asarray(z, float)
    return np.sqrt(om * (1 + z) ** 3 + ok * (1 + z) ** 2 + (1 - om - ok) * dark_energy(z, w0, wa))


def comoving_distance(z, om: float, w0: float = -1.0, wa: float = 0.0, ok: float = 0.0, h0: float = 70.0,
                      n: int = 64) -> np.ndarray:
    """Line-of-sight comoving distance in Mpc (Gauss-Legendre in each interval between sorted
    redshifts; exact to ~1e-10 relative for smooth E(z))."""
    z = np.asarray(z, float)
    flat = z.ravel()
    order = np.argsort(flat)
    zs = np.concatenate([[0.0], flat[order]])
    x, w = np.polynomial.legendre.leggauss(n)
    a, b = zs[:-1], zs[1:]
    nodes = 0.5 * (b - a)[:, None] * x[None, :] + 0.5 * (b + a)[:, None]
    seg = 0.5 * (b - a) * (w[None, :] / e_of_z(nodes, om, w0, wa, ok)).sum(1)
    out = np.empty(flat.size)
    out[order] = np.cumsum(seg)
    return (C_KM_S / h0) * out.reshape(z.shape)


def transverse_distance(z, om, w0=-1.0, wa=0.0, ok=0.0, h0=70.0) -> np.ndarray:
    dc = comoving_distance(z, om, w0, wa, ok, h0)
    if abs(ok) < 1e-12:
        return dc
    dh = C_KM_S / h0
    k = np.sqrt(abs(ok)) / dh
    return np.sinh(k * dc) / k if ok > 0 else np.sin(k * dc) / k


def distance_modulus(z_cmb, om: float, w0: float = -1.0, wa: float = 0.0, ok: float = 0.0, h0: float = 70.0,
                     z_hel=None) -> np.ndarray:
    z_cmb = np.asarray(z_cmb, float)
    z_hel = z_cmb if z_hel is None else np.asarray(z_hel, float)
    dl = (1 + z_hel) * transverse_distance(z_cmb, om, w0, wa, ok, h0)
    return 5 * np.log10(dl) + 25


def _inverse(cov_or_err, n):
    a = np.asarray(cov_or_err, float)
    if a.ndim == 1 or a.size == 1:
        var = np.broadcast_to(a, (n,)) ** 2
        return np.diag(1.0 / var)
    return np.linalg.inv(a)


def chi2_marginal_m(m, mu_model, inv_cov) -> float:
    """chi2 with the absolute magnitude M marginalised analytically (flat prior):
    A - B^2 / C, with r = m - mu_model, A = r C^-1 r, B = sum(C^-1 r), C = sum(C^-1)."""
    r = np.asarray(m, float) - mu_model
    ir = inv_cov @ r
    return float(r @ ir - ir.sum() ** 2 / inv_cov.sum())


def fit(z, m, cov_or_err, model: str = "lcdm", z_hel=None, start: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Best fit of ``model``: ``lcdm`` (Om), ``wcdm`` (Om, w0) or ``w0wa`` (Om, w0, wa), flat,
    M marginalised. Returns parameters, chi2, and 1-sigma errors from the curvature (Hessian)."""
    z = np.asarray(z, float)
    inv = _inverse(cov_or_err, z.size)
    names = {"lcdm": ["om"], "wcdm": ["om", "w0"], "w0wa": ["om", "w0", "wa"]}[model]
    x0 = np.array([(start or {}).get(k, v) for k, v in zip(names, (0.3, -1.0, 0.0))])
    bounds = [(0.01, 0.99), (-3.0, 0.5), (-5.0, 3.0)][:len(names)]

    def chi2(x):
        p = dict(zip(names, x))
        mu = distance_modulus(z, p["om"], p.get("w0", -1.0), p.get("wa", 0.0), z_hel=z_hel)
        return chi2_marginal_m(m, mu, inv)

    res = optimize.minimize(chi2, x0, method="L-BFGS-B", bounds=bounds)
    best = res.x
    # errors from a finite-difference Hessian of chi2/2
    h = np.array([1e-3, 1e-3, 1e-2])[:len(names)]
    hess = np.zeros((len(names), len(names)))
    f0 = chi2(best)
    for i in range(len(names)):
        for j in range(i, len(names)):
            ei, ej = np.eye(len(names))[i] * h[i], np.eye(len(names))[j] * h[j]
            fpp, fpm = chi2(best + ei + ej), chi2(best + ei - ej)
            fmp, fmm = chi2(best - ei + ej), chi2(best - ei - ej)
            hess[i, j] = hess[j, i] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j]) / 2
    try:
        cov = np.linalg.inv(hess)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        err = np.full(len(names), np.nan)
    return {"model": model, "params": dict(zip(names, map(float, best))), "errors": dict(zip(names, map(float, err))),
            "chi2": float(f0), "n": int(z.size), "dof": int(z.size - len(names) - 1)}
