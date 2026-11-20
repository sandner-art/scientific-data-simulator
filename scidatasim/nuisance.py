# scidatasim/nuisance.py
"""Nuisance models: what an instrument or a source adds to a signal, beyond white noise.

The open-data validations (``tools/validate_tess.py``, ``tools/validate_cms_dimuon.py``) found
where simple simulations are optimistic: spotted stars, ramps after an instrument restarts, a
trigger efficiency whose shape is not logistic, and the radiative tail below a mass peak. The
models here reproduce them. They are plain functions of a coordinate and a random generator, so
the science packs use them and the observation operators (``quasi_periodic``, ``flares``,
``ramps``, ``efficiency``, ``response``) put them on any experiment's series.

Line shapes
    :func:`crystal_ball_pdf`, :func:`crystal_ball_cdf`, :func:`tailed_bins`: a Gaussian core with
    a power-law tail on one side (Crystal Ball). Low side: energy lost before the measurement -
    final-state radiation in a mass peak, incomplete charge collection in a gamma-ray detector.
    High side: peak tailing in chromatography, pile-up.
Selection efficiencies
    :func:`logistic_efficiency`, :func:`erf_efficiency`, and :func:`two_body_efficiency` - the
    efficiency, as a function of the parent's mass, of requiring both daughters of a two-body decay
    above momentum thresholds (a dimuon trigger). Its rise is not logistic: it starts at the sum of
    the thresholds and approaches the plateau slowly. Any threshold applied to the parts of a system
    rather than the system itself behaves like this.
Correlated variability
    :func:`quasi_periodic`: harmonics of a period whose amplitudes and phases drift with a
    correlation time (starspots that evolve; seasonal cycles that vary from year to year; machine
    vibration). :func:`flares`: sudden rises with exponential decays at random times (stellar
    flares, bursts, spikes).
Instrument transients
    :func:`restart_ramps`: an exponential settling after each restart - after a gap in the
    coordinate and at the start (thermal settling after a downlink or a safe mode, sensor warm-up,
    drift after recalibration). :func:`gap_edges` and :func:`trim_after_gaps` find the restarts and
    drop the data right after them, the usual remedy.

All amplitudes are in the units of the series they are added to; the packs use them as relative
flux.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import special

# ------------------------------------------------------------------------------- line shapes


def _cb_constants(alpha: float, n: float):
    alpha, n = float(alpha), float(n)
    if alpha <= 0 or n <= 1:
        raise ValueError("Crystal Ball needs alpha > 0 and n > 1")
    log_a = n * np.log(n / alpha) - alpha ** 2 / 2  # in logarithms: (n / alpha)^n overflows for large n
    b = n / alpha - alpha
    tail = n / alpha / (n - 1) * np.exp(-alpha ** 2 / 2)  # integral of the tail, in units of sigma
    core = np.sqrt(np.pi / 2) * (1 + special.erf(alpha / np.sqrt(2)))
    return log_a, b, tail, core


def crystal_ball_pdf(x, mean: float, sigma: float, alpha: float, n: float, side: str = "low") -> np.ndarray:
    """Density of a Gaussian (``mean``, ``sigma``) whose ``side`` beyond ``alpha`` sigma is
    replaced by a power law of index ``n``; normalised to 1."""
    t = (np.asarray(x, float) - mean) / sigma
    if side == "high":
        t = -t
    elif side != "low":
        raise ValueError("side must be 'low' or 'high'")
    log_a, b, tail, core = _cb_constants(alpha, n)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        power = np.exp(log_a - n * np.log(np.maximum(b - t, 1e-300)))
        out = np.where(t > -alpha, np.exp(-t ** 2 / 2), power)
    return out / (sigma * (tail + core))


def crystal_ball_cdf(x, mean: float, sigma: float, alpha: float, n: float, side: str = "low") -> np.ndarray:
    """Cumulative distribution of :func:`crystal_ball_pdf` (closed form)."""
    t = (np.asarray(x, float) - mean) / sigma
    flip = side == "high"
    if flip:
        t = -t
    elif side != "low":
        raise ValueError("side must be 'low' or 'high'")
    log_a, b, tail, core = _cb_constants(alpha, n)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        low = np.exp(log_a + (1 - n) * np.log(np.maximum(b - t, 1e-300))) / (n - 1)
        high = tail + np.sqrt(np.pi / 2) * (special.erf(t / np.sqrt(2)) + special.erf(alpha / np.sqrt(2)))
    out = np.where(t <= -alpha, low, high) / (tail + core)
    out = np.clip(out, 0.0, 1.0)
    return 1.0 - out if flip else out


def tailed_bins(edges, mean: float, sigma: float, tail: Dict[str, float], width: float = 0.0,
                fine: int = 20) -> np.ndarray:
    """Fraction of a peak in each bin: a Crystal Ball response (``tail``: ``alpha``, ``n``,
    ``side``) of resolution ``sigma``, convolved with a Breit-Wigner of FWHM ``width`` when > 0
    (numerically, on ``fine`` sub-bins per bin; a width below 5 % of ``sigma`` is neglected, as
    it changes the shape by less than that)."""
    edges = np.asarray(edges, float)
    alpha, n, side = float(tail.get("alpha", 1.5)), float(tail.get("n", 3.0)), tail.get("side", "low")
    if width <= 0.05 * sigma:
        return np.diff(crystal_ball_cdf(edges, mean, sigma, alpha, n, side))
    from scipy.signal import fftconvolve
    grid = np.linspace(edges[0], edges[-1], (edges.size - 1) * fine + 1)
    mid = 0.5 * (grid[1:] + grid[:-1])
    h = grid[1] - grid[0]
    half = width / 2
    # true masses on the same step, extended so that what smears into the window is counted
    pad = int(np.ceil(max(20 * half, 10 * sigma * (n if side == "low" else 1.0)) / h))
    m_true = mid[0] + h * np.arange(-pad, mid.size + pad)
    bw = half / np.pi / ((m_true - mean) ** 2 + half ** 2) * h
    k = h * np.arange(-(mid.size + pad), mid.size + pad + 1)  # measured minus true
    kernel = crystal_ball_pdf(k, 0.0, sigma, alpha, n, side) * h
    dens = fftconvolve(bw, kernel)  # index i + j -> m_true[i] + k[j]
    start = pad + (mid.size + pad)  # where measured mass mid[0] sits in the full convolution
    out = np.clip(dens[start:start + mid.size], 0.0, None)
    return out.reshape(edges.size - 1, fine).sum(1)


def response_matrix(centres, sigma, tail: Optional[Dict[str, float]] = None) -> np.ndarray:
    """``R[i, j]``: the share of what is true at bin ``j`` that is measured in bin ``i`` (bins with
    edges halfway between the centres), for a Gaussian or tailed response of width ``sigma[j]``."""
    c = np.asarray(centres, float)
    if c.ndim != 1 or c.size < 2 or np.any(np.diff(c) <= 0):
        raise ValueError("response needs increasing bin centres")
    edges = np.concatenate([[c[0] - (c[1] - c[0]) / 2], (c[1:] + c[:-1]) / 2, [c[-1] + (c[-1] - c[-2]) / 2]])
    s = np.broadcast_to(np.asarray(sigma, float), c.shape)
    if np.any(s <= 0):
        raise ValueError("response sigma must be > 0")
    if tail:
        alpha, n, side = float(tail.get("alpha", 1.5)), float(tail.get("n", 3.0)), tail.get("side", "low")
        cdf = np.stack([crystal_ball_cdf(edges, cj, sj, alpha, n, side) for cj, sj in zip(c, s)], axis=1)
    else:
        cdf = special.ndtr((edges[:, None] - c[None, :]) / s[None, :])
    return np.diff(cdf, axis=0)


# ---------------------------------------------------------------------- selection efficiencies


def logistic_efficiency(x, threshold: float, width: float, plateau: float = 1.0) -> np.ndarray:
    return plateau / (1.0 + np.exp(-np.clip((np.asarray(x, float) - threshold) / width, -60, 60)))


def erf_efficiency(x, threshold: float, width: float, plateau: float = 1.0) -> np.ndarray:
    return plateau * 0.5 * (1.0 + special.erf((np.asarray(x, float) - threshold) / (np.sqrt(2) * width)))


def two_body_efficiency(mass, thresholds: Sequence[float] = (17.0, 8.0), turn_on: float = 1.0,
                        pair_pt: float = 10.0, eta_max: float = 2.4, rapidity_sd: float = 2.0,
                        daughter_mass: float = 0.1057, n: int = 20000, seed: int = 0) -> np.ndarray:
    """Share of decays of a parent of each ``mass`` whose daughters pass momentum thresholds:
    the leading daughter's transverse momentum above ``thresholds[0]``, the other's above
    ``thresholds[1]`` (each an erf turn-on of width ``turn_on``), both within |eta| < ``eta_max``.
    Parents: transverse momentum exponential with mean ``pair_pt``, rapidity N(0, ``rapidity_sd``),
    isotropic decays. Monte Carlo with ``n`` decays shared by every mass (common random numbers,
    fixed ``seed``), so the curve is smooth and the same on every call. Units: GeV."""
    m = np.atleast_1d(np.asarray(mass, float))
    if len(thresholds) != 2 or turn_on <= 0 or n < 100:
        raise ValueError("two_body_efficiency needs two thresholds, turn_on > 0 and n >= 100")
    rng = np.random.default_rng(seed)
    pt = rng.exponential(pair_pt, n) if pair_pt > 0 else np.zeros(n)
    y = rng.normal(0.0, rapidity_sd, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_t = rng.uniform(-1, 1, n)
    phi_d = rng.uniform(0, 2 * np.pi, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    hi_t, lo_t = max(thresholds), min(thresholds)
    out = np.empty(m.size)
    for i, mi in enumerate(m):
        if mi <= 2 * daughter_mass:
            out[i] = 0.0
            continue
        p = np.sqrt(mi ** 2 / 4 - daughter_mass ** 2)
        e = mi / 2
        # daughter in the parent's rest frame
        px, py, pz = p * sin_t * np.cos(phi_d), p * sin_t * np.sin(phi_d), p * cos_t
        # parent four-momentum in the lab
        mt = np.sqrt(mi ** 2 + pt ** 2)
        big_e, big_px, big_py, big_pz = mt * np.cosh(y), pt * np.cos(phi), pt * np.sin(phi), mt * np.sinh(y)
        bx, by, bz = big_px / big_e, big_py / big_e, big_pz / big_e
        b2 = bx ** 2 + by ** 2 + bz ** 2
        gamma = big_e / mi
        legs = []
        for sign in (1.0, -1.0):
            qx, qy, qz = sign * px, sign * py, sign * pz
            bp = bx * qx + by * qy + bz * qz
            k = np.where(b2 > 0, (gamma - 1) * bp / np.where(b2 > 0, b2, 1.0), 0.0) + gamma * e
            lx, ly, lz = qx + k * bx, qy + k * by, qz + k * bz
            lpt = np.hypot(lx, ly)
            eta = np.arcsinh(lz / np.maximum(lpt, 1e-12))
            legs.append((lpt, np.abs(eta) < eta_max))
        lead = np.maximum(legs[0][0], legs[1][0])
        sub = np.minimum(legs[0][0], legs[1][0])
        accepted = legs[0][1] & legs[1][1]
        w = erf_efficiency(lead, hi_t, turn_on) * erf_efficiency(sub, lo_t, turn_on)
        out[i] = float(np.mean(w * accepted))
    return out


# ---------------------------------------------------------------------- correlated variability


def quasi_periodic(t, rng: np.random.Generator, amplitude: float, period: float, lifetime: float,
                   harmonics: Sequence[float] = (1.0, 0.5, 0.25)) -> np.ndarray:
    """Variability with RMS ``amplitude``: harmonics k = 1, 2, ... of ``period`` with relative
    weights ``harmonics``, each with a complex amplitude that drifts as an Ornstein-Uhlenbeck
    process of correlation time ``lifetime`` (spots that grow, move and decay). A long lifetime
    gives a strictly periodic signal; a short one, red noise."""
    t = np.asarray(t, float)
    if amplitude < 0 or period <= 0 or lifetime <= 0:
        raise ValueError("quasi_periodic needs amplitude >= 0, period > 0, lifetime > 0")
    weights = np.asarray(harmonics, float)
    weights = weights / np.sqrt(np.sum(weights ** 2))
    out = np.zeros(t.size)
    if t.size == 0 or amplitude == 0:
        return out
    order = np.argsort(t)
    ts = t[order]
    rho = np.exp(-np.diff(ts) / lifetime)
    scale = np.sqrt(1.0 - rho ** 2)
    for k, w in enumerate(weights, start=1):
        z = rng.normal(size=(ts.size, 2)) / np.sqrt(2)  # complex N(0, 1)
        c = np.empty(ts.size, complex)
        c[0] = z[0, 0] + 1j * z[0, 1]
        zc = z[1:, 0] + 1j * z[1:, 1]
        for i in range(1, ts.size):
            c[i] = rho[i - 1] * c[i - 1] + scale[i - 1] * zc[i - 1]
        out[order] += np.sqrt(2) * w * amplitude * np.real(c * np.exp(2j * np.pi * k * ts / period))
    return out


def flares(t, rng: np.random.Generator, rate: float, amplitude: Tuple[float, float] = (1e-3, 2e-2),
           index: float = 2.0, rise: float = 0.003, decay: float = 0.02) -> Tuple[np.ndarray, list]:
    """Flares at random times (Poisson, ``rate`` per coordinate unit): peak amplitudes from a power
    law dN/dA ~ A^-index between ``amplitude``, a Gaussian rise of ``rise`` and an exponential
    decay of ``decay`` (coordinate units; a decay time grows with the amplitude). Returns the
    added series and the list of flares (time, amplitude)."""
    t = np.asarray(t, float)
    if rate < 0 or rise <= 0 or decay <= 0:
        raise ValueError("flares need rate >= 0, rise > 0 and decay > 0")
    lo_a, hi_a = (float(v) for v in amplitude)
    if not 0 < lo_a <= hi_a:
        raise ValueError("flare amplitude must be [low, high] with 0 < low <= high")
    out = np.zeros(t.size)
    if t.size == 0 or rate == 0:
        return out, []
    t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
    count = rng.poisson(rate * (t1 - t0))
    times = np.sort(rng.uniform(t0, t1, count))
    u = rng.uniform(size=count)
    if abs(index - 1.0) < 1e-9:
        amps = lo_a * (hi_a / lo_a) ** u
    else:
        g = 1.0 - index
        amps = (lo_a ** g + u * (hi_a ** g - lo_a ** g)) ** (1.0 / g)
    events = []
    for tf, a in zip(times, amps):
        tau = decay * (a / lo_a) ** 0.25
        dt = t - tf
        shape = np.where(dt < 0, np.exp(-0.5 * (dt / rise) ** 2), np.exp(-np.clip(dt, 0, None) / tau))
        out += a * shape
        events.append({"time": float(tf), "amplitude": float(a), "decay": float(tau)})
    return out, events


# ---------------------------------------------------------------------- instrument transients


def gap_edges(t, min_gap: float = 0.5) -> np.ndarray:
    """Coordinates where data resume: the first point and every point after a gap longer than
    ``min_gap``."""
    t = np.sort(np.asarray(t, float)[np.isfinite(t)])
    if t.size == 0:
        return np.zeros(0)
    jumps = np.nonzero(np.diff(t) > min_gap)[0]
    return np.concatenate([[t[0]], t[jumps + 1]])


def restart_ramps(t, rng: np.random.Generator, amplitude, timescale: float = 0.3, min_gap: float = 0.5,
                  at_start: bool = True, sign: Optional[float] = -1.0) -> Tuple[np.ndarray, list]:
    """An exponential settling after each restart: ``A exp(-(t - t_restart) / timescale)`` from
    every point that follows a gap longer than ``min_gap`` (and from the first point when
    ``at_start``). ``amplitude`` is a value or a range ``[low, high]`` drawn per restart; ``sign``
    -1 (default) makes dips, +1 rises, None a random sign per restart. Returns the added series and
    the list of ramps."""
    t = np.asarray(t, float)
    if timescale <= 0:
        raise ValueError("ramps need timescale > 0")
    starts = gap_edges(t, min_gap)
    if not at_start and starts.size:
        starts = starts[1:]
    out = np.zeros(t.size)
    events = []
    for s in starts:
        a = float(rng.uniform(*amplitude)) if isinstance(amplitude, (list, tuple)) else float(amplitude)
        sg = float(rng.choice([-1.0, 1.0])) if sign is None else float(sign)
        dt = t - s
        out += np.where(dt >= 0, sg * a * np.exp(-np.clip(dt, 0, None) / timescale), 0.0)
        events.append({"start": float(s), "amplitude": sg * a, "timescale": float(timescale)})
    return out, events


def trim_after_gaps(t, after: float, min_gap: float = 0.5, before: float = 0.0) -> np.ndarray:
    """Boolean mask that drops the ``after`` coordinate units following each restart (see
    :func:`gap_edges`) and the ``before`` units preceding each gap and the end."""
    t = np.asarray(t, float)
    keep = np.isfinite(t)
    if after > 0:
        for s in gap_edges(t, min_gap):
            keep &= ~((t >= s) & (t < s + after))
    if before > 0:
        tt = np.sort(t[np.isfinite(t)])
        if tt.size:
            ends = np.concatenate([tt[:-1][np.diff(tt) > min_gap], [tt[-1]]])
            for e in ends:
                keep &= ~((t <= e) & (t > e - before))
    return keep
