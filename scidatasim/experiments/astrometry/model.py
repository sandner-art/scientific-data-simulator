"""Gaia-like epoch astrometry: one along-scan position per field-of-view transit.

The measurement model follows the Gaia DR4 ``epoch_astrometry`` table (draft data model, June
2026): each transit gives the along-scan local-plane coordinate ``w`` of the source relative to a
reference point, the scan position angle ``theta`` and the along-scan parallax factor ``f_al``,
with

    w = (da + pmra * t) * sin(theta) + (dd + pmdec * t) * cos(theta) + parallax * f_al
        + (da_orb * sin(theta) + dd_orb * cos(theta))

where ``da`` is the offset in alpha* (= alpha cos delta) and ``t`` the time in Julian years from
the reference epoch (J2017.5 for DR4). A companion moves the photocentre on a Keplerian orbit
given by the Thiele-Innes elements, in Gaia's convention (as in ``nss_two_body_orbit``):

    da_orb = B X + G Y,    dd_orb = A X + F Y,
    X = cos E - e,         Y = sqrt(1 - e^2) sin E,

with E the eccentric anomaly. Units: mas, mas/yr, days for periods, years for ``t``.

The Gaia-like sampling here is a simplification (visits of a few transits, scan angles that change
between visits, a circular Earth-like orbit for the parallax factors). Real sampling comes from the
data: :mod:`scidatasim.experiments.astrometry.epochs` reads DR4 epoch astrometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

#: DR4 reference epoch (Julian year) and the DR4 time origin of ``obs_time_tcb`` (J2010.0).
REF_EPOCH = 2017.5
TCB_ORIGIN = 2010.0
DAYS_PER_YEAR = 365.25
#: Obliquity of the ecliptic (J2000), degrees.
OBLIQUITY = 23.4392911
#: Gaia's distance from the barycentre in au (L2 is about 1 % beyond Earth).
GAIA_DISTANCE_AU = 1.01


# ---------------------------------------------------------------------------- orbits
def kepler_eccentric_anomaly(mean_anomaly: np.ndarray, e: float, tol: float = 1e-12) -> np.ndarray:
    """Solves M = E - e sin E for E (Newton's method; e in [0, 1))."""
    if not 0.0 <= e < 1.0:
        raise ValueError("eccentricity must be in [0, 1)")
    m = np.mod(np.asarray(mean_anomaly, float), 2 * np.pi)
    ecc = m + e * np.sin(m) if e < 0.8 else np.full_like(m, np.pi)
    for _ in range(50):
        step = (ecc - e * np.sin(ecc) - m) / (1.0 - e * np.cos(ecc))
        ecc = ecc - step
        if np.max(np.abs(step)) < tol:
            break
    return ecc


def thiele_innes(a0: float, inclination: float, omega: float, node: float) -> Tuple[float, float, float, float]:
    """Campbell elements (a0 in mas; angles in radians: i, argument of periastron, ascending node)
    to the Thiele-Innes constants (A, B, F, G)."""
    co, so, cn, sn, ci = np.cos(omega), np.sin(omega), np.cos(node), np.sin(node), np.cos(inclination)
    a = a0 * (co * cn - so * sn * ci)
    b = a0 * (co * sn + so * cn * ci)
    f = -a0 * (so * cn + co * sn * ci)
    g = -a0 * (so * sn - co * cn * ci)
    return float(a), float(b), float(f), float(g)


def campbell(a: float, b: float, f: float, g: float) -> Dict[str, float]:
    """Thiele-Innes (A, B, F, G) back to a0 (mas), inclination, omega and node (radians),
    with the node in [0, pi) (astrometry alone cannot tell the two nodes apart)."""
    u = (a * a + b * b + f * f + g * g) / 2.0
    v = a * g - b * f
    a0 = float(np.sqrt(u + np.sqrt(max((u + v) * (u - v), 0.0))))
    inc = float(np.arccos(np.clip(v / a0 ** 2, -1.0, 1.0))) if a0 > 0 else 0.0
    plus = np.arctan2(b - f, a + g)
    minus = np.arctan2(-b - f, a - g)
    omega, node = (plus + minus) / 2.0, (plus - minus) / 2.0
    if node < 0:
        node, omega = node + np.pi, omega + np.pi
    if node >= np.pi:
        node, omega = node - np.pi, omega - np.pi
    return {"a0": a0, "inclination": inc, "omega": float(np.mod(omega, 2 * np.pi)), "node": float(node)}


def photocentre_a0(parallax: float, period_days: float, m1: float, m2: float, flux_ratio: float = 0.0) -> float:
    """Photocentre semi-major axis a0 (mas) of a binary: parallax (mas), period (days), masses
    (solar masses), flux ratio of the companion to the primary (0 for a dark companion or planet)."""
    a_au = ((period_days / DAYS_PER_YEAR) ** 2 * (m1 + m2)) ** (1.0 / 3.0)
    beta = flux_ratio / (1.0 + flux_ratio)
    return float(parallax * a_au * (m2 / (m1 + m2) - beta))


@dataclass
class Orbit:
    """A photocentre orbit: period (days), eccentricity, periastron time (days from the reference
    epoch, as ``t_periastron`` in DR4) and the Thiele-Innes constants (mas)."""
    period: float
    eccentricity: float
    t_periastron: float
    a: float
    b: float
    f: float
    g: float

    @classmethod
    def from_campbell(cls, period, eccentricity, t_periastron, a0, inclination, omega, node) -> "Orbit":
        return cls(period, eccentricity, t_periastron, *thiele_innes(a0, inclination, omega, node))

    @property
    def a0(self) -> float:
        return campbell(self.a, self.b, self.f, self.g)["a0"]

    def offsets(self, t_years: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Photocentre offsets (d alpha*, d delta) in mas at times ``t_years`` from the reference epoch."""
        days = np.asarray(t_years, float) * DAYS_PER_YEAR
        mean = 2 * np.pi * (days - self.t_periastron) / self.period
        ecc = kepler_eccentric_anomaly(mean, self.eccentricity)
        x = np.cos(ecc) - self.eccentricity
        y = np.sqrt(1.0 - self.eccentricity ** 2) * np.sin(ecc)
        return self.b * x + self.g * y, self.a * x + self.f * y


# ---------------------------------------------------------------------------- geometry
def gaia_position(t_years: np.ndarray) -> np.ndarray:
    """Approximate barycentric position of Gaia (au, ICRS axes) at ``t_years`` from J2017.5: a
    circular orbit in the ecliptic at 1.01 au. Enough for simulated parallax factors; real ones
    come from the data."""
    epoch = REF_EPOCH + np.asarray(t_years, float)
    lon = np.radians(100.46 + 360.0 * (epoch - 2000.0))  # Earth's mean longitude
    eps = np.radians(OBLIQUITY)
    x = GAIA_DISTANCE_AU * np.cos(lon)
    y = GAIA_DISTANCE_AU * np.sin(lon)
    return np.stack([x, y * np.cos(eps), y * np.sin(eps)], axis=-1)


def local_triad(ra_deg: float, dec_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Unit vectors p (towards increasing alpha) and q (towards increasing delta) at a sky position."""
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    p = np.array([-np.sin(ra), np.cos(ra), 0.0])
    q = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    return p, q


def parallax_factor_al(t_years: np.ndarray, theta: np.ndarray, ra_deg: float, dec_deg: float) -> np.ndarray:
    """Along-scan parallax factor: the derivative of w with respect to parallax."""
    p, q = local_triad(ra_deg, dec_deg)
    b = gaia_position(t_years)
    return -(b @ p) * np.sin(theta) - (b @ q) * np.cos(theta)


def along_scan(t_years, theta, f_al, astrometry: Dict[str, float], orbit: Optional[Orbit] = None) -> np.ndarray:
    """The along-scan position w (mas) of a source with five-parameter ``astrometry`` (``ra_offset``,
    ``dec_offset``, ``parallax``, ``pmra``, ``pmdec``; missing keys are 0) and an optional orbit."""
    t, s, c = np.asarray(t_years, float), np.sin(theta), np.cos(theta)
    da = astrometry.get("ra_offset", 0.0) + astrometry.get("pmra", 0.0) * t
    dd = astrometry.get("dec_offset", 0.0) + astrometry.get("pmdec", 0.0) * t
    w = da * s + dd * c + astrometry.get("parallax", 0.0) * np.asarray(f_al, float)
    if orbit is not None:
        oa, od = orbit.offsets(t)
        w = w + oa * s + od * c
    return w


# ---------------------------------------------------------------------------- sampling
@dataclass
class Epochs:
    """Transit-level sampling: times (years from J2017.5), scan angles (radians), along-scan
    parallax factors, and the per-transit standard error of w (mas)."""
    t: np.ndarray
    theta: np.ndarray
    f_al: np.ndarray
    sigma: np.ndarray

    def __len__(self) -> int:
        return int(self.t.size)


def simulated_epochs(rng: np.random.Generator, ra_deg: float = 45.0, dec_deg: float = 30.0,
                     n_visits: int = 30, transits_per_visit: Tuple[int, int] = (2, 3),
                     span_years: float = 5.5, sigma: float = 0.2) -> Epochs:
    """Gaia-like sampling, simplified: ``n_visits`` visits spread over ``span_years`` (centred on
    the reference epoch, as DR4's 66 months), each with two or three field-of-view transits a few
    hours apart under almost the same scan angle; the scan angle changes from visit to visit.
    ``sigma`` is the along-scan standard error of one transit (mas)."""
    starts = np.sort(rng.uniform(-span_years / 2, span_years / 2, size=n_visits))
    t, theta = [], []
    for start in starts:
        angle = rng.uniform(0, 2 * np.pi)
        k = int(rng.integers(transits_per_visit[0], transits_per_visit[1] + 1))
        hours = np.concatenate([[0.0], np.cumsum(rng.choice([1.775, 6.0], size=k - 1))])  # 106.5 min, 6 h
        t.extend(start + hours / (24 * DAYS_PER_YEAR))
        theta.extend(angle + rng.normal(0, np.radians(0.3), size=k))
    t, theta = np.array(t), np.mod(np.array(theta), 2 * np.pi)
    return Epochs(t, theta, parallax_factor_al(t, theta, ra_deg, dec_deg), np.full(t.size, float(sigma)))
