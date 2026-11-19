"""Astrometric binaries: Gaia-like along-scan measurements of a star that may orbit a companion.

One run is one source: five astrometric parameters, an optional photocentre orbit, simulated
Gaia-like transits and noisy along-scan positions ``w``. The questions ask whether there is an
orbit, its period, and the parallax - which a single-star fit gets wrong when the orbit is not
modelled and its period is near a year.

Configuration (all optional):

``companion``
    ``"random"`` (default: an orbit with probability ``companion_probability``, 0.5), ``true`` or
    ``false``.
``orbit``
    Fixed orbit elements: ``period`` (days), ``eccentricity``, ``t_periastron`` (days from J2017.5),
    ``a0`` (mas), ``inclination``, ``omega``, ``node`` (degrees). Missing ones are drawn.
``period_range`` / ``signal_range`` / ``eccentricity_max``
    Draw ranges: period log-uniform in days (default [20, 2000]); a0 / ``sigma_al`` log-uniform
    (default [0.3, 10]); eccentricity uniform up to 0.7. Orientation is isotropic.
``parallax_range`` / ``proper_motion_sigma``
    Parallax uniform in mas (default [2, 20]); proper motion normal per axis (default 20 mas/yr).
``sky``
    ``"random"`` (isotropic, default) or ``[ra, dec]`` in degrees; sets the parallax factors.
``n_visits`` / ``span_years`` / ``sigma_al``
    Sampling: visits (default 25, two or three transits each) over the span (default 5.5 years,
    DR4's 66 months), and the along-scan standard error of one transit (default 0.2 mas).
``excess_noise``
    Extra white noise (mas) not included in the reported errors (``w_error``), as an unmodelled
    instrument or source effect would be. Default 0.
``detection_fap``
    False-alarm probability a detection is held to (default 1e-3); sets which companions are
    labelled ``signal_undetectable``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import stats

from scidatasim.base import ExperimentLogic
from scidatasim.utils import DataDescriptor, DataType

from .fit import (five_parameter_design, frequency_grid, independent_frequencies, orbit_design,
                  orbit_noncentrality, weighted_lstsq)
from .model import Epochs, Orbit, along_scan, campbell, simulated_epochs

#: Tolerances are what a fit of the true model reaches in this share of noise replicates.
TOLERANCE_QUANTILE = 0.9
#: The period answer is scored within this fraction of the true period.
PERIOD_TOLERANCE = 0.05


class AstrometricBinaryExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = ("Measurement equation, Kepler's equation and the Thiele-Innes elements are verified "
                     "(round trips, finite differences, recovery tests); the scanning law and Gaia's orbit are "
                     "simplified, so the sampling is Gaia-like, not Gaia's.")

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.companion = config.get("companion", "random")
        if self.companion not in ("random", True, False):
            raise ValueError("companion must be 'random', true or false")
        self.companion_probability = float(config.get("companion_probability", 0.5))
        self.fixed_orbit = dict(config.get("orbit") or {})
        self.period_range = tuple(float(v) for v in config.get("period_range", (20.0, 2000.0)))
        self.signal_range = tuple(float(v) for v in config.get("signal_range", (0.3, 10.0)))
        self.eccentricity_max = float(config.get("eccentricity_max", 0.7))
        self.parallax_range = tuple(float(v) for v in config.get("parallax_range", (2.0, 20.0)))
        self.pm_sigma = float(config.get("proper_motion_sigma", 20.0))
        self.sky = config.get("sky", "random")
        self.n_visits = int(config.get("n_visits", 25))
        self.span_years = float(config.get("span_years", 5.5))
        self.sigma_al = float(config.get("sigma_al", 0.2))
        self.excess_noise = float(config.get("excess_noise", 0.0))
        self.detection_fap = float(config.get("detection_fap", 1e-3))
        for name, (lo, hi) in (("period_range", self.period_range), ("signal_range", self.signal_range),
                               ("parallax_range", self.parallax_range)):
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be [low, high] with 0 < low <= high")
        if not 0 <= self.eccentricity_max < 1:
            raise ValueError("eccentricity_max must be in [0, 1)")
        if self.sigma_al <= 0 or self.excess_noise < 0:
            raise ValueError("sigma_al must be > 0 and excess_noise >= 0")
        self.rng = None
        self.epochs: Epochs = None
        self.astrometry: Dict[str, float] = {}
        self.orbit = None
        self.ra = self.dec = 0.0
        self._observed = None

    # ------------------------------------------------------------------ simulation
    def initialize(self, config):
        rng = self.get_rng(config)
        # draw order is fixed, so a given seed gives the same source whatever is fixed in the config
        if self.sky == "random":
            ra, dec = float(rng.uniform(0, 360)), float(np.degrees(np.arcsin(rng.uniform(-1, 1))))
        else:
            ra, dec = (float(v) for v in self.sky)
        self.ra, self.dec = ra, dec
        self.astrometry = {"ra_offset": float(rng.normal(0, 1.0)), "dec_offset": float(rng.normal(0, 1.0)),
                           "parallax": float(rng.uniform(*self.parallax_range)),
                           "pmra": float(rng.normal(0, self.pm_sigma)), "pmdec": float(rng.normal(0, self.pm_sigma))}
        has = bool(rng.uniform() < self.companion_probability) if self.companion == "random" else bool(self.companion)
        log_p = rng.uniform(*np.log(self.period_range))
        draws = {"period": float(np.exp(log_p)),
                 "eccentricity": float(rng.uniform(0, self.eccentricity_max)),
                 "a0": float(self.sigma_al * np.exp(rng.uniform(*np.log(self.signal_range)))),
                 "inclination": float(np.degrees(np.arccos(rng.uniform(-1, 1)))),
                 "omega": float(rng.uniform(0, 360)), "node": float(rng.uniform(0, 180))}
        draws["t_periastron"] = float(rng.uniform(-0.5, 0.5) * draws["period"])
        unknown = set(self.fixed_orbit) - set(draws)
        if unknown:
            raise ValueError(f"unknown orbit element(s) {sorted(unknown)}; use {sorted(draws)}")
        elements = {**draws, **{k: float(v) for k, v in self.fixed_orbit.items()}}
        self.elements = elements
        self.orbit = Orbit.from_campbell(elements["period"], elements["eccentricity"], elements["t_periastron"],
                                         elements["a0"], *np.radians([elements["inclination"], elements["omega"],
                                                                      elements["node"]])) if has else None
        ep = simulated_epochs(rng, ra, dec, self.n_visits, span_years=self.span_years, sigma=self.sigma_al)
        order = np.argsort(ep.t)
        self.epochs = Epochs(ep.t[order], ep.theta[order], ep.f_al[order], ep.sigma[order])
        noise = rng.normal(0.0, 1.0, size=len(self.epochs)) * np.sqrt(self.sigma_al ** 2 + self.excess_noise ** 2)
        self._latent = along_scan(self.epochs.t, self.epochs.theta, self.epochs.f_al, self.astrometry, self.orbit)
        self._observed = self._latent + noise
        return {}

    def run_step(self, state, step):
        return state

    def get_results(self):
        if self.epochs is None:
            self.initialize(self.config)
        ep = self.epochs
        orbit_part = np.zeros(len(ep)) if self.orbit is None else along_scan(ep.t, ep.theta, np.zeros(len(ep)), {},
                                                                            self.orbit)

        def series(name, data, role, units, plot=None):
            return {"data": np.asarray(data, float), "descriptor": DataDescriptor(
                name, DataType.NDARRAY, units=units, group="time_series", plot_type=plot,
                x_axis="t", role=role)}

        return {
            "t": {"data": ep.t.copy(), "descriptor": DataDescriptor("t", DataType.NDARRAY, units="yr from J2017.5",
                                                                     group="time_series", role="coordinate")},
            "w": series("w", self._latent, "latent", "mas"),
            "w_orbit": series("w_orbit", orbit_part, "latent", "mas"),
            "w_observed": series("w_observed", self._observed, "observation", "mas", "scatter"),
            "w_error": series("w_error", ep.sigma, "auxiliary", "mas"),
            "scan_angle": series("scan_angle", np.degrees(ep.theta), "auxiliary", "deg"),
            "parallax_factor_al": series("parallax_factor_al", ep.f_al, "auxiliary", "dimensionless"),
            "has_companion": {"data": "yes" if self.orbit is not None else "no",
                              "descriptor": DataDescriptor("has_companion", DataType.STRING, role="label")},
        }

    def ground_truth_parameters(self, config):
        params = super().ground_truth_parameters(config)
        params.update({f"astrometry.{k}": v for k, v in self.astrometry.items()})
        params.update({"sky.ra": self.ra, "sky.dec": self.dec, "has_companion": self.orbit is not None,
                       "n_transits": len(self.epochs)})
        if self.orbit is not None:
            params.update({f"orbit.{k}": v for k, v in self.elements.items()})
            params.update({f"orbit.{k}_thiele_innes": getattr(self.orbit, k) for k in "abfg"})
            params["orbit.noncentrality"] = orbit_noncentrality(self.epochs, self.orbit)
        return params

    # ------------------------------------------------------------------ bundle hooks
    def detection_threshold(self) -> float:
        """Delta chi2 the periodogram's highest peak must exceed for FAP < ``detection_fap``."""
        n_ind = independent_frequencies(self.epochs, frequency_grid(self.epochs))
        p_single = -np.expm1(np.log1p(-self.detection_fap) / n_ind)
        return float(stats.chi2.isf(p_single, 4))

    def signal_undetectable(self) -> bool:
        """True if the orbit is there but a search at the detection FAP finds it in fewer than half
        of noise replicates even at the true period (non-central chi2 with 4 degrees of freedom).
        Such a 'yes' cannot be read from the data; the label is private (it hints at the answer)."""
        if self.orbit is None:
            return False
        lam = orbit_noncentrality(self.epochs, self.orbit) * self.sigma_al ** 2 / (
            self.sigma_al ** 2 + self.excess_noise ** 2)
        return bool(stats.ncx2.sf(self.detection_threshold(), 4, lam) < 0.5)

    def parallax_error(self) -> float:
        """Standard error of the parallax from a weighted fit of the true model (the five parameters
        plus, for a binary, the orbit's Thiele-Innes constants at its true period, eccentricity and
        periastron time), with the actual noise."""
        design = five_parameter_design(self.epochs)
        if self.orbit is not None:
            design = np.column_stack([design, orbit_design(self.epochs, self.orbit.period, self.orbit.eccentricity,
                                                           self.orbit.t_periastron)])
        sigma = np.sqrt(self.epochs.sigma ** 2 + self.excess_noise ** 2)
        fit = weighted_lstsq(design, np.zeros(len(self.epochs)), sigma)
        return float(np.sqrt(fit.cov[2, 2]))

    def questions(self, results, rng) -> List[Dict[str, Any]]:
        undetectable = self.signal_undetectable()
        z = float(stats.norm.isf((1 - TOLERANCE_QUANTILE) / 2))
        items = [
            dict(template="astrometric_companion",
                 question="Besides parallax and proper motion, does the star's photocentre move on an orbit "
                          "(an unseen or faint companion)? Answer yes or no.",
                 answer="yes" if self.orbit is not None else "no", answer_type="choice", choices=["no", "yes"],
                 tolerance=None, tolerance_rule="exact", answerable_from="inference",
                 signal_undetectable=undetectable),
            dict(template="astrometric_period",
                 question="If the photocentre moves on an orbit, what is its period in days? Answer 0 if it does not.",
                 answer=float(self.orbit.period) if self.orbit is not None else 0.0, answer_type="number",
                 tolerance=PERIOD_TOLERANCE * float(self.orbit.period) if self.orbit is not None else 0.0,
                 tolerance_rule=f"{PERIOD_TOLERANCE:.0%} of the true period (exactly 0 when there is no orbit)",
                 answerable_from="inference", signal_undetectable=undetectable),
            dict(template="astrometric_parallax", question="Estimate the parallax in mas.",
                 answer=self.astrometry["parallax"], answer_type="number",
                 tolerance=z * self.parallax_error(),
                 tolerance_rule=(f"the error a weighted least-squares fit of the true model (orbit included) stays "
                                 f"within in {TOLERANCE_QUANTILE:.0%} of noise replicates"),
                 answerable_from="inference"),
        ]
        for item in items:
            item.setdefault("series", "w_observed")
        return items

    def inputs_for_baseline(self, results):
        """(Epochs with the reported errors, observed w) - what a solver is given."""
        ep = Epochs(results["t"]["data"], np.radians(results["scan_angle"]["data"]),
                    results["parallax_factor_al"]["data"], results["w_error"]["data"])
        return ep, results["w_observed"]["data"]

    @staticmethod
    def campbell_elements(orbit: Orbit) -> Dict[str, float]:
        c = campbell(orbit.a, orbit.b, orbit.f, orbit.g)
        return {"a0": c["a0"], "inclination": float(np.degrees(c["inclination"])),
                "omega": float(np.degrees(c["omega"])), "node": float(np.degrees(c["node"]))}
