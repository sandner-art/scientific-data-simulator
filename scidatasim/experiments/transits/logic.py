"""Transiting planets in a star's light curve: TESS-like photometry with or without a planet.

One run is one star observed for ``span`` days at ``cadence``: its relative flux (latent: the
transits times the star's own variability) and, from the observation operators, the measured flux
``flux_observed``. Configure the noise there: white noise (``gaussian``), red noise (``ar1``), gaps
(``missing``), or a real light curve (``real_noise``, ``combine: multiply``) - then the noise is
real and the transits are planted.

Configuration (all optional):

``planet``
    ``"random"`` (default: probability ``planet_probability``, 0.5), ``true`` or ``false``.
``period_range`` / ``rp_range`` / ``b_max``
    Draws: period log-uniform (days, default [0.5, 13]), Rp/R* log-uniform (default [0.01, 0.1]),
    impact parameter uniform in [0, b_max] (default 0.9); t0 uniform within the first period.
``orbit``
    Fixed values: ``period``, ``t0``, ``rp``, ``b``, ``u1``, ``u2``, ``density`` (g/cm^3, sets a/R*).
``density``, ``u1``, ``u2``
    Stellar density (default 1.41, the Sun) and quadratic limb darkening (default 0.4, 0.25).
``span``, ``cadence``, ``exposure``, ``gap``
    Days observed (default 27.4, a TESS sector), sampling (default 2 min), integration time
    (default: the cadence, supersampled 5 times when longer than 10 min), and a data gap
    ``[start, length]`` in days (default ``[13.2, 1.0]``, the mid-sector downlink; ``null`` for none).
``variability``
    Stellar rotation: ``{amplitude: 0.002, period: [2, 15]}`` (relative amplitude, days; drawn when a
    range). Default amplitude 0 (a quiet star).
``spots``
    Starspots that evolve (quasi-periodic, :func:`scidatasim.nuisance.quasi_periodic`):
    ``{amplitude: 0.01, period: [2, 10], lifetime: 15, harmonics: [1, 0.5, 0.25]}`` - RMS relative
    amplitude, rotation period (days; drawn log-uniform when a range), correlation time of the spot
    pattern (default three periods). A young star like AU Mic: amplitude about 0.015, period 4.86.
    The running-median detrending leaves such spots behind as dips.
``flares``
    ``{rate: 0.5, amplitude: [0.001, 0.02], index: 2, rise: 0.003, decay: 0.02}``: flares per day,
    peak relative amplitudes from a power law, rise and decay in days.
``ramps``
    Instrument settling at the start of the sector and after the data gap:
    ``{amplitude: 0.005, timescale: 0.3, sign: -1}`` (relative; ``[low, high]`` draws one per
    restart; ``sign: random``). The questions carry the private label ``instrument_artifact``.
``sigma_ppm``
    The per-point error the noise model reports (``flux_error``; default 1000 ppm). Keep it equal to
    the noise configured in ``observation``: it sets the tolerances and the detectability labels.

Questions: is there a transiting planet (private label ``signal_undetectable`` when the expected
box SNR is below 7.1), its period (0 if none; label ``period_ambiguous`` when fewer than two transits
fall in the data), and its depth in ppm (0 if none); ``instrument_artifact`` when ramps are planted.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import stats

from scidatasim.base import ExperimentLogic
from scidatasim.utils import DataDescriptor, DataType

from .model import Planet, a_over_rstar

SNR_THRESHOLD = 7.1
DEPTH_SHAPE_TOLERANCE = 0.10  # a box fit to a limb-darkened transit: mean vs largest decrement
TOLERANCE_QUANTILE = 0.9


class TransitExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = ("Transit shape: exact overlap areas (uniform star) and annulus integration that converges at "
                     "second order (limb darkening), checked against closed forms and the small-planet limit; "
                     "circular orbits; rotation as a sinusoid or evolving spots, flares, ramps after restarts. Noise "
                     "comes from the observation operators.")

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.planet_mode = config.get("planet", "random")
        if self.planet_mode not in ("random", True, False):
            raise ValueError("planet must be 'random', true or false")
        self.planet_probability = float(config.get("planet_probability", 0.5))
        self.period_range = tuple(float(v) for v in config.get("period_range", (0.5, 13.0)))
        self.rp_range = tuple(float(v) for v in config.get("rp_range", (0.01, 0.1)))
        self.b_max = float(config.get("b_max", 0.9))
        self.fixed = dict(config.get("orbit") or {})
        unknown = set(self.fixed) - {"period", "t0", "rp", "b", "u1", "u2", "density"}
        if unknown:
            raise ValueError(f"unknown orbit key(s) {sorted(unknown)}")
        self.span = float(config.get("span", 27.4))
        self.cadence = float(config.get("cadence", 2.0 / 1440))
        self.exposure = float(config.get("exposure", self.cadence))
        self.gap = config.get("gap", [13.2, 1.0])
        self.sigma_ppm = float(config.get("sigma_ppm", 1000.0))
        self.variability = dict(config.get("variability") or {})
        self.spots = dict(config.get("spots") or {})
        self.flare_config = dict(config.get("flares") or {})
        self.ramp_config = dict(config.get("ramps") or {})
        if self.ramp_config and "amplitude" not in self.ramp_config:
            raise ValueError("ramps needs an amplitude")
        if self.flare_config and "rate" not in self.flare_config:
            raise ValueError("flares needs a rate")
        if self.span <= 0 or self.cadence <= 0 or self.sigma_ppm <= 0:
            raise ValueError("span, cadence and sigma_ppm must be > 0")
        for name, (lo, hi) in (("period_range", self.period_range), ("rp_range", self.rp_range)):
            if not 0 < lo <= hi:
                raise ValueError(f"{name} must be [low, high] with 0 < low <= high")
        self.planet = None
        self.time = self.flux = self.rotation = None

    def initialize(self, config):
        rng = self.get_rng(config)
        t = np.arange(0.0, self.span, self.cadence)
        if self.gap:
            start, length = (float(v) for v in self.gap)
            t = t[(t < start) | (t >= start + length)]
        self.time = t
        has = bool(rng.uniform() < self.planet_probability) if self.planet_mode == "random" else bool(self.planet_mode)
        period = float(np.exp(rng.uniform(*np.log(self.period_range))))
        draws = {"period": period, "t0": float(rng.uniform(0, 1)), "rp": float(np.exp(rng.uniform(*np.log(self.rp_range)))),
                 "b": float(rng.uniform(0, self.b_max)), "u1": float(self.config.get("u1", 0.4)),
                 "u2": float(self.config.get("u2", 0.25)), "density": float(self.config.get("density", 1.41))}
        v = {**draws, **{k: float(x) for k, x in self.fixed.items()}}
        if "t0" not in self.fixed:
            v["t0"] = draws["t0"] * v["period"]
        self.elements = v
        self.planet = Planet(v["period"], v["t0"], v["rp"], a_over_rstar(v["period"], v["density"]), v["b"],
                             v["u1"], v["u2"]) if has else None
        amp = float(self.variability.get("amplitude", 0.0))
        prot = self.variability.get("period", [2.0, 15.0])
        prot = float(np.exp(rng.uniform(*np.log(prot)))) if isinstance(prot, (list, tuple)) else float(prot)
        phase = float(rng.uniform(0, 2 * np.pi))
        self.rotation = {"amplitude": amp, "period": prot, "phase": phase}
        star = 1.0 + amp * np.sin(2 * np.pi * t / prot + phase) + 0.3 * amp * np.sin(4 * np.pi * t / prot + 2 * phase)
        super_n = 5 if self.exposure > 10.0 / 1440 else 1
        self.transit = self.planet.flux(t, self.exposure, super_n) if self.planet else np.ones_like(t)
        self.activity, self.instrument = self._nuisances(t, rng)  # no draws unless configured
        self.flux = (star + self.activity) * self.transit * (1.0 + self.instrument)
        return {}

    def _nuisances(self, t, rng):
        from scidatasim.nuisance import flares, quasi_periodic, restart_ramps
        activity, instrument = np.zeros(t.size), np.zeros(t.size)
        self.nuisance_info = {}
        if self.spots:
            prot = self.spots.get("period", [2.0, 10.0])
            prot = float(np.exp(rng.uniform(*np.log(prot)))) if isinstance(prot, (list, tuple)) else float(prot)
            life = float(self.spots.get("lifetime", 3.0 * prot))
            activity += quasi_periodic(t, rng, float(self.spots.get("amplitude", 0.01)), prot, life,
                                       [float(v) for v in self.spots.get("harmonics", (1.0, 0.5, 0.25))])
            self.nuisance_info["spots"] = {"period": prot, "lifetime": life, "rms": float(np.std(activity))}
        if self.flare_config:
            f = self.flare_config
            extra, events = flares(t, rng, float(f["rate"]), tuple(float(v) for v in f.get("amplitude", (1e-3, 2e-2))),
                                   float(f.get("index", 2.0)), float(f.get("rise", 0.003)), float(f.get("decay", 0.02)))
            activity += extra
            self.nuisance_info["flares"] = events
        if self.ramp_config:
            r = self.ramp_config
            amp = r["amplitude"]
            amp = [float(v) for v in amp] if isinstance(amp, (list, tuple)) else float(amp)
            sign = r.get("sign", -1.0)
            min_gap = min(0.5, float(self.gap[1]) / 2) if self.gap else 0.5
            extra, events = restart_ramps(t, rng, amp, float(r.get("timescale", 0.3)), min_gap, True,
                                          None if sign == "random" else float(sign))
            instrument += extra
            self.nuisance_info["ramps"] = events
        return activity, instrument

    def run_step(self, state, step):
        return state

    def get_results(self):
        if self.time is None:
            self.initialize(self.config)
        t = self.time

        def series(name, data, role, units, plot=None):
            return {"data": np.asarray(data, float), "descriptor": DataDescriptor(
                name, DataType.NDARRAY, shape=np.shape(data), units=units, group="time_series", plot_type=plot,
                x_axis="time", role=role)}

        return {
            "time": {"data": t.copy(), "descriptor": DataDescriptor("time", DataType.NDARRAY, shape=t.shape, units="d",
                                                                     group="time_series", role="coordinate")},
            "flux": series("flux", self.flux, "latent", "relative flux", "line"),
            "transit_model": series("transit_model", self.transit, "latent", "relative flux"),
            "flux_error": series("flux_error", np.full(t.size, self.sigma_ppm * 1e-6), "auxiliary", "relative flux"),
            **({"stellar_activity": series("stellar_activity", self.activity, "latent", "relative flux")}
               if (self.spots or self.flare_config) else {}),
            **({"systematics": series("systematics", self.instrument, "latent", "relative flux")}
               if self.ramp_config else {}),
            "has_planet": {"data": "yes" if self.planet else "no",
                           "descriptor": DataDescriptor("has_planet", DataType.STRING, role="label")},
        }

    def ground_truth_parameters(self, config):
        params = super().ground_truth_parameters(config)
        params.update({"has_planet": self.planet is not None, "rotation.amplitude": self.rotation["amplitude"],
                       "rotation.period": self.rotation["period"], "n_points": int(self.time.size)})
        if self.nuisance_info.get("spots"):
            params.update({f"spots.{k}": v for k, v in self.nuisance_info["spots"].items()})
        if "flares" in self.nuisance_info:
            params["flares.n"] = len(self.nuisance_info["flares"])
        if "ramps" in self.nuisance_info:
            params["ramps"] = self.nuisance_info["ramps"]
        if self.planet:
            params.update({f"planet.{k}": v for k, v in self.elements.items()})
            params.update({"planet.a_over_rstar": self.planet.a, "planet.duration": self.planet.duration,
                           "planet.depth_ppm": self.depth_ppm(), "planet.n_transits": self.planet.transits_in(self.time),
                           "planet.snr": self.expected_snr()})
        return params

    # ------------------------------------------------------------------ labels and tolerances
    def depth_ppm(self) -> float:
        """Largest flux decrement of the transit (mid-transit), ppm."""
        if not self.planet:
            return 0.0
        mid = self.planet.flux(np.array([self.planet.t0]))[0]
        return float(1e6 * (1.0 - mid))

    def n_in_transit(self) -> int:
        return int(np.sum(self.planet.in_transit(self.time))) if self.planet else 0

    def expected_snr(self) -> float:
        """Box signal-to-noise of the noise-free transits: sqrt(sum over in-transit points of
        (1 - transit)^2) / sigma - the matched-filter SNR the data allow."""
        if not self.planet:
            return 0.0
        return float(np.sqrt(np.sum((1.0 - self.transit) ** 2)) / (self.sigma_ppm * 1e-6))

    def questions(self, results, rng) -> List[Dict[str, Any]]:
        has = self.planet is not None
        undetectable = has and self.expected_snr() < SNR_THRESHOLD
        single = has and self.planet.transits_in(self.time) < 2
        z = float(stats.norm.isf((1 - TOLERANCE_QUANTILE) / 2))
        series = "flux_observed" if "flux_observed" in results else "flux"
        artifact = {"instrument_artifact": True} if self.ramp_config else {}  # old runs keep their records
        if has:
            duration = self.planet.duration
            n_tr = max(self.planet.transits_in(self.time), 1)
            period_tol = max(duration / n_tr, 1e-3 * self.planet.period)
            n_in = max(self.n_in_transit(), 1)
            depth_tol = z * self.sigma_ppm / np.sqrt(n_in) + DEPTH_SHAPE_TOLERANCE * self.depth_ppm()
        return [
            dict(template="transit_present", question="Does the light curve show a transiting planet? Answer yes or no.",
                 answer="yes" if has else "no", answer_type="choice", choices=["no", "yes"], tolerance=None,
                 tolerance_rule="exact", answerable_from="inference", series=series, signal_undetectable=undetectable, **artifact),
            dict(template="transit_period", question="What is the orbital period of the transiting planet, in days? "
                 "Answer 0 if there is none.", answer=float(self.planet.period) if has else 0.0, answer_type="number",
                 tolerance=float(period_tol) if has else 0.0,
                 tolerance_rule="the period error that moves the last observed transit by less than one transit "
                                "duration (exactly 0 when there is no planet)",
                 answerable_from="inference", series=series, signal_undetectable=undetectable,
                 period_ambiguous=bool(single and not undetectable), **artifact),
            dict(template="transit_depth", question="What is the transit depth in ppm (the largest flux decrement, "
                 "relative to the flux outside transit)? Answer 0 if there is no transit.",
                 answer=self.depth_ppm(), answer_type="number", tolerance=float(depth_tol) if has else 0.0,
                 tolerance_rule=f"the {TOLERANCE_QUANTILE:.0%} error of a box depth with the true ephemeris plus "
                                f"{DEPTH_SHAPE_TOLERANCE:.0%} for the transit's shape (exactly 0 when there is none)",
                 answerable_from="inference", series=series, signal_undetectable=undetectable, **artifact),
        ]
