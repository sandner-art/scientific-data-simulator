"""A bump hunt: is there a new resonance in this invariant-mass spectrum?

One run is one spectrum: expected counts per mass bin (latent ``counts``) from a falling
Drell-Yan-like background, known resonances (PDG masses and widths), possibly a planted new one,
and the trigger efficiency; the measured counts come from the observation operators
(``{type: poisson}``). The signal strength is drawn as a target median significance (Asimov Z),
so the difficulty is controlled.

Configuration (all optional):

``window``, ``n_bins``, ``n_background``
    Mass range in GeV (default [12, 70]), bins (default 116, 0.5 GeV), expected background events
    (default 2e5).
``background``
    ``{power: 3.0, slope: 0.01}``: dN/dm proportional to m^-power exp(-slope m).
``resolution``
    Relative mass resolution sigma/m (default 0.01).
``known``
    Known resonances: names from ``KNOWN`` (J/psi, psi(2S), Upsilon(1S-3S), Z) or
    ``{mass, width, n}``; those inside the window are added (default: none).
``signal``
    ``"random"`` (default, probability ``signal_probability`` 0.5), ``true`` or ``false``; a planted
    resonance with mass uniform in the window's inner 80 %, natural width ``width`` (GeV, default 0)
    and a target Asimov Z drawn uniformly from ``z_range`` (default [0, 6]); ``mass`` / ``z`` fix them.
``trigger``
    ``{threshold: 20.0, width: 1.5, disclosed: true}``: efficiency 1 / (1 + exp(-(m - threshold) /
    width)) times the counts. Rising efficiency on a falling spectrum makes a peak that is not a
    particle; ``disclosed`` puts the efficiency in the question (an analysis knows its trigger).
    ``shape: erf`` (same parameters) or ``shape: two_body`` - both muons above transverse-momentum
    thresholds, the real dimuon trigger: ``{thresholds: [17, 8], turn_on: 1.0, pair_pt: 10,
    eta_max: 2.4, rapidity_sd: 2.0}`` (:func:`scidatasim.nuisance.two_body_efficiency`); its rise
    is not logistic. ``disclosed_as: logistic`` discloses (in the question and in the published
    ``trigger_efficiency``) the logistic closest to the true efficiency instead of the true one:
    the analysis' efficiency model is wrong in shape, as on the CMS data, where such a model
    leaves an 8-sigma "resonance".
``line_shape``
    ``{tail: {alpha: 1.0, n: 3}}``: peaks with a radiative tail below the mass (Crystal Ball
    response, :func:`scidatasim.nuisance.tailed_bins`) instead of a Gaussian resolution; ``side:
    high`` for a tail above. A Gaussian template then misses part of each known peak.
    ``{mixture: [[0.6, 0.75], [0.4, 1.4]]}``: the resolution is a mixture - weights and factors on
    ``resolution`` - as when it depends on the particles' direction (:func:`line_shape_bins`).
    ``{scale: -0.0013}``: the mass scale is miscalibrated, so every peak is measured at (1 + scale)
    times its mass (the questions keep the true masses; the known peaks are there to calibrate it,
    as on the CMS data, where the peaks sit 0.07-0.38 % low).

Questions: is there a new resonance (private label ``signal_undetectable`` when the Asimov Z is
below 3; ``instrument_artifact`` when a trigger turn-on is in the spectrum), its mass in GeV (0 if
none).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import special, stats

from scidatasim.base import ExperimentLogic
from scidatasim.baselines.counting import BinnedFit, asimov_z, global_significance
from scidatasim.utils import DataDescriptor, DataType

#: PDG 2024 masses and widths, GeV.
KNOWN = {"J/psi": (3.0969, 9.26e-5), "psi(2S)": (3.6861, 2.94e-4), "Upsilon(1S)": (9.4603, 5.4e-5),
         "Upsilon(2S)": (10.0233, 3.2e-5), "Upsilon(3S)": (10.3552, 2.0e-5), "Z": (91.1876, 2.4955)}
EVIDENCE_Z = 3.0
TOLERANCE_QUANTILE = 0.9
FINE = 20  # sub-bins per bin for integrating shapes


def voigt_bins(edges, mass: float, width: float, sigma: float) -> np.ndarray:
    """Fraction of a Breit-Wigner (FWHM ``width``) convolved with a Gaussian (``sigma``) in each bin."""
    edges = np.asarray(edges, float)
    if width <= 0:
        return np.diff(stats.norm.cdf(edges, mass, sigma))
    fine = np.linspace(edges[0], edges[-1], (edges.size - 1) * FINE + 1)
    mid = 0.5 * (fine[1:] + fine[:-1])
    dens = special.voigt_profile(mid - mass, sigma, width / 2) * np.diff(fine)
    return dens.reshape(edges.size - 1, FINE).sum(1)


def line_shape_bins(edges, mass: float, width: float, sigma: float, tail=None, mixture=None) -> np.ndarray:
    """Fraction of a peak in each bin: a Breit-Wigner (FWHM ``width``) seen with resolution
    ``sigma`` - Gaussian, or with ``tail`` a Crystal Ball - and with ``mixture`` (pairs
    ``[weight, factor]``) a weighted sum of such responses of widths ``factor * sigma``: a detector
    whose resolution depends on where the particles go (narrow in the barrel, wide in the
    endcaps)."""
    def one(s):
        if tail:
            from scidatasim.nuisance import tailed_bins
            return tailed_bins(edges, mass, s, tail, width)
        return voigt_bins(edges, mass, width, s)
    if not mixture:
        return one(sigma)
    w = np.array([float(m[0]) for m in mixture])
    f = np.array([float(m[1]) for m in mixture])
    if np.any(w < 0) or w.sum() <= 0 or np.any(f <= 0):
        raise ValueError("resolution mixture needs weights >= 0 (not all 0) and factors > 0")
    w = w / w.sum()
    return sum(wk * one(fk * sigma) for wk, fk in zip(w, f))


def background_bins(edges, power: float, slope: float) -> np.ndarray:
    edges = np.asarray(edges, float)
    fine = np.linspace(edges[0], edges[-1], (edges.size - 1) * FINE + 1)
    mid = 0.5 * (fine[1:] + fine[:-1])
    dens = mid ** -power * np.exp(-slope * mid) * np.diff(fine)
    out = dens.reshape(edges.size - 1, FINE).sum(1)
    return out / out.sum()


def efficiency(centres, threshold: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(centres, float) - threshold) / width))


def fit_logistic(centres, eff):
    """The logistic (threshold, width, plateau) closest to an efficiency curve in least squares:
    the model an analysis would fit to a measured trigger efficiency."""
    from scipy import optimize
    c, e = np.asarray(centres, float), np.asarray(eff, float)
    plateau0 = float(e.max()) or 1.0
    mid = float(c[np.argmin(np.abs(e - plateau0 / 2))])
    p, _ = optimize.curve_fit(lambda x, t, w, a: a * efficiency(x, t, w), c, e,
                              p0=[mid, max((c[-1] - c[0]) / 20, 0.1), plateau0], maxfev=20000)
    return float(p[0]), float(abs(p[1])), float(p[2])


class ResonanceExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = ("Phenomenological spectrum (power law times exponential, Breit-Wigner convolved with a Gaussian "
                     "or radiative-tail resolution, logistic, erf or two-muon trigger efficiency), integrated per bin "
                     "and checked against closed forms; not an event generator.")

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.window = tuple(float(v) for v in config.get("window", (12.0, 70.0)))
        self.n_bins = int(config.get("n_bins", 116))
        self.n_background = float(config.get("n_background", 2e5))
        bg = config.get("background") or {}
        self.power, self.slope = float(bg.get("power", 3.0)), float(bg.get("slope", 0.01))
        self.resolution = float(config.get("resolution", 0.01))
        self.signal_mode = config.get("signal", "random")
        if self.signal_mode not in ("random", True, False):
            raise ValueError("signal must be 'random', true or false")
        self.signal_probability = float(config.get("signal_probability", 0.5))
        self.z_range = tuple(float(v) for v in config.get("z_range", (0.0, 6.0)))
        self.width = float(config.get("width", 0.0))
        self.trigger = config.get("trigger")
        if self.trigger:
            self.trigger = dict(self.trigger)
            if self.trigger.get("shape", "logistic") not in ("logistic", "erf", "two_body"):
                raise ValueError("trigger shape must be 'logistic', 'erf' or 'two_body'")
            if self.trigger.get("disclosed_as") not in (None, "logistic"):
                raise ValueError("trigger disclosed_as must be 'logistic'")
        self.tail = (config.get("line_shape") or {}).get("tail")
        self.mixture = (config.get("line_shape") or {}).get("mixture")
        self.scale = float((config.get("line_shape") or {}).get("scale", 0.0))
        if not -0.1 < self.scale < 0.1:
            raise ValueError("line_shape scale is a relative miscalibration, |scale| < 0.1")
        if self.mixture:
            line_shape_bins(np.array([0.0, 1.0, 2.0]), 1.0, 0.0, 0.1, None, self.mixture)  # validates
        if not 0 < self.window[0] < self.window[1] or self.n_bins < 10 or self.resolution <= 0:
            raise ValueError("window must be [low, high] with 0 < low < high; n_bins >= 10; resolution > 0")
        self.known = []
        for k in config.get("known") or []:
            if isinstance(k, str):
                if k not in KNOWN:
                    raise ValueError(f"unknown resonance '{k}'; known: {sorted(KNOWN)}")
                m, w = KNOWN[k]
                self.known.append({"name": k, "mass": m, "width": w, "n": float(config.get("known_n", 2e4))})
            else:
                self.known.append({"name": k.get("name", f"m={k['mass']}"), "mass": float(k["mass"]),
                                   "width": float(k.get("width", 0.0)), "n": float(k.get("n", 2e4))})
        self.known = [k for k in self.known if self.window[0] < k["mass"] < self.window[1]]
        self.edges = np.linspace(*self.window, self.n_bins + 1)
        self.centres = 0.5 * (self.edges[1:] + self.edges[:-1])
        self.signal = None

    def sigma_at(self, mass: float) -> float:
        return self.resolution * mass

    def peak(self, mass: float, width: float) -> np.ndarray:
        """Share of a peak in each bin: Breit-Wigner and resolution, with the radiative tail and the
        resolution mixture when ``line_shape`` has them."""
        if self.scale:  # the detector's mass scale: every peak is measured at (1 + scale) times its mass
            mass = mass * (1.0 + self.scale)
        if self.tail or self.mixture:
            return line_shape_bins(self.edges, mass, width, self.sigma_at(mass), self.tail, self.mixture)
        return voigt_bins(self.edges, mass, width, self.sigma_at(mass))

    def _efficiency(self):
        """(true efficiency, disclosed efficiency, disclosed description)."""
        if not self.trigger:
            return np.ones(self.n_bins), np.ones(self.n_bins), ""
        shape = self.trigger.get("shape", "logistic")
        if shape == "logistic":  # unchanged arithmetic for runs made before the other shapes
            true = efficiency(self.centres, float(self.trigger.get("threshold", 20.0)), float(self.trigger.get("width", 1.5)))
        else:
            from scidatasim.observation import selection_efficiency
            true = selection_efficiency(self.centres, {k: v for k, v in self.trigger.items()
                                                       if k not in ("disclosed", "disclosed_as")})
        if self.trigger.get("disclosed_as") == "logistic":
            thr, w, plateau = fit_logistic(self.centres, true)
            self.disclosed_logistic = {"threshold": thr, "width": w, "plateau": plateau}
            return true, plateau * efficiency(self.centres, thr, w), (
                f"logistic around {thr:.3g} GeV, width {w:.3g} GeV, plateau {plateau:.3g}")
        if shape == "two_body":
            th = self.trigger.get("thresholds", (17.0, 8.0))
            return true, true, (f"two muons with transverse momenta above {float(max(th)):g} and {float(min(th)):g} GeV; "
                                "the efficiency per bin is given")
        return true, true, (f"{shape}, around {float(self.trigger.get('threshold', 20.0)):g} GeV, width "
                            f"{float(self.trigger.get('width', 1.5)):g} GeV")

    def initialize(self, config):
        rng = self.get_rng(config)
        eff, self.disclosed_eff, self.disclosed_text = self._efficiency()
        self.eff = eff
        background = self.n_background * background_bins(self.edges, self.power, self.slope)
        known = np.zeros(self.n_bins)
        for k in self.known:
            known += k["n"] * self.peak(k["mass"], k["width"])
        has = bool(rng.uniform() < self.signal_probability) if self.signal_mode == "random" else bool(self.signal_mode)
        lo, hi = self.window
        mass = float(self.config.get("mass", rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))))
        z_target = float(self.config.get("z", rng.uniform(*self.z_range)))
        shape = self.peak(mass, self.width)
        core = np.abs(self.centres - mass) <= 2 * np.hypot(self.sigma_at(mass), self.width / 2.355)
        b_core = float(np.sum((background + known)[core] * eff[core]))
        frac = float(np.sum((shape * eff)[core]))
        n_sig = 0.0
        if has and z_target > 0:  # signal events whose Asimov Z over the core's background is z_target
            lo_n, hi_n = 0.0, 10 * z_target * np.sqrt(b_core) / max(frac, 1e-9) + 10.0
            for _ in range(100):
                mid = 0.5 * (lo_n + hi_n)
                lo_n, hi_n = (mid, hi_n) if asimov_z(mid * frac, b_core) < z_target else (lo_n, mid)
            n_sig = 0.5 * (lo_n + hi_n)
        self.signal = {"present": has, "mass": mass, "z_target": z_target if has else 0.0, "n": n_sig,
                       "width": self.width, "b_core": b_core, "s_core": n_sig * frac}
        self.truth_counts = (background + known + (n_sig * shape if has else 0.0)) * eff
        self.background_counts = background * eff
        return {}

    def run_step(self, state, step):
        return state

    def get_results(self):
        if self.signal is None:
            self.initialize(self.config)
        desc = lambda name, role, units, plot=None: DataDescriptor(  # noqa: E731
            name, DataType.NDARRAY, shape=self.centres.shape, units=units, group="spectrum", plot_type=plot,
            x_axis="mass", role=role)
        return {
            "mass": {"data": self.centres.copy(), "descriptor": DataDescriptor(
                "mass", DataType.NDARRAY, shape=self.centres.shape, units="GeV", group="spectrum", role="coordinate")},
            "counts": {"data": self.truth_counts.copy(), "descriptor": desc("counts", "latent", "events / bin", "line")},
            "background_counts": {"data": self.background_counts.copy(), "descriptor": desc(
                "background_counts", "latent", "events / bin")},
            "trigger_efficiency": {"data": self.disclosed_eff.copy(), "descriptor": desc("trigger_efficiency", "auxiliary", "")},
            **({"true_trigger_efficiency": {"data": self.eff.copy(), "descriptor": desc(
                "true_trigger_efficiency", "latent", "")}} if self.trigger and self.trigger.get("disclosed_as") else {}),
            "has_new_resonance": {"data": "yes" if self.signal["present"] else "no",
                                  "descriptor": DataDescriptor("has_new_resonance", DataType.STRING, role="label")},
        }

    def ground_truth_parameters(self, config):
        params = super().ground_truth_parameters(config)
        params.update({f"signal.{k}": v for k, v in self.signal.items()})
        params["signal.asimov_z"] = self.asimov()
        params["known"] = ", ".join(k["name"] for k in self.known)
        return params

    def asimov(self) -> float:
        return asimov_z(self.signal["s_core"], self.signal["b_core"]) if self.signal["present"] else 0.0

    def mass_tolerance(self) -> float:
        """The 90 % error of the peak position: sigma_eff / sqrt(s) inflated by sqrt(1 + b/s) for the
        background under the peak, but at least half a bin."""
        s, b = max(self.signal["s_core"], 1.0), self.signal["b_core"]
        sigma = np.hypot(self.sigma_at(self.signal["mass"]), self.width / 2.355)
        z = float(stats.norm.isf((1 - TOLERANCE_QUANTILE) / 2))
        return float(max(z * sigma / np.sqrt(s) * np.sqrt(1 + b / s), 0.5 * (self.edges[1] - self.edges[0])))

    def questions(self, results, rng) -> List[Dict[str, Any]]:
        has = self.signal["present"] and self.signal["n"] > 0
        undetectable = has and self.asimov() < EVIDENCE_Z
        series = "counts_observed" if "counts_observed" in results else "counts"
        known = ", ".join(f"{k['name']} at {k['mass']:g} GeV" for k in self.known) or "none in this window"
        trig = ""
        if self.trigger and self.trigger.get("disclosed", True):
            if self.trigger.get("shape", "logistic") == "logistic" and not self.trigger.get("disclosed_as"):
                trig = (f" The trigger efficiency rises around {float(self.trigger.get('threshold', 20.0)):g} GeV "
                        f"(logistic, width {float(self.trigger.get('width', 1.5)):g} GeV).")
            else:
                trig = f" The trigger efficiency: {self.disclosed_text}."
        context = f" Known resonances: {known}.{trig} Mass resolution: {self.resolution:.1%} of the mass."
        return [
            dict(template="new_resonance", question="Is there a new resonance in the spectrum, beyond the known ones?"
                 + context + " Answer yes or no.", answer="yes" if has else "no", answer_type="choice",
                 choices=["no", "yes"], tolerance=None, tolerance_rule="exact", answerable_from="inference",
                 series=series, signal_undetectable=bool(undetectable), instrument_artifact=bool(self.trigger)),
            dict(template="resonance_mass", question="What is the mass of the new resonance, in GeV? Answer 0 if "
                 "there is none." + context, answer=float(self.signal["mass"]) if has else 0.0, answer_type="number",
                 tolerance=self.mass_tolerance() if has else 0.0,
                 tolerance_rule=f"the {TOLERANCE_QUANTILE:.0%} error of the peak position given the signal and the "
                                "background under it, at least half a bin (exactly 0 when there is none)",
                 answerable_from="inference", series=series, signal_undetectable=bool(undetectable),
                 instrument_artifact=bool(self.trigger)),
        ]


def bump_hunt(edges, counts, resolution: float, efficiency=None, known=(), degree: int = 3,
              z_threshold: float = EVIDENCE_Z, tail=None, mixture=None) -> Dict[str, Any]:
    """The baseline: background fit (exp of a cubic times the efficiency, known peaks with free
    yields), a Gaussian signal scanned over the bin centres with width ``resolution`` x mass, and
    the global significance by Gross-Vitells; detected when the global Z exceeds ``z_threshold``.
    ``tail`` (``{alpha, n, side}``) gives the known peaks a radiative tail (Crystal Ball) and
    ``mixture`` a resolution mixture (see :func:`line_shape_bins`)."""
    if tail or mixture:
        templates = [line_shape_bins(edges, m, 0.0, resolution * m, tail, mixture) for m in known]
    else:
        templates = [(m, resolution * m) for m in known]
    fit = BinnedFit(edges, degree=degree, efficiency=efficiency, known=templates)
    lo, hi = fit.edges[0], fit.edges[-1]
    inner = fit.centres[(fit.centres > lo + 0.05 * (hi - lo)) & (fit.centres < hi - 0.05 * (hi - lo))]
    scan = fit.scan(counts, lambda m: resolution * m, masses=inner)
    sig = global_significance(scan)
    return {"mass": scan["best_mass"], "z_local": sig["z_local"], "z_global": sig["z_global"],
            "p_global": sig["p_global"], "yield": scan["best_yield"], "detected": bool(sig["z_global"] > z_threshold),
            "scan": scan}
