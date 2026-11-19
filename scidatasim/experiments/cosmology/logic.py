"""A supernova Hubble diagram: is dark energy a cosmological constant?

One run is one sample of type Ia supernovae: redshifts, the true distance moduli (latent ``mu``) of
a flat w0wa-CDM universe, and the standardised peak magnitudes (latent ``m_b`` = mu + M, plus a
calibration offset for the low-redshift sample if configured); the measured magnitudes come from
the observation operators (``{type: gaussian, sigma: 0.15}``: measurement plus intrinsic scatter).

Configuration (all optional):

``n_sne``, ``redshifts``
    Sample size (default 1000) drawn Pantheon+-like (a quarter at 0.01-0.1, log-uniform; the rest
    log-normal around 0.35, up to 2.3); or ``redshifts: {source: file, column: zHD}`` to take the
    redshifts of a real sample (any format ``scidatasim.ingest`` reads; ``min_z`` default 0.01).
``dark_energy``
    ``"random"`` (default: probability ``evolving_probability`` 0.5 of evolving), ``"lcdm"`` or
    ``"w0wa"``; ``om``, ``w0``, ``wa`` fix values, else drawn: Om U[0.2, 0.4], w0 U[-1.3, -0.6],
    wa U[-1.5, 0.5] (w0 + wa < 0, so dark energy dominated early on is excluded).
``calibration``
    ``{offset: 0.03, split_z: 0.1}``: magnitude offset of the supernovae below ``split_z`` - the
    low-redshift samples come from other surveys and their cross-calibration is uncertain at a few
    hundredths of a magnitude. Such an offset bends the Hubble diagram like evolving dark energy.
``sigma_m``
    The per-supernova error reported in ``m_b_error`` (default 0.15); keep it equal to the noise in
    ``observation``: it sets the tolerances and labels.

Questions: is dark energy a cosmological constant (private label ``signal_undetectable`` when the
noise-free data separate the true w0wa model from the best LCDM fit by less than Delta chi2 = 11.8,
3 sigma for two parameters; ``instrument_artifact`` when a calibration offset is planted), Om, and
w0. Tolerances: the 90 % errors of a fit of the true model family (Om) and of the w0wa family (w0),
from the curvature of chi2 at the truth.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy import stats

from scidatasim.base import ExperimentLogic
from scidatasim.utils import DataDescriptor, DataType

from .model import chi2_marginal_m, distance_modulus, fit

ABSOLUTE_M = -19.25
EVIDENCE_DCHI2 = 11.8  # 3 sigma for 2 parameters
TOLERANCE_QUANTILE = 0.9


class SupernovaHubbleExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = ("Distances by Gauss-Legendre integration, checked against closed forms (Einstein-de Sitter, "
                     "LCDM, w0wa) and astropy; the SN-only likelihood with M marginalised reproduces Pantheon+'s "
                     "published Om (tools/validate_pantheon.py). Supernova physics is not simulated: magnitudes "
                     "are standardised.")

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        self.n = int(config.get("n_sne", 1000))
        self.mode = config.get("dark_energy", "random")
        if self.mode not in ("random", "lcdm", "w0wa"):
            raise ValueError("dark_energy must be 'random', 'lcdm' or 'w0wa'")
        self.sigma_m = float(config.get("sigma_m", 0.15))
        cal = config.get("calibration") or {}
        self.offset, self.split_z = float(cal.get("offset", 0.0)), float(cal.get("split_z", 0.1))
        if self.n < 10 or self.sigma_m <= 0:
            raise ValueError("n_sne must be >= 10 and sigma_m > 0")
        self.z = None

    def _redshifts(self, rng) -> np.ndarray:
        spec = self.config.get("redshifts")
        if spec:
            from scidatasim import ingest
            s = ingest.read_series(spec["source"], value=spec.get("column", "z"), fmt=spec.get("format"))
            z = s.value[np.isfinite(s.value) & (s.value >= float(spec.get("min_z", 0.01)))]
            self.redshift_source = s.provenance
            return np.sort(z)
        n_low = self.n // 4
        low = np.exp(rng.uniform(np.log(0.01), np.log(0.1), n_low))
        high = np.clip(np.exp(rng.normal(np.log(0.35), 0.6, self.n - n_low)), 0.1, 2.3)
        return np.sort(np.concatenate([low, high]))

    def initialize(self, config):
        rng = self.get_rng(config)
        self.redshift_source = None
        self.z = self._redshifts(rng)
        evolving = bool(rng.uniform() < float(self.config.get("evolving_probability", 0.5))) if self.mode == "random" \
            else self.mode == "w0wa"
        om = float(self.config.get("om", rng.uniform(0.2, 0.4)))
        w0, wa = float(rng.uniform(-1.3, -0.6)), float(rng.uniform(-1.5, 0.5))
        while w0 + wa >= 0:
            wa = float(rng.uniform(-1.5, 0.5))
        if not evolving:
            w0, wa = -1.0, 0.0
        w0, wa = float(self.config.get("w0", w0)), float(self.config.get("wa", wa))
        self.truth = {"om": om, "w0": w0, "wa": wa, "evolving": bool(w0 != -1.0 or wa != 0.0)}
        self.mu = distance_modulus(self.z, om, w0, wa)
        self.low = self.z < self.split_z
        self.m_b = self.mu + ABSOLUTE_M + np.where(self.low, self.offset, 0.0)
        return {}

    def run_step(self, state, step):
        return state

    def get_results(self):
        if self.z is None:
            self.initialize(self.config)

        def series(name, data, role, units, plot=None):
            return {"data": np.asarray(data, float), "descriptor": DataDescriptor(
                name, DataType.NDARRAY, shape=np.shape(data), units=units, group="hubble_diagram", plot_type=plot,
                x_axis="z", role=role)}

        return {
            "z": {"data": self.z.copy(), "descriptor": DataDescriptor("z", DataType.NDARRAY, shape=self.z.shape,
                                                                       units="", group="hubble_diagram",
                                                                       role="coordinate")},
            "mu": series("mu", self.mu, "latent", "mag"),
            "m_b": series("m_b", self.m_b, "latent", "mag", "scatter"),
            "m_b_error": series("m_b_error", np.full(self.z.size, self.sigma_m), "auxiliary", "mag"),
            "low_z_sample": series("low_z_sample", self.low.astype(float), "auxiliary", ""),
            "dark_energy": {"data": "evolving" if self.truth["evolving"] else "cosmological constant",
                            "descriptor": DataDescriptor("dark_energy", DataType.STRING, role="label")},
        }

    def ground_truth_parameters(self, config):
        params = super().ground_truth_parameters(config)
        params.update({f"cosmology.{k}": v for k, v in self.truth.items()})
        params.update({"absolute_magnitude": ABSOLUTE_M, "calibration.offset": self.offset,
                       "calibration.split_z": self.split_z, "n_sne": int(self.z.size),
                       "lcdm_dchi2_noise_free": self.lcdm_dchi2()})
        if self.redshift_source:
            params.update({f"redshifts.{k}": self.redshift_source[k] for k in ("path", "sha256")})
        return params

    # ------------------------------------------------------------------ labels and tolerances
    def lcdm_dchi2(self) -> float:
        """chi2 of the best flat LCDM fit to the noise-free magnitudes (0 when the truth is LCDM and
        nothing is miscalibrated): how strongly the data could reject a cosmological constant."""
        if not hasattr(self, "_lcdm_dchi2"):
            self._lcdm_dchi2 = fit(self.z, self.m_b, self.sigma_m, "lcdm")["chi2"]
        return self._lcdm_dchi2

    def _errors_at_truth(self, family: str) -> Dict[str, float]:
        names = ["om"] if family == "lcdm" else ["om", "w0", "wa"]
        inv = np.eye(self.z.size) / self.sigma_m ** 2
        x0 = np.array([self.truth[k] for k in names])
        h = np.array([1e-3, 1e-3, 1e-2])[:len(names)]

        def chi2(x):
            p = dict(zip(names, x))
            return chi2_marginal_m(self.m_b, distance_modulus(self.z, p["om"], p.get("w0", -1.0), p.get("wa", 0.0))
                                   + ABSOLUTE_M + np.where(self.low, self.offset, 0.0), inv)

        hess = np.zeros((len(names),) * 2)
        for i in range(len(names)):
            for j in range(i, len(names)):
                ei, ej = np.eye(len(names))[i] * h[i], np.eye(len(names))[j] * h[j]
                hess[i, j] = hess[j, i] = (chi2(x0 + ei + ej) - chi2(x0 + ei - ej) - chi2(x0 - ei + ej)
                                           + chi2(x0 - ei - ej)) / (8 * h[i] * h[j])
        err = np.sqrt(np.clip(np.diag(np.linalg.pinv(hess)), 0, None))
        return dict(zip(names, map(float, err)))

    def questions(self, results, rng) -> List[Dict[str, Any]]:
        evolving = self.truth["evolving"]
        undetectable = evolving and self.lcdm_dchi2() < EVIDENCE_DCHI2
        artifact = self.offset != 0.0
        z = float(stats.norm.isf((1 - TOLERANCE_QUANTILE) / 2))
        own = self._errors_at_truth("w0wa" if evolving else "lcdm")
        general = own if evolving else self._errors_at_truth("w0wa")
        series = "m_b_observed" if "m_b_observed" in results else "m_b"
        note = (" Magnitudes are standardised; the absolute magnitude is unknown (degenerate with H0). "
                f"Supernovae below z = {self.split_z:g} come from separately calibrated low-redshift surveys.")
        return [
            dict(template="cosmological_constant", question="Is dark energy a cosmological constant (w = -1 at all "
                 "times), or does it evolve?" + note + " Answer 'cosmological constant' or 'evolving'.",
                 answer="evolving" if evolving else "cosmological constant", answer_type="choice",
                 choices=["cosmological constant", "evolving"], tolerance=None, tolerance_rule="exact",
                 answerable_from="inference", series=series, signal_undetectable=bool(undetectable),
                 instrument_artifact=artifact),
            dict(template="omega_m", question="What is the matter density Omega_m (flat universe)?" + note,
                 answer=self.truth["om"], answer_type="number", tolerance=z * own["om"],
                 tolerance_rule=f"the {TOLERANCE_QUANTILE:.0%} error of a fit of the true model family",
                 answerable_from="inference", series=series, instrument_artifact=artifact),
            dict(template="w0", question="What is the dark-energy equation of state today, w0 (CPL w0-wa "
                 "parametrisation)?" + note, answer=self.truth["w0"], answer_type="number",
                 tolerance=z * general["w0"],
                 tolerance_rule=f"the {TOLERANCE_QUANTILE:.0%} error of w0 in a w0-wa fit",
                 answerable_from="inference", series=series, instrument_artifact=artifact),
        ]


def compare_models(z, m, err) -> Dict[str, object]:
    """The baseline: flat LCDM and flat w0wa fits; 'evolving' when Delta chi2 > 11.8 (3 sigma, 2 dof)."""
    lcdm, w0wa = fit(z, m, err, "lcdm"), fit(z, m, err, "w0wa")
    d = lcdm["chi2"] - w0wa["chi2"]
    return {"lcdm": lcdm, "w0wa": w0wa, "dchi2": float(d), "evolving": bool(d > EVIDENCE_DCHI2),
            "om": lcdm["params"]["om"] if d <= EVIDENCE_DCHI2 else w0wa["params"]["om"],
            "w0": w0wa["params"]["w0"]}
