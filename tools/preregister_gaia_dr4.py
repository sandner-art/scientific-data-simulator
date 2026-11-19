"""Pre-registration record for the Gaia DR4 astrometric-companion test.

Builds a JSON record that fixes, before Gaia DR4 is public (2 December 2026, 12:00 CET), what
will be done with its epoch astrometry and what is predicted:

* the source selection (rules and ADQL; numbers that depend on the release, such as percentiles,
  are fixed as rules and resolved on release day),
* the method (every constant of the detection baseline in ``scidatasim.experiments.astrometry``),
* the predictions, computed here by simulation with fixed seeds,
* the hypotheses: which comparisons are confirmatory, with their pass bands, and which exploratory,
* the analysis after the release and the rule for reporting deviations,
* provenance: git commit, SHA-256 of every file the method depends on, package versions,
* ``record_sha256``: SHA-256 of the canonical JSON of everything above.

Usage, from the repository root (a clean checkout of the commit to register)::

    python tools/preregister_gaia_dr4.py --out preregistration_gaia_dr4.json      # about 40 minutes
    python tools/preregister_gaia_dr4.py --verify preregistration_gaia_dr4.json

``--verify`` recomputes the record's hash and compares the method files with the current checkout.
Deposit the JSON (Zenodo) and tag the commit before the deadline; the record's own timestamp is not
evidence, the deposit's is.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

SCHEMA = "scidatasim.preregistration/1"
DEADLINE_UTC = "2026-12-02T11:00:00Z"  # 12:00 CET
METHOD_FILES = ("scidatasim/experiments/astrometry/__init__.py", "scidatasim/experiments/astrometry/model.py",
                "scidatasim/experiments/astrometry/fit.py", "scidatasim/experiments/astrometry/epochs.py",
                "scidatasim/experiments/astrometry/logic.py", "tools/check_astrometry_detection.py",
                "tools/preregister_gaia_dr4.py")
G_BINS = ((8.0, 10.0), (10.0, 12.0), (12.0, 14.0), (14.0, 16.0), (16.0, 17.0))
PER_BIN = 600

CONTROL_ADQL = """SELECT TOP {n} g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.parallax, g.parallax_over_error,
       g.ruwe, g.visibility_periods_used, g.astrometric_matched_transits, g.astrometric_n_obs_al,
       g.astrometric_n_good_obs_al, g.scan_direction_strength_k1, g.scan_direction_strength_k2,
       g.ipd_frac_multi_peak, g.ipd_gof_harmonic_amplitude, g.duplicated_source, g.random_index
FROM gaiadr4.gaia_source AS g
JOIN gaiadr4.all_source_flags AS f ON f.source_id = g.source_id
WHERE f.has_epoch_astrometry = 'true'
  AND g.phot_g_mean_mag >= {g_lo} AND g.phot_g_mean_mag < {g_hi}
  AND g.astrometric_matched_transits >= {matched_transits_p25}
  AND g.scan_direction_strength_k1 <= {k1_p75} AND g.scan_direction_strength_k2 <= {k2_p75}
  AND g.duplicated_source = 'false'
ORDER BY g.random_index"""

PERCENTILE_ADQL = """SELECT g.astrometric_matched_transits, g.scan_direction_strength_k1, g.scan_direction_strength_k2
FROM gaiadr4.gaia_source AS g
JOIN gaiadr4.all_source_flags AS f ON f.source_id = g.source_id
WHERE f.has_epoch_astrometry = 'true' AND g.phot_g_mean_mag >= {g_lo} AND g.phot_g_mean_mag < {g_hi}
  AND g.random_index < 2000000"""


def method() -> dict:
    from scidatasim.experiments.astrometry import epochs, fit, logic
    return {
        "measurement_model": "w = (da + pmra t) sin(theta) + (dd + pmdec t) cos(theta) + parallax f_al + orbit "
                             "(Thiele-Innes: da = B X + G Y, dd = A X + F Y); t in Julian years from J2017.5",
        "input": "DR4 epoch_astrometry, CCD level; CCDs with used_by_agis_al; barycentric time "
                 "obs_time_tcb + obs_time_bary_corr; parallax_factor_al of the transit",
        "weights": "centroid_pos_error_al only (primary); centroid_pos_error_al^2 + agis_source_excess_noise^2 "
                   "reported as a secondary line",
        "gross_outliers_mas": epochs.GROSS_LIMIT_MAS,
        "gross_outlier_rule": "drop CCDs farther than the limit from a least-absolute-deviation five-parameter fit",
        "periodogram": {"model": "five parameters + circular orbit (4 linear constants) per trial period",
                        "min_period_days": 10.0, "max_period": "1.5 x time span", "oversample": 5.0,
                        "grid": "uniform in frequency"},
        "false_alarm_probability": "1 - (1 - p)^N, p = chi2 survival (4 dof) of the highest Delta chi2, "
                                   "N = span x frequency range",
        "detection_threshold_fap": 1e-3,
        "keplerian": {"candidates": "3 highest periodogram peaks and twice their periods (1 % merge)",
                      "grid": "period +/- 2 % (9), eccentricity 0-0.9 (8), periastron time (16)",
                      "polish": "Nelder-Mead in (log P, e, T), e < 0.95; lowest chi2 kept"},
        "parallax_estimate": "Keplerian fit's when an orbit is detected, else the five-parameter fit's",
        "period_tolerance": logic.PERIOD_TOLERANCE,
        "noncentrality": "Delta chi2 of the noise-free orbit over the best five-parameter fit (fit.orbit_noncentrality)",
        "functions": [f"{fit.__name__}.{name}" for name in ("detect", "best_keplerian", "orbit_noncentrality",
                                                            "excess_noise")]
        + [f"{epochs.__name__}.{name}" for name in ("read_epoch_astrometry", "source_epochs")],
    }


def selection() -> dict:
    return {
        "controls": {
            "purpose": "real epochs and real noise for injection-recovery, the noise-model check and the "
                       "sampling-exact null; not assumed to be single stars",
            "size": f"{PER_BIN} per G bin, {PER_BIN * len(G_BINS)} in all",
            "g_bins": [list(b) for b in G_BINS],
            "rules": [
                "has_epoch_astrometry",
                "astrometric_matched_transits at or above the bin's 25th percentile (sampling completeness)",
                "scan_direction_strength_k1 and _k2 at or below the bin's 75th percentile (scan-angle coverage)",
                "duplicated_source false",
                "first sources by random_index (reproducible random subset)",
            ],
            "not_cut": ["ruwe", "astrometric_excess_noise", "parallax_over_error", "ipd_frac_multi_peak",
                        "presence in nss_two_body_orbit"],
            "why_not_ruwe": "RUWE and the excess noise are computed from residuals of the very signal tested; a cut "
                            "on them selects on the outcome and removes real companions. They are recorded and used "
                            "only for stratified (exploratory) reports.",
            "percentiles": "from PERCENTILE_ADQL on release day (random_index < 2,000,000 per bin), recorded with "
                           "the selection",
            "adql": CONTROL_ADQL, "percentile_adql": PERCENTILE_ADQL,
        },
        "known_companions": {
            "size": "all that pass, expected a few hundred",
            "rules": ["DR3 nss_two_body_orbit astrometric solutions (Orbital, AstroSpectroSB1) with significance > 20 "
                      "and period < 1500 d",
                      "Gaia BH1, BH2, BH3",
                      "planets and brown dwarfs from the NASA Exoplanet Archive with a predicted a0 > 0.1 mas",
                      "has_epoch_astrometry in DR4"],
        },
        "population": {"rule": "aggregate TAP counts of DR4 nss_two_body_orbit by period and G bin; no bulk download"},
        "auxiliary": ["DR4 nss_two_body_orbit and spurious_signal rows of all selected sources (TAP_UPLOAD join)"],
    }


def hypotheses(pred: dict) -> list:
    """Confirmatory hypotheses with pass bands derived from the predictions, and exploratory ones."""
    comp = {r["non-centrality"]: r for r in pred["by_noncentrality"]}
    high = [r for r in pred["by_noncentrality"] if r["non-centrality"] in ("50-80", "80-150", "150-inf")]
    period_high = sum(r["final period within 5 %"] * r["n"] for r in high if r["final period within 5 %"] is not None) \
        / max(1, sum(r["n"] for r in high))
    return [
        {"id": "H1", "kind": "confirmatory", "statement": "The FAP approximation is not anti-conservative on DR4 "
         "sampling: in the sampling-exact null (the controls' real epochs with simulated Gaussian noise at their "
         "reported errors, 20 replicates each) the share with FAP < 0.01 is at most 0.02 and with FAP < 0.001 at "
         "most 0.004 (simulation: 0.011 and 0.002).", "predicted": pred["false_alarms"]},
        {"id": "H2", "kind": "confirmatory", "statement": "The reported CCD errors describe the noise: the median "
         "unit-weight error of the controls' five-parameter fits (measurement weights) lies in [0.9, 1.2] in every "
         "G bin.", "fallback": "If not, errors are rescaled per G bin by that median, H3-H5 are reported with and "
         "without the rescaling, and H2 is reported as failed."},
        {"id": "H3", "kind": "confirmatory", "statement": "Injection-recovery on the controls' real epochs "
         "(orbits drawn as in AstrometricBinaryExperiment, one per control, injected into the measured w): in each "
         "non-centrality bin with at least 30 injections, (a) the detection rate (FAP < 0.001) and (b) the "
         "recovery rate (detected and the final period within 5 % of the injected one) lie within max(0.05, 2 "
         "binomial standard errors) of the predicted rates.",
         "predicted": {k: {"detected": v["detected"],
                           "recovered": None if v["detected"] is None else
                           v["detected"] * (v["final period within 5 %"] or 0.0)} for k, v in comp.items()}},
        {"id": "H4", "kind": "confirmatory", "statement": "Weights with the source excess noise (AGIS) detect fewer "
         "injected orbits than measurement weights for non-centrality 20-80 (one-sided, paired by source, McNemar "
         "test at 0.05).", "predicted": {k: comp[k]["AGIS weights"] for k in ("20-35", "35-50", "50-80")}},
        {"id": "H5", "kind": "confirmatory", "statement": "Known companions from DR3 with DR4 non-centrality "
         "(from their DR3 orbit) of at least 50: the final period lies within 5 % of the DR3 period for at least "
         f"{max(0.0, period_high - 0.1):.2f} of them (the predicted {period_high:.2f} minus 0.10).",
         "predicted": period_high},
        {"id": "E1", "kind": "exploratory", "statement": "DPAC's circular-orbit periodogram (spurious_signal."
         "astro_circ_freq, astro_circ_freq_fap) against ours on the same sources: agreement of peak frequencies, "
         "rank correlation of FAPs, false-alarm rates. Exploratory because the draft data model leaves these "
         "columns' definitions open (TODO)."},
        {"id": "E2", "kind": "exploratory", "statement": "Companion fraction among controls, estimated as "
         "(detection rate - false-alarm rate) / completeness, by G bin; stratified by RUWE."},
        {"id": "E3", "kind": "exploratory", "statement": "Comparison with DR4's own orbit solutions "
         "(nss_two_body_orbit) for sources both detect: period, a0, eccentricity."},
        {"id": "E4", "kind": "exploratory", "statement": "Parallax shift between the five-parameter fit and the "
         "pipeline for detected orbits, by period (the simulation predicts the largest near one year)."},
    ]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def provenance(args) -> dict:
    import numpy
    import scipy
    import scidatasim
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", *METHOD_FILES)),
            "scidatasim": getattr(scidatasim, "__version__", "unknown"), "python": platform.python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "files_sha256": {p: sha256_file(os.path.join(ROOT, p)) for p in METHOD_FILES},
            "simulation": {"nulls": args.nulls, "binaries": args.binaries,
                           "seeds": "null runs 100000.., binary runs 200000.. (tools/check_astrometry_detection.py)",
                           "model_defaults": "AstrometricBinaryExperiment defaults (simplified Gaia-like sampling, "
                                             "sigma_al 0.2 mas per transit)"}}


def canonical(record: dict) -> bytes:
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=float).encode("utf-8")


def build(args) -> dict:
    import check_astrometry_detection as check
    nulls = [check.null_run(check.NULL_SEED0 + i) for i in range(args.nulls)]
    bins = [check.binary_run(check.BINARY_SEED0 + i) for i in range(args.binaries)]
    pred = check.summarise(nulls, bins)
    record = {
        "schema": SCHEMA,
        "title": "Detection of astrometric companions in Gaia DR4 epoch astrometry: pre-registered method, "
                 "selection and predictions",
        "author": args.author, "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deadline_utc": DEADLINE_UTC,
        "data": {"release": "Gaia DR4 (public 2026-12-02, 12:00 CET)",
                 "tables": ["gaia_source", "all_source_flags", "epoch_astrometry (DataLink, EPOCH_ASTROMETRY)",
                            "nss_two_body_orbit", "spurious_signal"],
                 "data_model": "Gaia DR4 draft data model, issue D (2026-05-20); names checked on release day, any "
                               "renaming reported as a deviation",
                 "credit": "ESA/Gaia/DPAC"},
        "selection": selection(), "method": method(), "predictions": pred, "hypotheses": hypotheses(pred),
        "analysis_after_release": [
            "Resolve the selection percentiles; run the selection ADQL; store ADQL, IDs, files and SHA-256.",
            "Fetch epoch astrometry (DataLink, batches of 1,000) and the auxiliary rows.",
            "Run the method above unchanged (the registered commit) on: the sampling-exact null (H1), the controls' "
            "five-parameter fits (H2), injections into the controls (H3, H4), the known companions (H5).",
            "Report every hypothesis, passed or failed, with the numbers; then the exploratory analyses, labelled so.",
        ],
        "deviations": "Any change to selection, method or analysis after this record is reported with its reason and "
                      "the result under the registered version where it can be computed.",
        "provenance": provenance(args),
    }
    record["record_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
    return record


def verify(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    ok = hashlib.sha256(canonical(record)).hexdigest() == record.get("record_sha256")
    print(f"record hash: {'ok' if ok else 'MISMATCH'}")
    changed = [p for p, h in record["provenance"]["files_sha256"].items()
               if not os.path.exists(os.path.join(ROOT, p)) or sha256_file(os.path.join(ROOT, p)) != h]
    print("method files: " + ("unchanged" if not changed else "changed: " + ", ".join(changed)))
    return 0 if ok and not changed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="preregistration_gaia_dr4.json")
    parser.add_argument("--nulls", type=int, default=2000)
    parser.add_argument("--binaries", type=int, default=1000)
    parser.add_argument("--author", default="Daniel Sandner")
    parser.add_argument("--verify", metavar="RECORD")
    args = parser.parse_args(argv)
    if args.verify:
        return verify(args.verify)
    record = build(args)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1, ensure_ascii=False, default=float)
        f.write("\n")
    print(f"wrote {args.out}\nrecord_sha256 {record['record_sha256']}\ngit commit {record['provenance']['git_commit']}"
          f"{' (method files modified!)' if record['provenance']['git_dirty'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
