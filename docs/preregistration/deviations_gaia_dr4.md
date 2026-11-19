# Gaia DR4 pre-registration: deviations log

Record: 10.5281/zenodo.24370371 (`record_sha256` 945306f9…), registered commit b30a7e9 (v0.7.0).
Every change after registration that touches the registered files, the analysis or the selection
goes here, with its date, reason and effect, before the results are read. "No effect" means the
registered method, selection and predictions are unchanged.

`sds check-preregistration docs/preregistration/preregistration_gaia_dr4.json` at a later commit lists
the registered files that changed; each must have an entry below.

## Changes to registered files

| # | Date | File | Change | Reason | Effect |
|---|---|---|---|---|---|
| D1 | 2026-11-19 | `tools/preregister_gaia_dr4.py` | File hashes read CRLF as LF; the docstring names the deposit's DOI (commit "Pre-registration: hash files as git stores them") | 0.7.0 hashed Windows checkouts byte for byte, so a record made on Windows failed verification elsewhere | No effect. The deposited record was made on Linux; its hashes are the LF ones, which the new function reproduces on every system |
| D2 | 2026-11-19 | `scidatasim/experiments/astrometry/__init__.py` | Two docstring lines naming the new modules `dr4` and `dr4_analysis` | Documentation | No effect: no code changed |

The method files `model.py`, `fit.py`, `epochs.py` and `logic.py` and `tools/check_astrometry_detection.py` are unchanged since registration.

## Code written after the record

The fetch and analysis were written after the record (commit "Gaia DR4: fetch and analysis as pre-registered", 19 November 2026): `scidatasim/experiments/astrometry/dr4.py`, `dr4_analysis.py`, `tools/gaia_dr4_fetch.py`, `tools/gaia_dr4_analyze.py`. They call the registered functions (`fit.detect`, `fit.orbit_noncentrality`, `fit.excess_noise`, `epochs.source_epochs`) unchanged. They fix choices the record left open (`dr4.CHOICES`, also printed in every report):

| Choice | Fixed as |
|---|---|
| Seeds of the H1 null replicates | `numpy.random.default_rng(400000 + control index)`, replicates drawn in order |
| Injected orbits (H3, H4) | `AstrometricBinaryExperiment({"seed": 300000 + control index, "companion": True})` with its defaults: period 20–2000 d log-uniform, a0 0.06–2 mas log-uniform, e < 0.7, isotropic orientation |
| DR3 orbits on the DR4 time axis (H5) | DR3 Thiele–Innes A, B, F, G, period and eccentricity as published; `t_periastron` moved from J2016.0 to J2017.5 (−547.875 d) |
| DR3–DR4 match | `gaiadr4.dr3_neighbourhood`: the DR4 source nearest each DR3 source with \|ΔG\| < 0.5 |
| H5 final period | `fit.detect`'s `period_best` (the Keplerian period when detected, else the periodogram peak) |
| H3 test bins | Only non-centrality bins with at least 30 injections are tested; a hypothesis with no testable bin is reported "not tested" |
| H4 test | Exact one-sided McNemar (binomial) on the discordant pairs, p < 0.05 |
| H2 fallback groups | Companions take the scale factor of their G bin, from DR4 `gaia_source.phot_g_mean_mag` |

The analysis code is frozen with the tag `gaia-dr4-analysis` before 2 December 2026, 12:00 CET; any later change gets an entry here.

## To check on release day (2 December 2026)

| Item | Registered assumption | Outcome |
|---|---|---|
| DataLink `data_structure` and format | `RAW`, `ecsv`; read by `epochs.read_epoch_astrometry` | (fill in after `gaia_dr4_fetch.py --probe`) |
| Table and column names | Draft data model, issue D | (fill in: any renaming) |
| Percentile thresholds per G bin | Resolved by the registered ADQL | (fill in: values from `manifest.json`) |
