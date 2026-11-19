# Example 16: the science packs scored with the standard baselines

Four packs with planted truth - transiting planets, new resonances, supernova cosmology, astrometric
binaries - each answered by its field's standard method and scored against the private answer
keys. Accuracy is split by the private labels: items the data cannot settle
(`signal_undetectable`: below the detection limit; `period_ambiguous`: a single transit, or a
wrong period that fits better than the true one) and items with a planted instrument artefact
(`instrument_artifact`: a trigger turn-on, half of them left out of the background model; a
calibration offset of the low-redshift supernovae).

```bash
python -m examples.example_16_science_packs.run_experiment            # about 2 minutes
python -m examples.example_16_science_packs.run_experiment --quick    # about 20 s
```

| pack | method |
|---|---|
| transits | running-median detrending, box least squares, SNR > 7.1 |
| resonances | binned likelihood fit, Gaussian signal scan, global significance (Gross-Vitells) > 3 |
| cosmology | flat LCDM against flat w0wa, M marginalised; evolving when Delta chi2 > 11.8 |
| astrometry | five-parameter fit, circular-orbit periodogram (FAP < 0.001), Keplerian refit |

## Results (seed 16, 30 items per pack)

| pack | question | accuracy | undetectable (n) | ambiguous (n) | artefact (n) |
|---|---|---|---|---|---|
| transits | planet present | 0.73 | 0.00 (8) | - | - |
| transits | period | 0.73 | 0.00 (8) | - | - |
| transits | depth | 0.50 | 0.00 (8) | - | - |
| resonances | new resonance | 0.77 | 0.33 (9) | - | 0.70 (10) |
| resonances | mass | 0.60 | 0.00 (9) | - | 0.30 (10) |
| cosmology | cosmological constant | 0.57 | 0.27 (15) | - | 0.60 (10) |
| cosmology | Omega_m | 0.80 | - | - | 0.70 (10) |
| cosmology | w0 | 0.73 | - | - | 0.50 (10) |
| astrometry | companion | 0.77 | 0.00 (7) | - | - |
| astrometry | period | 0.70 | 0.00 (7) | 0.00 (2) | - |
| astrometry | parallax | 0.90 | - | - | - |

Read it by column. On items the data cannot settle, the baselines are wrong almost always - as
they should be: a benchmark that scores them without the label rewards guessing. The baselines
also lose a third to a half of the items with an instrument artefact, which a method that models
the instrument does not. The transit depth fails at high signal-to-noise for a third reason: a box
fit is shallower than the largest decrement of a limb-darkened transit - the question asks for a
transit model.
