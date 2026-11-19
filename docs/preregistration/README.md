# Pre-registrations

`preregistration_gaia_dr4.json`: a copy of the record deposited on Zenodo before Gaia DR4
(public 2 December 2026, 12:00 CET): detection of astrometric companions in DR4 epoch astrometry.

- Deposit: https://doi.org/10.5281/zenodo.24370371 (the deposit, not this copy, is the evidence)
- Made with scidatasim 0.7.0, commit b30a7e9 (https://doi.org/10.5281/zenodo.24362514)
- `record_sha256` 945306f960f9db7723a547cea137398a3a6d102f72f6c0c96c0ba906229215ac

To verify it, check out v0.7.0 and run `python tools/preregister_gaia_dr4.py --verify <record>`.
On Windows, clone with `-c core.autocrlf=false`: 0.7.0's script hashed files byte for byte.

After the release: `tools/gaia_dr4_fetch.py` fetches the registered sample and
`tools/gaia_dr4_analyze.py` runs the registered analysis (H1-H5, E1-E4) on it.

`deviations_gaia_dr4.md` logs every change after registration and every choice the analysis
code fixes that the record left open. Add to it before reading any result.
