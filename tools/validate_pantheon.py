"""Validate the cosmology pack against Pantheon+ (Brout et al. 2022, ApJ 938, 110).

Downloads the public Pantheon+ distances and their STAT+SYS covariance from the collaboration's
GitHub release (or reads them from ``--data``), and runs the same estimator - the SN-only
likelihood with the absolute magnitude marginalised - on

* the real data (z_HD > 0.01, full covariance), compared with the published flat-LCDM
  Om = 0.334 +/- 0.018 and flat-wCDM w0 = -0.90 +/- 0.14, and
* simulations of the same design: the real redshifts, magnitudes drawn from the published
  covariance around the published cosmology, fitted the same way (bias and 1-sigma coverage).

Writes ``validation_pantheon_om.{json,md}`` and ``validation_pantheon_w0.{json,md}`` to ``--out``,
with the files' URLs and SHA-256. Needs network access to GitHub unless ``--data`` is given.

    python tools/validate_pantheon.py --out validation               # about 5 minutes
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/"
FILES = {"distances": "Pantheon%2BSH0ES.dat", "covariance": "Pantheon%2BSH0ES_STAT%2BSYS.cov"}
REFERENCE = "Brout et al. 2022, ApJ 938, 110 (Pantheon+ cosmological constraints, SNe Ia alone)"
PUBLISHED = {"om": (0.334, 0.018), "w0": (-0.90, 0.14)}


def fetch(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for key, name in FILES.items():
        path = os.path.join(data_dir, name.replace("%2B", "+"))
        if not os.path.exists(path):
            urllib.request.urlretrieve(BASE + name, path)
        paths[key] = path
    return paths


def load(paths, min_z=0.01):
    import pandas as pd
    from scidatasim import ingest
    d = pd.read_csv(paths["distances"], sep=r"\s+")
    with open(paths["covariance"]) as f:
        n = int(f.readline())
        cov = np.loadtxt(f).reshape(n, n)
    sel = (d["zHD"] > min_z).to_numpy()
    prov = {k: ingest.provenance(p, "text") for k, p in paths.items()}
    for k in prov:
        prov[k]["url"] = BASE + FILES[k]
    return d[sel].reset_index(drop=True), cov[np.ix_(sel, sel)], prov


def main(argv=None):
    from scidatasim import validation
    from scidatasim.experiments.cosmology.model import distance_modulus, fit
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=None, help="folder with Pantheon+SH0ES.dat and the .cov (else download)")
    parser.add_argument("--out", default="validation")
    parser.add_argument("--n-sim", type=int, default=150)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    paths = fetch(args.data or os.path.join(args.out, "pantheon_plus"))
    d, cov, prov = load(paths)
    z, zhel, m = d["zHD"].to_numpy(), d["zHEL"].to_numpy(), d["m_b_corr"].to_numpy()
    chol = np.linalg.cholesky(cov)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    reports = {}
    for param, model, n_sim in (("om", "lcdm", args.n_sim), ("w0", "wcdm", max(args.n_sim // 2, 2))):
        real = fit(z, m, cov, model, z_hel=zhel)
        truth_om = PUBLISHED["om"][0] if model == "lcdm" else real["params"]["om"]
        truth_w0 = -1.0 if model == "lcdm" else PUBLISHED["w0"][0]
        mu = distance_modulus(z, truth_om, truth_w0, z_hel=zhel) - 19.25
        est, err = [], []
        for _ in range(n_sim):
            sim = fit(z, mu + chol @ rng.normal(size=z.size), cov, model, z_hel=zhel)
            est.append(sim["params"][param])
            err.append(sim["errors"][param])
        value, error = PUBLISHED[param]
        rep = validation.validation_report(
            f"Pantheon+ flat {'LCDM' if model == 'lcdm' else 'wCDM'}: {param}",
            "SN-only chi2, M marginalised, full STAT+SYS covariance (scidatasim.experiments.cosmology.model.fit)",
            param, {"estimate": real["params"][param], "error": real["errors"][param], "n": real["n"],
                    "provenance": prov},
            {"value": value, "error": error, "reference": REFERENCE},
            truth_w0 if param == "w0" else truth_om, est, err,
            notes=f"{real['n']} supernovae with z_HD > 0.01; simulations: the real redshifts, magnitudes drawn from the "
                  "published covariance around the truth shown, fitted the same way.")
        validation.save(rep, os.path.join(args.out, f"validation_pantheon_{param}"))
        reports[param] = rep
        print(validation.markdown(rep))
    return 0 if all(r["passed"] for r in reports.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
