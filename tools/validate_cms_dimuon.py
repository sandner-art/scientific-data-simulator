"""Validate the resonance pack against the CMS 2012 dimuon spectrum (CMS open data, CC0).

The data: opposite-charge muon pairs in events with exactly two muons, from
``Run2012BC_DoubleMuParked_Muons.root`` (CERN Open Data record 12341, 61.5 million events,
2.1 GiB; the cuts of the record 12342 example). ``--root`` histograms that file (needs ``uproot``);
``--data`` reads the histogram as a CSV with columns ``low, high, counts``. ``make_histogram``
writes that CSV: 0.25-150 GeV in 0.05 GeV bins.

Three checks, each comparing what the pack's model and baseline say with what the real spectrum
says:

1. **Known resonances where the PDG puts them.** J/psi, psi(2S), the three Upsilons and the Z, each
   fitted with a Gaussian (Z: Voigt with the PDG width) of free mass and resolution over an
   exp(quadratic) background. Reports the offset from the PDG mass and the fitted resolution,
   against the pack's default resolution of 1 % of the mass.
2. **The trigger turn-on as a false discovery.** The pack's baseline (``bump_hunt``) on the pack's
   design (12-70 GeV, 0.5 GeV bins) at the pack's statistics: the real spectrum is thinned
   binomially to ``--events`` events per spectrum (a Poisson histogram thinned with probability q
   is a Poisson histogram with mean q mu), once without an efficiency model and once with a
   logistic efficiency - the pack's form - fitted on an independent half of the data. The pack is
   run with the same form fitted to the real spectrum (power law times exponential, logistic
   trigger) and the same number of events, both ways. The question: does the simulated artefact
   behave like the real one?
3. **A known particle as a "new" one.** The bump hunt in 8.5-11.5 GeV without known peaks must find
   Upsilon(1S) at its mass; with the three Upsilon masses given, whatever it still finds is the
   real line shape the Gaussian template misses (final-state radiation, resolution varying with
   the muons' angles).

Writes ``validation_cms_dimuon.{json,md}`` to ``--out``.

    python tools/validate_cms_dimuon.py --data dimuon_2012_os.csv --out validation   # about 2 minutes
    python tools/validate_cms_dimuon.py --root Run2012BC_DoubleMuParked_Muons.root --out validation
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np
from scipy import optimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RECORD = "CERN Open Data record 12341 (CMS 2012 DoubleMuParked, Run2012B+C, NanoAOD muons), CC0; cuts of record 12342"
EDGES = np.round(np.arange(0.25, 150.0 + 1e-9, 0.05), 2)
EXP = "scidatasim.experiments.resonances.logic.ResonanceExperiment"
PEAKS = [((2.8, 3.4), ["J/psi"]), ((3.45, 3.95), ["psi(2S)"]),
         ((8.8, 11.0), ["Upsilon(1S)", "Upsilon(2S)", "Upsilon(3S)"]), ((80.0, 102.0), ["Z"])]
PACK_WINDOW, PACK_REBIN, PACK_RESOLUTION = (12.0, 70.0), 10, 0.01
UPSILON_WINDOW = (8.5, 11.5)


# ---------------------------------------------------------------------------------------- data
def make_histogram(root_path, csv_path, step_size="200 MB"):
    """Histogram the opposite-charge dimuon mass of events with exactly two muons (record 12342's
    selection) from the NanoAOD file, and write ``low, high, counts``."""
    import uproot
    counts = np.zeros(EDGES.size - 1, dtype=np.int64)
    branches = ["nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"]
    with uproot.open(root_path) as f:
        for arr in f["Events"].iterate(branches, step_size=step_size, library="np"):
            two = arr["nMuon"] == 2
            q = np.stack(arr["Muon_charge"][two])
            keep = q[:, 0] != q[:, 1]
            pt, eta, phi, m = (np.stack(arr[b][two])[keep] for b in branches[1:5])
            px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
            e = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2)
            mass = np.sqrt(np.clip(e.sum(1) ** 2 - px.sum(1) ** 2 - py.sum(1) ** 2 - pz.sum(1) ** 2, 0, None))
            counts += np.histogram(mass, EDGES)[0]
    with open(csv_path, "w", encoding="utf-8") as out:
        out.write("low,high,counts\n")
        for lo, hi, c in zip(EDGES[:-1], EDGES[1:], counts):
            out.write(f"{lo:.2f},{hi:.2f},{c}\n")
    return csv_path


def load(csv_path):
    from scidatasim import ingest
    table = ingest.read_table(csv_path, columns=["low", "high", "counts"])
    f = table.frame
    edges = np.concatenate([f["low"].to_numpy(float), [float(f["high"].iloc[-1])]])
    return edges, f["counts"].to_numpy(np.int64), table.provenance


def window(edges, counts, lo, hi, rebin=1):
    i0, i1 = int(np.argmin(np.abs(edges - lo))), int(np.argmin(np.abs(edges - hi)))
    n = (i1 - i0) // rebin * rebin
    return edges[i0:i0 + n + 1:rebin], counts[i0:i0 + n].reshape(-1, rebin).sum(1)


# ------------------------------------------------------------------------------ 1. known masses
def _hessian(f, x, h):
    n = x.size
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            a, b = np.zeros(n), np.zeros(n)
            a[i], b[j] = h[i], h[j]
            H[i, j] = H[j, i] = (f(x + a + b) - f(x + a - b) - f(x - a + b) + f(x - a - b)) / (4 * h[i] * h[j])
    return H


def fit_peaks(edges, counts, lo, hi, names, degree=2):
    """Gaussian peaks (Voigt with the PDG width for the Z) with free mass, resolution and yield over
    exp(polynomial) background, by binned Poisson likelihood; errors from the Hessian."""
    from scidatasim.baselines.counting import poisson_nll
    from scidatasim.experiments.resonances.logic import KNOWN, voigt_bins
    e, c = window(edges, counts, lo, hi)
    c = c.astype(float)
    centres, widths = 0.5 * (e[1:] + e[:-1]), np.diff(e)
    x = 2 * (centres - lo) / (hi - lo) - 1
    side = np.ones(c.size, bool)
    for n in names:
        side &= np.abs(centres - KNOWN[n][0]) > (0.05 if n == "Z" else 0.04) * KNOWN[n][0]
    poly = np.polynomial.polynomial.polyfit(x[side], np.log(np.clip(c[side], 1, None) / widths[side]), degree)
    bkg0 = np.exp(np.polyval(poly[::-1], x)) * widths
    start = list(poly)
    for n in names:
        m = KNOWN[n][0]
        start += [m, np.log(0.01 * m), np.log(max(float((c - bkg0)[np.abs(centres - m) < 0.03 * m].sum()), 10.0))]

    def model(p):
        out = np.exp(np.polyval(p[:degree + 1][::-1], x)) * widths
        for j, n in enumerate(names):
            m, log_s, log_n = p[degree + 1 + 3 * j: degree + 4 + 3 * j]
            out = out + np.exp(log_n) * voigt_bins(e, m, KNOWN[n][1] if n == "Z" else 0.0, np.exp(log_s))
        return out

    def nll(p):
        return poisson_nll(c, model(p))

    res = optimize.minimize(nll, start, method="Powell", options={"maxiter": 200000, "xtol": 1e-9, "ftol": 1e-12})
    res = optimize.minimize(nll, res.x, method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-9})
    cov = np.linalg.inv(_hessian(nll, res.x, np.maximum(np.abs(res.x) * 1e-4, 1e-5)))
    expected = model(res.x)
    chi2_ndf = float(np.sum((c - expected) ** 2 / expected) / (c.size - res.x.size))
    rows = []
    for j, n in enumerate(names):
        i = degree + 1 + 3 * j
        mass, pdg = float(res.x[i]), KNOWN[n][0]
        sigma = float(np.exp(res.x[i + 1]))
        rows.append({"name": n, "pdg_mass": pdg, "mass": mass, "mass_error": float(np.sqrt(abs(cov[i, i]))),
                     "offset_percent": 100 * (mass - pdg) / pdg, "offset_in_resolution": (mass - pdg) / sigma,
                     "resolution": sigma, "resolution_percent": 100 * sigma / mass,
                     "events": float(np.exp(res.x[i + 2])), "window": [lo, hi], "chi2_ndf": chi2_ndf})
    return rows


# ------------------------------------------------------------------------- 2. trigger turn-on
def fit_pack_form(e, c):
    """The pack's spectrum, N * m^-power exp(-slope m) * logistic(threshold, width), fitted to counts."""
    from scidatasim.baselines.counting import poisson_nll
    from scidatasim.experiments.resonances.logic import background_bins, efficiency
    centres = 0.5 * (e[1:] + e[:-1])
    c = np.asarray(c, float)

    def model(p):
        return np.exp(p[0]) * background_bins(e, p[1], p[2]) * efficiency(centres, p[3], np.exp(p[4]))

    best = None
    for thr in (20.0, 24.0, 28.0):
        for power in (2.0, 3.0, 4.0):
            r = optimize.minimize(lambda p: poisson_nll(c, model(p)), [np.log(1.5 * c.sum()), power, 0.01, thr, np.log(2.0)],
                                  method="Nelder-Mead", options={"maxiter": 20000, "maxfev": 20000, "xatol": 1e-8, "fatol": 1e-8})
            if best is None or r.fun < best.fun:
                best = r
    p = best.x
    m = model(p)
    return {"power": float(p[1]), "slope": float(p[2]), "threshold": float(p[3]), "width": float(np.exp(p[4])),
            "kept_fraction": float(m.sum() / np.exp(p[0])), "chi2_ndf": float(np.sum((c - m) ** 2 / m) / (c.size - 5))}


def fit_logistic_efficiency(e, c, degree=3):
    """The analysis' efficiency model: the baseline's background (exp of a cubic in log m) times a
    logistic, its threshold and width profiled on a grid."""
    from scidatasim.baselines.counting import BinnedFit
    from scidatasim.experiments.resonances.logic import efficiency
    centres = 0.5 * (e[1:] + e[:-1])
    best = None
    for thr in np.arange(18.0, 32.0 + 1e-9, 0.25):
        for w in np.arange(0.5, 5.0 + 1e-9, 0.25):
            nll = BinnedFit(e, degree=degree, efficiency=efficiency(centres, thr, w)).fit(c)["nll"]
            if best is None or nll < best[0]:
                best = (nll, float(thr), float(w))
    return {"threshold": best[1], "width": best[2]}


def _rate(rows):
    n = len(rows)
    k = sum(r["detected"] for r in rows)
    return {"n": n, "detected": k, "rate": k / n if n else None,
            "masses": sorted({round(r["mass"], 2) for r in rows if r["detected"]}),
            "median_z_local": float(np.median([r["z_local"] for r in rows])) if n else None}


def trigger_check(edges, counts, events, n_spectra, seed):
    from scidatasim.engine import simulate
    from scidatasim.experiments.resonances.logic import background_bins, bump_hunt, efficiency
    e, c = window(edges, counts, *PACK_WINDOW, rebin=PACK_REBIN)
    centres = 0.5 * (e[1:] + e[:-1])
    rng = np.random.default_rng(seed)
    calib = rng.binomial(c, 0.5)
    test = c - calib
    form = fit_pack_form(e, c)
    eff_fit = fit_logistic_efficiency(e, calib)
    eff = efficiency(centres, eff_fit["threshold"], eff_fit["width"])
    q = min(1.0, events / test.sum())
    real = {"not modelled": [], "logistic, fitted": []}
    for _ in range(n_spectra):
        thinned = rng.binomial(test, q)
        for key, ef in (("not modelled", None), ("logistic, fitted", eff)):
            r = bump_hunt(e, thinned, PACK_RESOLUTION, efficiency=ef)
            real[key].append({k: r[k] for k in ("detected", "mass", "z_local", "z_global")})
    kept = float(np.sum(background_bins(e, form["power"], form["slope"]) * efficiency(centres, form["threshold"], form["width"])))
    sim = {"not modelled": [], "logistic, true": []}
    for i in range(n_spectra):
        run = simulate(dict(experiment_type=EXP, seed=seed + 1 + i, signal=False, window=list(PACK_WINDOW),
                            n_bins=e.size - 1, n_background=events / kept, resolution=PACK_RESOLUTION,
                            background={"power": form["power"], "slope": form["slope"]},
                            trigger={"threshold": form["threshold"], "width": form["width"]},
                            observation={"targets": ["counts"], "operators": [{"type": "poisson"}]}))
        sc, ef = run.results["counts_observed"]["data"], run.results["trigger_efficiency"]["data"]
        for key, eff_ in (("not modelled", None), ("logistic, true", ef)):
            r = bump_hunt(e, sc, PACK_RESOLUTION, efficiency=eff_)
            sim[key].append({k: r[k] for k in ("detected", "mass", "z_local", "z_global")})
    full = {key: {k: bump_hunt(e, c, PACK_RESOLUTION, efficiency=ef)[k] for k in ("detected", "mass", "z_local")}
            for key, ef in (("not modelled", None), ("logistic, fitted", eff))}
    return {"window": list(PACK_WINDOW), "bin_width": float(e[1] - e[0]), "events_in_window": int(c.sum()),
            "events_per_spectrum": int(round(q * test.sum())), "spectra": n_spectra,
            "pack_form_fit": form, "efficiency_fit_on_half": eff_fit,
            "real": {k: _rate(v) for k, v in real.items()}, "simulated": {k: _rate(v) for k, v in sim.items()},
            "full_statistics": full}


# ------------------------------------------------------------------------- 3. a known particle
def upsilon_check(edges, counts, events, n_spectra, seed):
    from scidatasim.experiments.resonances.logic import KNOWN, bump_hunt
    e, c = window(edges, counts, *UPSILON_WINDOW)
    ups = [KNOWN[k][0] for k in ("Upsilon(1S)", "Upsilon(2S)", "Upsilon(3S)")]
    out = {"window": list(UPSILON_WINDOW), "bin_width": float(e[1] - e[0]), "events": int(c.sum())}
    for label, known in (("no known peaks", ()), ("Upsilon(1S-3S) given", ups)):
        r = bump_hunt(e, c, PACK_RESOLUTION, known=known)
        out[label] = {"full_statistics": {k: r[k] for k in ("detected", "mass", "z_local", "z_global")}}
    rng = np.random.default_rng(seed)
    q = min(1.0, events / c.sum())
    rows = []
    for _ in range(n_spectra):
        r = bump_hunt(e, rng.binomial(c, q), PACK_RESOLUTION, known=ups)
        rows.append({k: r[k] for k in ("detected", "mass", "z_local", "z_global")})
    out["Upsilon(1S-3S) given"]["thinned"] = {"events_per_spectrum": int(round(q * c.sum())), **_rate(rows)}
    return out


# -------------------------------------------------------------------------------------- report
def markdown(rep):
    lines = ["# Validation: the resonance pack on the CMS 2012 dimuon spectrum", "",
             f"Data: {rep['data']['record']}", f"File: `{os.path.basename(rep['data']['provenance']['path'])}`, "
             f"SHA-256 `{rep['data']['provenance']['sha256']}`, {rep['data']['events']:,} pairs in 0.25-150 GeV.", "",
             "## 1. Known resonances", "",
             "| resonance | PDG mass (GeV) | fitted mass (GeV) | offset | offset / resolution | resolution | events |",
             "|---|---|---|---|---|---|---|"]
    for r in rep["masses"]:
        lines.append(f"| {r['name']} | {r['pdg_mass']:.4f} | {r['mass']:.4f} ± {r['mass_error']:.1g} | "
                     f"{r['offset_percent']:+.2f} % | {r['offset_in_resolution']:+.2f} | {r['resolution_percent']:.2f} % | "
                     f"{r['events']:,.0f} |")
    lines += ["", "The pack's default resolution is 1.0 % of the mass.", "", "## 2. The trigger turn-on", ""]
    t = rep["trigger"]
    f = t["pack_form_fit"]
    lines += [f"Window {t['window'][0]:g}-{t['window'][1]:g} GeV in {t['bin_width']:g} GeV bins, {t['events_in_window']:,} "
              f"events; {t['spectra']} thinned spectra of {t['events_per_spectrum']:,} events each.", "",
              f"The pack's form fitted to the full spectrum: power {f['power']:.2f}, slope {f['slope']:+.4f} /GeV, "
              f"logistic threshold {f['threshold']:.2f} GeV, width {f['width']:.2f} GeV; chi2/ndf {f['chi2_ndf']:.0f} "
              "at full statistics. The simulations use these values.", "",
              f"The analysis' efficiency (logistic times the baseline's background, fitted on an independent half of the "
              f"data): threshold {t['efficiency_fit_on_half']['threshold']:g} GeV, width "
              f"{t['efficiency_fit_on_half']['width']:g} GeV.", "",
              "| spectrum | efficiency | 'new resonance' found | masses (GeV) | median local Z |", "|---|---|---|---|---|"]
    for src in ("real", "simulated"):
        for k, v in t[src].items():
            m = ", ".join(f"{x:g}" for x in v["masses"]) or "-"
            lines.append(f"| {src} | {k} | {v['detected']} of {v['n']} | {m} | {v['median_z_local']:.1f} |")
    lines += ["", "At full statistics (real): " + "; ".join(
        f"{k}: {'found' if v['detected'] else 'nothing'} at {v['mass']:g} GeV, local Z {v['z_local']:.0f}"
        for k, v in t["full_statistics"].items()) + ".", "", "## 3. A known particle as a 'new' one", ""]
    u = rep["upsilon"]
    a, b = u["no known peaks"]["full_statistics"], u["Upsilon(1S-3S) given"]
    lines += [f"Window {u['window'][0]:g}-{u['window'][1]:g} GeV, {u['events']:,} events.", "",
              f"- No known peaks given: found at {a['mass']:g} GeV (Upsilon(1S) is at 9.4603), local Z {a['z_local']:.0f}.",
              f"- The three Upsilon masses given, full statistics: "
              f"{'found' if b['full_statistics']['detected'] else 'nothing'} at {b['full_statistics']['mass']:g} GeV, "
              f"local Z {b['full_statistics']['z_local']:.0f}.",
              f"- The same, thinned to {b['thinned']['events_per_spectrum']:,} events: found in {b['thinned']['detected']} "
              f"of {b['thinned']['n']} spectra"
              + (f" (at {', '.join(f'{x:g}' for x in b['thinned']['masses'])} GeV)." if b["thinned"]["masses"] else "."),
              "", "## Verdict", ""]
    lines += [f"- {k}: {v}" for k, v in rep["verdict"].items()]
    return "\n".join(lines) + "\n"


def verdict(rep):
    masses = rep["masses"]
    t, u = rep["trigger"], rep["upsilon"]
    quark = [r for r in masses if r["name"] != "Z"]
    res = [r["resolution_percent"] for r in quark]
    return {
        "known peaks at the PDG masses": (
            f"yes, within {max(abs(r['offset_percent']) for r in masses):.2f} % "
            f"({max(abs(r['offset_in_resolution']) for r in masses):.2f} of the resolution); all low, as an "
            "uncalibrated momentum scale and final-state radiation make them"),
        "resolution": f"{min(res):.2f}-{max(res):.2f} % for J/psi to Upsilon (pack default 1 %)",
        "artefact found when the trigger is not modelled": (
            f"real {t['real']['not modelled']['rate']:.2f}, simulated {t['simulated']['not modelled']['rate']:.2f}"),
        "artefact found when the trigger is modelled": (
            f"real {t['real']['logistic, fitted']['rate']:.2f} (logistic fitted), simulated "
            f"{t['simulated']['logistic, true']['rate']:.2f} (logistic, the true efficiency): the simulation is optimistic "
            "where the efficiency's shape is known exactly" if t['real']['logistic, fitted']['rate'] >
            t['simulated']['logistic, true']['rate'] + 0.2 else
            f"real {t['real']['logistic, fitted']['rate']:.2f}, simulated {t['simulated']['logistic, true']['rate']:.2f}"),
        "Upsilon(1S) found as new": f"at {u['no known peaks']['full_statistics']['mass']:g} GeV",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="histogram CSV (low, high, counts)")
    src.add_argument("--root", help="Run2012BC_DoubleMuParked_Muons.root; the CSV is written next to --out")
    parser.add_argument("--events", type=float, default=2e5, help="events per thinned spectrum (the pack's default)")
    parser.add_argument("--spectra", type=int, default=30)
    parser.add_argument("--seed", type=int, default=12341)
    parser.add_argument("--out", default=".")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    csv = args.data or make_histogram(args.root, os.path.join(args.out, "dimuon_2012_os.csv"))
    edges, counts, prov = load(csv)
    from scidatasim.provenance import numerical_environment
    rep = {"name": "resonances on CMS 2012 dimuons", "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "data": {"record": RECORD, "provenance": prov, "events": int(counts.sum())},
           "settings": {"events": args.events, "spectra": args.spectra, "seed": args.seed},
           "masses": [row for (lo, hi), names in PEAKS for row in fit_peaks(edges, counts, lo, hi, names)],
           "trigger": trigger_check(edges, counts, args.events, args.spectra, args.seed),
           "upsilon": upsilon_check(edges, counts, args.events, args.spectra, args.seed + 1000),
           "numerics": numerical_environment()}
    rep["verdict"] = verdict(rep)
    base = os.path.join(args.out, "validation_cms_dimuon")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=float)
    text = markdown(rep)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
