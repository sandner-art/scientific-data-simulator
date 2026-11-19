"""The analysis pre-registered for Gaia DR4 (10.5281/zenodo.24370371): hypotheses H1-H5 and the
exploratory analyses E1-E4, on the files :class:`.dr4.Fetch` wrote.

* H1 - sampling-exact null: each control's real epochs with Gaussian noise at its reported errors,
  ``replicates`` times; share with FAP < 0.01 (bound 0.02) and < 0.001 (bound 0.004).
* H2 - noise model: median unit-weight error of the controls' five-parameter fits per G bin in
  [0.9, 1.2]. If a bin fails, H1 and H3-H5 are also reported with that bin's errors scaled by its
  median (the registered fallback).
* H3 - injection-recovery: one orbit per control, injected into the measured w; detection and
  recovery rates per non-centrality bin (bins with >= 30 injections) within
  max(0.05, 2 binomial standard errors) of the predicted rates.
* H4 - AGIS weights detect fewer injected orbits than measurement weights for non-centrality
  20-80: one-sided exact McNemar test, p < 0.05.
* H5 - DR3 companions with non-centrality >= 50 on their DR4 epochs: share whose final period is
  within 5 % of the DR3 period at least the predicted rate minus 0.10.
* E1-E4 - DPAC's periodogram, detections and companion fraction among controls, DR4's own orbits,
  parallax shifts.

Every hypothesis is reported, passed or failed, with its numbers; one that has too few sources to
test (no non-centrality bin with 30 injections, no companion above the cut) is reported as
``None``, "not tested", which is not a pass.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from . import dr4
from .epochs import read_epoch_astrometry, source_epochs
from .fit import detect, excess_noise, fit_single_star, orbit_noncentrality
from .logic import PERIOD_TOLERANCE, AstrometricBinaryExperiment
from .model import Epochs, inject_orbit

H1_BOUNDS = {0.01: 0.02, 0.001: 0.004}
H2_BAND = (0.9, 1.2)
H3_MIN_N = 30
H4_RANGE = (20.0, 80.0)
H5_MIN_LAMBDA = 50.0
INJECTION_SEED0, NULL_SEED0 = 300_000, 400_000


# ---------------------------------------------------------------------------- loading
def load(data_dir: str, weights: str = "measurement") -> Tuple[pd.DataFrame, Dict[int, tuple], Dict[str, pd.DataFrame]]:
    """``sample.csv``, the epoch files (all ZIPs in the folder) and the auxiliary DR4 tables.
    Returns (sample, {source_id: (Epochs, w, info)}, {name: table})."""
    sample = pd.read_csv(os.path.join(data_dir, "sample.csv"))
    zips = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.startswith("epoch_astrometry") and f.endswith(".zip"))
    frame = read_epoch_astrometry(zips)
    sources = {}
    for sid, rows in dr4.iter_sources(frame):
        try:
            sources[sid] = source_epochs(rows, sid, weights=weights)
        except ValueError:
            continue
    aux = {}
    for name in ("gaia_source", "nss_two_body_orbit", "spurious_signal"):
        path = os.path.join(data_dir, f"dr4_{name}.csv")
        if os.path.exists(path):
            aux[name] = pd.read_csv(path)
    return sample, sources, aux


def _scaled(ep: Epochs, factor: float) -> Epochs:
    return ep if factor == 1.0 else Epochs(ep.t, ep.theta, ep.f_al, ep.sigma * factor)


# ---------------------------------------------------------------------------- workers (picklable)
def _null_worker(args):
    index, ep, replicates = args
    rng = np.random.default_rng(NULL_SEED0 + index)
    return [detect(ep, rng.normal(0.0, ep.sigma), keplerian=False)["fap"] for _ in range(replicates)]


def _injection_worker(args):
    index, sid, ep, w = args
    model = AstrometricBinaryExperiment({"seed": INJECTION_SEED0 + index, "companion": True})
    model.initialize({"seed": INJECTION_SEED0 + index})
    orbit = model.orbit
    lam = orbit_noncentrality(ep, orbit)
    w_inj = inject_orbit(ep, w, orbit)
    found = detect(ep, w_inj)
    eps = excess_noise(ep, w_inj)
    agis = detect(Epochs(ep.t, ep.theta, ep.f_al, np.sqrt(ep.sigma ** 2 + eps ** 2)), w_inj, keplerian=False)
    close = abs(found["period_best"] - orbit.period) <= PERIOD_TOLERANCE * orbit.period
    return {"source_id": sid, "noncentrality": lam, "period": orbit.period, "a0": model.elements["a0"],
            "detected": found["detected"], "recovered": bool(found["detected"] and close),
            "detected_agis": agis["detected"]}


def _detect_worker(args):
    sid, ep, w = args
    found = detect(ep, w)
    out = {"source_id": sid, "fap": found["fap"], "detected": found["detected"], "period_peak": found["period"],
           "period_best": found["period_best"], "parallax": found["parallax"],
           "parallax_five": found["single_star"]["parallax"], "uwe": found["single_star"]["uwe"]}
    if "keplerian" in found:
        out.update(a0=found["keplerian"]["a0"], significance=found["keplerian"]["significance"],
                   eccentricity=found["keplerian"]["orbit"].eccentricity)
    return out


def _map(fn: Callable, items: List, jobs: int, log=None, label="") -> List:
    if jobs <= 1:
        out = []
        for k, item in enumerate(items):
            out.append(fn(item))
            if log and (k + 1) % 200 == 0:
                log(f"{label}: {k + 1}/{len(items)}")
        return out
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(fn, items, chunksize=max(1, len(items) // (jobs * 8) or 1)))


# ---------------------------------------------------------------------------- hypotheses
def h2_noise(sample: pd.DataFrame, sources: Dict[int, tuple]) -> Dict[str, Any]:
    controls = sample[sample["sample"] == "control"]
    bins = {}
    for tag, rows in controls.groupby("g_bin", sort=False):
        uwe = [fit_single_star(*sources[int(s)][:2])["uwe"] for s in rows["source_id"] if int(s) in sources]
        med = float(np.median(uwe)) if uwe else float("nan")
        bins[str(tag)] = {"n": len(uwe), "median_uwe": med, "passed": bool(H2_BAND[0] <= med <= H2_BAND[1])}
    return {"band": list(H2_BAND), "bins": bins, "passed": bool(bins) and all(b["passed"] for b in bins.values())}


def h1_null(controls: List[Tuple[int, Epochs]], replicates: int, jobs: int, log=None) -> Dict[str, Any]:
    faps = np.concatenate([np.asarray(f) for f in _map(
        _null_worker, [(i, ep, replicates) for i, (_, ep) in enumerate(controls)], jobs, log, "H1")]) \
        if controls else np.array([])
    rates = {str(a): float(np.mean(faps < a)) if faps.size else None for a in H1_BOUNDS}
    passed = bool(faps.size) and all(np.mean(faps < a) <= bound for a, bound in H1_BOUNDS.items())
    return {"replicates": replicates, "n": int(faps.size), "rates": rates,
            "bounds": {str(a): b for a, b in H1_BOUNDS.items()}, "passed": passed}


def _bin_edges(label: str) -> Tuple[float, float]:
    lo, hi = label.split("-")
    return float(lo), float("inf") if hi == "inf" else float(hi)


def h3_injection(rows: List[Dict[str, Any]], record) -> Dict[str, Any]:
    predicted = dr4.hypothesis(record, "H3")["predicted"]
    bins, tested = {}, []
    for label, pred in predicted.items():
        lo, hi = _bin_edges(label)
        sel = [r for r in rows if lo <= r["noncentrality"] < hi]
        entry = {"n": len(sel)}
        for key in ("detected", "recovered"):
            p = pred.get(key)
            obs = float(np.mean([r[key] for r in sel])) if sel else None
            band = max(0.05, 2 * np.sqrt(p * (1 - p) / len(sel))) if sel and p is not None else None
            ok = None if len(sel) < H3_MIN_N or p is None else bool(abs(obs - p) <= band)
            entry[key] = {"observed": obs, "predicted": p, "band": band, "passed": ok}
            if ok is not None:
                tested.append(ok)
        bins[label] = entry
    return {"bins": bins, "min_n": H3_MIN_N, "tested": len(tested), "passed": all(tested) if tested else None}


def h4_agis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sel = [r for r in rows if H4_RANGE[0] <= r["noncentrality"] < H4_RANGE[1]]
    b = sum(r["detected"] and not r["detected_agis"] for r in sel)
    c = sum(r["detected_agis"] and not r["detected"] for r in sel)
    p = float(stats.binom.sf(b - 1, b + c, 0.5)) if b + c else 1.0
    return {"n": len(sel), "measurement_only": int(b), "agis_only": int(c),
            "detected": float(np.mean([r["detected"] for r in sel])) if sel else None,
            "detected_agis": float(np.mean([r["detected_agis"] for r in sel])) if sel else None,
            "p_value": p, "passed": bool(p < 0.05) if sel else None}


def h5_companions(companions: pd.DataFrame, sources: Dict[int, tuple], record, jobs: int,
                  scale: Optional[Dict[int, float]] = None) -> Dict[str, Any]:
    pred = float(dr4.hypothesis(record, "H5")["predicted"])
    threshold = pred - 0.10
    items, meta = [], []
    for _, row in companions.iterrows():
        sid = int(row["source_id"])
        if sid not in sources:
            continue
        ep, w, _ = sources[sid]
        ep = _scaled(ep, (scale or {}).get(sid, 1.0))
        orbit = dr4.dr3_orbit(row)
        lam = orbit_noncentrality(ep, orbit)
        if lam >= H5_MIN_LAMBDA:
            items.append((sid, ep, w))
            meta.append({"source_id": sid, "dr3_source_id": int(row["dr3_source_id"]), "dr3_period": orbit.period,
                         "noncentrality": lam})
    results = _map(_detect_worker, items, jobs)
    for m, r in zip(meta, results):
        m.update(period_best=r["period_best"], detected=r["detected"],
                 period_ok=bool(abs(r["period_best"] - m["dr3_period"]) <= PERIOD_TOLERANCE * m["dr3_period"]))
    rate = float(np.mean([m["period_ok"] for m in meta])) if meta else None
    return {"n": len(meta), "rate": rate, "predicted": pred, "threshold": threshold,
            "passed": bool(rate >= threshold) if meta else None, "rows": meta}


# ---------------------------------------------------------------------------- exploratory
def exploratory(sample, sources, aux, control_results, injections) -> Dict[str, Any]:
    res = pd.DataFrame(control_results)
    out: Dict[str, Any] = {}
    if res.empty:
        return out
    res = res.merge(sample[["source_id", "g_bin"]], on="source_id", how="left")
    if "gaia_source" in aux:
        res = res.merge(aux["gaia_source"][["source_id", "ruwe"]], on="source_id", how="left")
    # E2: detections among controls and a companion-fraction estimate per G bin
    null_rate = 1e-3
    completeness = float(np.mean([r["detected"] for r in injections])) if injections else float("nan")
    e2 = {}
    for tag, rows in res.groupby("g_bin"):
        rate = float(rows["detected"].mean())
        e2[str(tag)] = {"n": int(len(rows)), "detected": rate,
                        "companion_fraction": (rate - null_rate) / completeness if completeness else None}
    if "ruwe" in res:
        strata = pd.cut(res["ruwe"], [0, 1.2, 1.4, 2.0, np.inf])
        e2["by_ruwe"] = {str(k): float(v) for k, v in res.groupby(strata, observed=True)["detected"].mean().items()}
    out["E2"] = {"completeness_all_injections": completeness, "bins": e2}
    # E3: DR4's own orbits for sources we detect
    if "nss_two_body_orbit" in aux and not aux["nss_two_body_orbit"].empty:
        nss = aux["nss_two_body_orbit"].rename(columns=str.lower)
        both = res[res["detected"]].merge(nss[["source_id", "period"]], on="source_id", how="inner")
        ratio = (both["period_best"] / both["period"]).to_numpy()
        out["E3"] = {"both": int(len(both)), "period_within_5pct": float(np.mean(np.abs(ratio - 1) <= 0.05))
                     if len(both) else None,
                     "dr4_orbits_we_miss": int(len(set(nss["source_id"]) & set(res.loc[~res["detected"], "source_id"])))}
    # E1: DPAC's circular-orbit periodogram against ours
    if "spurious_signal" in aux and not aux["spurious_signal"].empty:
        sp = aux["spurious_signal"].rename(columns=str.lower)
        cols = [c for c in ("astro_circ_freq", "astro_circ_freq_fap") if c in sp]
        if cols:
            m = res.merge(sp[["source_id"] + cols], on="source_id", how="inner")
            e1 = {"n": int(len(m))}
            if "astro_circ_freq_fap" in m and len(m) > 2 and np.ptp(m["astro_circ_freq_fap"]) > 0:
                ours = np.log10(np.clip(m["fap"].to_numpy(float), 1e-300, 1))
                e1["rank_correlation_log_fap"] = float(stats.spearmanr(ours, m["astro_circ_freq_fap"]).statistic)
            if "astro_circ_freq" in m and len(m):
                det = m[m["detected"]]
                if len(det):
                    e1["peak_frequency_agreement_5pct"] = float(np.mean(
                        np.abs(det["astro_circ_freq"] * det["period_peak"] - 1) <= 0.05))
            out["E1"] = e1
    # E4: parallax shift for detected controls
    det = res[res["detected"]]
    if len(det):
        out["E4"] = {"n": int(len(det)), "median_abs_shift_mas": float(np.median(np.abs(det["parallax"]
                                                                                      - det["parallax_five"])))}
    return out


# ---------------------------------------------------------------------------- driver
def analyze(record: Dict[str, Any], data_dir: str, jobs: int = 1, replicates: int = 20, log=print) -> Dict[str, Any]:
    sample, sources, aux = load(data_dir)
    sample = sample.drop_duplicates(subset=["source_id", "sample"])
    controls = [(int(s), sources[int(s)]) for s in sample.loc[sample["sample"] == "control", "source_id"]
                if int(s) in sources]
    companions = sample[sample["sample"].isin(["dr3_companion", "named"])]
    companions = companions[companions.get("period").notna()] if "period" in companions else companions.iloc[:0]
    report: Dict[str, Any] = {
        "record_sha256": record.get("record_sha256"), "registration_doi": "10.5281/zenodo.24370371",
        "choices_not_in_record": dr4.CHOICES,
        "counts": {"controls_selected": int((sample["sample"] == "control").sum()), "controls_with_epochs": len(controls),
                   "companions_selected": int(len(companions)),
                   "sources_without_usable_epochs": int(len(set(sample["source_id"].astype(int)) - set(sources)))},
        "outliers": {"gross_ccd_dropped": int(sum(v[2]["n_gross"] for v in sources.values())),
                     "ccd_used": int(sum(v[2]["n_ccd"] for v in sources.values()))},
    }
    log("H2: noise model")
    report["H2"] = h2_noise(sample, sources)

    def run(scale: Dict[int, float]) -> Dict[str, Any]:
        ctrl = [(sid, _scaled(v[0], scale.get(sid, 1.0)), v[1]) for sid, v in controls]
        log("H1: sampling-exact null")
        part = {"H1": h1_null([(sid, ep) for sid, ep, _ in ctrl], replicates, jobs, log)}
        log("H3/H4: injections")
        inj = _map(_injection_worker, [(i, sid, ep, w) for i, (sid, ep, w) in enumerate(ctrl)], jobs, log, "H3")
        part["H3"] = h3_injection(inj, record)
        part["H4"] = h4_agis(inj)
        log("H5: DR3 companions")
        part["H5"] = h5_companions(companions, sources, record, jobs, scale)
        log("controls: detection on the real data")
        found = _map(_detect_worker, [(sid, ep, w) for sid, ep, w in ctrl], jobs, log, "controls")
        part["exploratory"] = exploratory(sample, sources, aux, found, inj)
        part["_injections"], part["_controls"] = inj, found
        return part

    report.update(run({}))
    if not report["H2"]["passed"]:
        factors = {b: v["median_uwe"] for b, v in report["H2"]["bins"].items() if not v["passed"]}
        by_bin = dict(zip(sample["source_id"].astype(int), sample.get("g_bin", pd.Series(dtype=str))))
        g = aux.get("gaia_source")
        if g is not None:  # companions: their G bin from DR4 gaia_source
            for sid, mag in zip(g["source_id"].astype(int), g["phot_g_mean_mag"]):
                for lo, hi in record["selection"]["controls"]["g_bins"]:
                    if lo <= mag < hi:
                        by_bin.setdefault(sid, f"{lo:g}-{hi:g}")
        scale = {sid: factors[b] for sid, b in by_bin.items() if isinstance(b, str) and b in factors}
        log(f"H2 failed in {sorted(factors)}: running the registered fallback (errors scaled per bin)")
        report["fallback_rescaled"] = {"factors": factors, **run(scale)}
    report["summary"] = {h: report[h]["passed"] for h in ("H1", "H2", "H3", "H4", "H5")}
    return report


def markdown(report: Dict[str, Any]) -> str:
    def fmt(v):
        return "-" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v)

    lines = ["# Gaia DR4 astrometric companions: pre-registered analysis",
             f"Registration: {report['registration_doi']} (record_sha256 {report['record_sha256']})", "",
             "| hypothesis | passed |", "|---|---|"]
    lines += [f"| {h} | {'not tested (too few sources)' if ok is None else 'yes' if ok else 'NO'} |"
              for h, ok in report["summary"].items()]
    lines += ["", f"Counts: {report['counts']}", f"Outliers: {report['outliers']}", "",
              "## H1 sampling-exact null", "", f"rates {report['H1']['rates']} (bounds {report['H1']['bounds']}), "
              f"{report['H1']['n']} replicates in all", "", "## H2 noise model", ""]
    lines += [f"- G {b}: median UWE {fmt(v['median_uwe'])} (n = {v['n']}) {'ok' if v['passed'] else 'OUTSIDE'}"
              for b, v in report["H2"]["bins"].items()]
    lines += ["", "## H3 injection-recovery", "", "| non-centrality | n | detected (pred.) | recovered (pred.) |",
              "|---|---|---|---|"]
    for label, b in report["H3"]["bins"].items():
        lines.append(f"| {label} | {b['n']} | {fmt(b['detected']['observed'])} ({fmt(b['detected']['predicted'])}) "
                     f"{'' if b['detected']['passed'] is None else 'ok' if b['detected']['passed'] else 'FAIL'} | "
                     f"{fmt(b['recovered']['observed'])} ({fmt(b['recovered']['predicted'])}) "
                     f"{'' if b['recovered']['passed'] is None else 'ok' if b['recovered']['passed'] else 'FAIL'} |")
    h4, h5 = report["H4"], report["H5"]
    lines += ["", "## H4 AGIS weights", "", f"n = {h4['n']}, detected {fmt(h4['detected'])} vs AGIS "
              f"{fmt(h4['detected_agis'])}; discordant {h4['measurement_only']} / {h4['agis_only']}, p = {fmt(h4['p_value'])}",
              "", "## H5 DR3 companions", "", f"n = {h5['n']}, period within 5 %: {fmt(h5['rate'])} "
              f"(threshold {fmt(h5['threshold'])})", "", "## Exploratory", ""]
    lines += [f"- {k}: {v}" for k, v in report.get("exploratory", {}).items()]
    if "fallback_rescaled" in report:
        fb = report["fallback_rescaled"]
        lines += ["", "## Registered fallback: errors rescaled per G bin", "", f"factors {fb['factors']}"]
        lines += [f"- {h}: {'not tested' if fb[h]['passed'] is None else 'passed' if fb[h]['passed'] else 'failed'}"
                  for h in ("H1", "H3", "H4", "H5")]
    lines += ["", "## Choices not fixed by the record", ""] + [f"- {k}: {v}" for k, v in
                                                                report["choices_not_in_record"].items()]
    return "\n".join(lines) + "\n"
