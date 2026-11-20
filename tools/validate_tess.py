"""Validate the transit pack against TESS light curves of stars with known planets.

The data: SPOC 2-minute light curves (PDCSAP flux, quality flags applied) of WASP-18, pi Men,
TOI-270, AU Mic and HD 213885, read from ``--data`` (any folder holding ``*_lc.fits``, e.g. as
downloaded by lightkurve). Known planets: NASA Exoplanet Archive composite parameters
(``pscomppars``), pi Men c from TEPCat. Each light curve is one TESS sector, the design the pack
simulates.

Three checks, each comparing the pack's model and baseline with the real light curves:

1. **Recovery of the known planets.** The pack's baseline (``transit_search``: running-median
   detrending, BLS, SNR > 7.1) run sector by sector, masking each signal it finds and searching
   again. A planet counts as found when the period is within the pack's tolerance (P/2 and 2P are
   counted apart). The measured SNR is compared with the white-noise expectation from the
   archive's depth, the number of in-transit points and the light curve's scatter - the quantity
   the pack uses to label a planet ``signal_undetectable``.
2. **Injection and recovery in real noise.** Planets from the pack (limb-darkened, circular
   orbits) are planted into the real light curves with the known planets masked, through the
   ``real_noise`` observation operator, and into a white-noise twin: the same planet, time stamps
   and gaps with Gaussian noise of the light curve's scatter. Known planets are masked at their
   transits and secondary eclipses. Each arm is searched up to ``--rounds`` times, masking what
   was found, so a planted planet under a stronger real signal can still be found. Recovery is reported by expected SNR
   for both. Where they differ, the real noise (spots, flares, systematics) is what the white-noise
   simulation misses.
3. **False alarms.** The search on each real light curve with its known planets masked, and on its
   white-noise twins.

Writes ``validation_tess.{json,md}`` to ``--out``.

    python tools/validate_tess.py --data C:/opendata/tess/lc --out validation          # about 10 minutes
    python tools/validate_tess.py --data lc --injections 30 --jobs 4
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EXP = "scidatasim.experiments.transits.logic.TransitExperiment"
BTJD = 2457000.0
CADENCE = 2.0 / 1440
SNR_THRESHOLD = 7.1
SNR_BINS = [(3, 5), (5, 7.1), (7.1, 10), (10, 15), (15, 20)]
ARCHIVE = "NASA Exoplanet Archive, Planetary Systems Composite Parameters (pscomppars), rows retrieved 2026-11-19"
#: TIC -> (host, [(planet, period d, mid-transit BJD_TDB, depth ppm, duration h, Rp/R*, source)])
HOSTS = {
    100100827: ("WASP-18", [("WASP-18 b", 0.94145223, 2456740.8056, 10410.0, 2.21, 0.1018, ARCHIVE)]),
    261136679: ("pi Men", [("pi Men c", 6.267825, 2458325.5042, 300.0, 3.10, 0.01532,
                            "TEPCat (ephemeris of arXiv:2607.12088, depth 0.03 %); duration from a/R* 13.10 "
                            "(Gandolfi et al. 2018) and cos i 0.0427 (Guenther & Daylan 2021)")]),
    259377017: ("TOI-270", [("TOI-270 b", 3.35992, 2458461.01464, 976.7, 1.294, 0.0307, ARCHIVE),
                            ("TOI-270 c", 5.66051, 2458463.08056, 3451.8, 1.682, 0.056, ARCHIVE),
                            ("TOI-270 d", 11.38194, 2458469.33823, 2645.0, 2.117, 0.0491, ARCHIVE)]),
    441420236: ("AU Mic", [("AU Mic b", 8.463446, 2458330.350878, 2490.0, 3.24, 0.0499, ARCHIVE),
                           ("AU Mic c", 18.859023, 2458342.223325, 850.0, 4.29, 0.0291, ARCHIVE)]),
    403224672: ("HD 213885", [("HD 213885 b", 1.008035, 2458379.97043, 200.0, 1.41, 0.01453, ARCHIVE)]),
}


# ---------------------------------------------------------------------------------------- data
def load_lightcurves(data_dir):
    """Every ``*_lc.fits`` under ``data_dir`` whose TIC is in ``HOSTS``, as dicts with the time
    (BTJD, gaps in the time column filled from the cadence), relative PDCSAP flux (NaN where
    flagged) and the file's provenance."""
    from astropy.io import fits
    from scidatasim import ingest
    out = []
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*_lc.fits"), recursive=True)):
        head = fits.getheader(path, 0)
        tic, sector = int(head.get("TICID", -1)), int(head.get("SECTOR", -1))
        if tic not in HOSTS:
            print(f"skipping {os.path.basename(path)}: TIC {tic} has no known planets in this table")
            continue
        s = ingest.read_series(path, value="PDCSAP_FLUX", time="TIME", error="PDCSAP_FLUX_ERR", quality="QUALITY",
                               normalize="median", key=1)
        t = s.time.copy()
        bad = ~np.isfinite(t)
        if bad.any():  # rows without a time stamp: interpolate on the row index (one row per cadence)
            idx = np.arange(t.size)
            t[bad] = np.interp(idx[bad], idx[~bad], t[~bad])
        y = np.where(np.isfinite(s.value), s.value, np.nan)
        out.append({"tic": tic, "host": HOSTS[tic][0], "sector": sector, "time": t, "flux": y,
                    "provenance": s.provenance, "flagged": s.meta.get("flagged")})
    out.sort(key=lambda lc: (lc["host"], lc["sector"]))
    return out


def scatter(t, y):
    """Robust point-to-point scatter of the detrended light curve (1.4826 MAD): the white-noise
    level the pack's ``sigma_ppm`` stands for."""
    from scidatasim.experiments.transits.model import detrend
    ok = np.isfinite(y)
    flat = detrend(t[ok], y[ok])
    return float(1.4826 * np.median(np.abs(flat - np.median(flat))))


def phase_distance(t, period, t0):
    """Time from the nearest mid-transit."""
    return np.abs((np.asarray(t, float) - t0 + 0.5 * period) % period - 0.5 * period)


def planet_in(lc, planet):
    """Transits with data, in-transit points and the box expectation of the SNR."""
    name, period, t0, depth, dur_h, _, _ = planet
    ok = np.isfinite(lc["flux"])
    t = lc["time"][ok]
    dur = dur_h / 24.0
    inside = phase_distance(t, period, t0 - BTJD) < dur / 2
    epochs = np.unique(np.round((t[inside] - (t0 - BTJD)) / period))
    return {"n_transits": int(epochs.size), "n_in": int(inside.sum()), "duration": dur}


def tolerance(period, duration, n_transits):
    """The pack's period tolerance: the duration over the number of transits, at least 0.1 %."""
    return max(duration / max(n_transits, 1), 1e-3 * period)


def near_known(tic, period, rel=0.005):
    """The known planet whose period (or half or double) this one is, within ``rel``: a detection
    left by a known planet - its secondary eclipse, its phase curve, or transits the mask missed."""
    for p in HOSTS[tic][1]:
        if any(abs(period - k * p[1]) <= rel * k * p[1] for k in (1.0, 0.5, 2.0)):
            return p[0]
    return None


def classify(found_period, period, tol):
    if abs(found_period - period) <= tol:
        return "period"
    if any(abs(found_period - k * period) <= k * tol for k in (0.5, 2.0)):
        return "alias"
    return None


# ------------------------------------------------------------------------------ 1. recovery
def iterative_search(t, y, sigma, max_rounds):
    """The baseline run again and again, each signal it finds masked (one duration either side of
    its mid-transits), until it finds nothing or ``max_rounds`` is reached: a second planet, or a
    planted one under a stronger real signal, is found the way an analyst would find it."""
    from scidatasim.experiments.transits.model import transit_search
    t, y = np.asarray(t, float), np.asarray(y, float)
    mask = np.isfinite(t) & np.isfinite(y)
    rounds = []
    for _ in range(max_rounds):
        if mask.sum() < 100:
            break
        r = transit_search(t[mask], y[mask], sigma)
        rounds.append({k: r[k] for k in ("period", "depth_ppm", "duration", "t0", "snr", "detected")})
        if not r["detected"]:
            break
        mask &= phase_distance(t, r["period"], r["t0"]) > r["duration"]
    return rounds


def recover(lc, extra_rounds=3):
    ok = np.isfinite(lc["flux"])
    t, y = lc["time"][ok], lc["flux"][ok]
    sigma = scatter(lc["time"], lc["flux"])
    span = float(t.max() - t.min())
    max_period = min(span / 2.0, 20.0)
    planets = []
    for p in HOSTS[lc["tic"]][1]:
        info = planet_in(lc, p)
        planets.append({"name": p[0], "period": p[1], "depth_ppm": p[3], "duration_h": p[4], **info,
                        "searchable": bool(info["n_transits"] >= 2 and p[1] <= max_period),
                        "expected_snr": p[3] * 1e-6 * np.sqrt(info["n_in"]) / sigma, "found": None})
    rounds = iterative_search(t, y, sigma, sum(p["searchable"] for p in planets) + extra_rounds)
    unmatched = []
    for k, r in enumerate(rounds):
        if not r["detected"]:
            break
        for p in planets:
            how = classify(r["period"], p["period"], tolerance(p["period"], p["duration"], p["n_transits"]))
            if how and p["found"] is None:
                p.update(found=how, round=k + 1, found_period=r["period"], found_t0=r["t0"],
                         found_duration=r["duration"], found_depth_ppm=r["depth_ppm"], measured_snr=r["snr"],
                         depth_ratio=r["depth_ppm"] / p["depth_ppm"],
                         snr_ratio=r["snr"] / p["expected_snr"] if p["expected_snr"] > 0 else None)
                break
        else:
            unmatched.append({**r, "near_known": near_known(lc["tic"], r["period"])})
    return {"host": lc["host"], "tic": lc["tic"], "sector": lc["sector"], "points": int(t.size),
            "span_d": span, "scatter_ppm": 1e6 * sigma, "planets": planets, "rounds": rounds,
            "unmatched_detections": unmatched, "file": os.path.basename(lc["provenance"]["path"]),
            "sha256": lc["provenance"]["sha256"]}


# ------------------------------------------------------------------- 2-3. injection and nulls
def masked_background(lc, recovered=None):
    """Relative flux with every known planet's transits and secondary eclipses (one full duration
    either side of each mid-transit and mid-eclipse) set to NaN - by the archive's ephemeris and,
    where check 1 found the planet in this light curve, by the ephemeris found there too (an
    archive ephemeris propagated over years can be off by more than a transit duration)."""
    bg = lc["flux"].copy()
    t = lc["time"]
    found = {p["name"]: p for p in (recovered or {}).get("planets", []) if p.get("found")}
    for p in HOSTS[lc["tic"]][1]:
        ephemerides = [(p[1], p[2] - BTJD, p[4] / 24.0)]
        if p[0] in found:
            f = found[p[0]]
            ephemerides.append((f["found_period"], f["found_t0"], max(f["found_duration"], p[4] / 24.0)))
        for period, t0, dur in ephemerides:
            for offset in (0.0, 0.5 * period):  # transit and (circular orbit) secondary eclipse
                bg[phase_distance(t, period, t0 + offset) < dur] = np.nan
    return bg


def _inject(task):
    from scidatasim.engine import simulate
    from scidatasim.experiments.transits.model import Planet, a_over_rstar
    seed, bg, sigma, label, rounds = task
    n = bg.size
    tp = np.arange(n) * CADENCE
    valid = np.isfinite(bg)
    rng = np.random.default_rng(seed)
    period = float(np.exp(rng.uniform(np.log(1.0), np.log(12.0))))
    t0, b, target = float(rng.uniform(0, period)), float(rng.uniform(0, 0.7)), float(rng.uniform(3.0, 20.0))
    rp = 0.02
    for _ in range(3):  # the radius that gives the target white-noise SNR on this light curve's points
        f = Planet(period, t0, rp, a_over_rstar(period), b).flux(tp[valid])
        snr = float(np.sqrt(np.sum((1 - f) ** 2)) / sigma)
        if snr <= 0:
            break
        rp = float(min(0.2, rp * np.sqrt(target / snr)))
    sim = simulate(dict(experiment_type=EXP, seed=seed, planet=True, orbit={"period": period, "t0": t0, "rp": rp, "b": b},
                        span=(n - 0.5) * CADENCE, cadence=CADENCE, gap=None, sigma_ppm=1e6 * sigma,
                        observation={"targets": ["flux"], "operators": [
                            {"type": "real_noise", "values": bg.tolist(), "align": "index", "offset": 0,
                             "combine": "multiply"}]}))
    time, real, model = (np.asarray(sim.results[k]["data"], float) for k in ("time", "flux_observed", "transit_model"))
    ok = np.isfinite(real)
    white = np.where(ok, model * (1.0 + rng.normal(0.0, sigma, n)), np.nan)
    planet = sim.logic.planet
    n_tr = planet.transits_in(time[ok])
    tol = tolerance(period, planet.duration, n_tr)
    row = {"light_curve": label, "seed": seed, "period": period, "rp": rp, "b": b, "n_transits": n_tr,
           "expected_snr": float(np.sqrt(np.sum((1 - model[ok]) ** 2)) / sigma),
           # the expectation used for the known planets: largest depth x sqrt(in-transit points) / scatter
           "box_expected_snr": float((1 - np.min(model[ok])) * np.sqrt(np.sum(model[ok] < 1)) / sigma)}
    for arm, flux in (("real", real), ("white", white)):
        found = iterative_search(time, flux, sigma, rounds)
        hits = [(k, classify(r["period"], period, tol)) for k, r in enumerate(found) if r["detected"]]
        k, how = next(((k, h) for k, h in hits if h == "period"), next(((k, h) for k, h in hits if h), (None, None)))
        r = found[k] if k is not None else found[0]
        row[arm] = {"detected": bool(hits), "found": how, "round": None if k is None else k + 1, "snr": r["snr"],
                    "period": r["period"], "rounds_detected": len(hits)}
    return row


def _null(task):
    from scidatasim.experiments.transits.model import transit_search
    seed, bg, sigma, label, arm, tic = task
    tp = np.arange(bg.size) * CADENCE
    flux = bg if arm == "real" else np.where(np.isfinite(bg), 1.0 + np.random.default_rng(seed).normal(0, sigma, bg.size),
                                              np.nan)
    r = transit_search(tp, flux, sigma)
    return {"light_curve": label, "arm": arm, "seed": seed, "detected": r["detected"], "snr": r["snr"],
            "period": r["period"], "depth_ppm": r["depth_ppm"],
            "near_known": near_known(tic, r["period"]) if r["detected"] else None}


def _binom(k, n):
    p = k / n if n else float("nan")
    return {"n": n, "k": k, "rate": p, "error": float(np.sqrt(p * (1 - p) / n)) if n else float("nan")}


def summarise_injections(rows):
    out = []
    for lo, hi in SNR_BINS:
        sel = [r for r in rows if lo <= r["expected_snr"] < hi]
        line = {"expected_snr": f"{lo:g}-{hi:g}"}
        for arm in ("real", "white"):
            line[arm] = _binom(sum(r[arm]["found"] == "period" for r in sel), len(sel))
            line[arm]["alias"] = sum(r[arm]["found"] == "alias" for r in sel)
            line[arm]["other"] = sum(bool(r[arm]["detected"]) and r[arm]["found"] is None for r in sel)
        d = line["real"]["rate"] - line["white"]["rate"]
        err = np.hypot(line["real"]["error"], line["white"]["error"])
        line["real_minus_white"] = d
        line["differs"] = bool(len(sel) and abs(d) > 2 * max(err, 1.0 / len(sel)))
        out.append(line)
    return out


def snr_ratios(rows):
    """Measured / box-expected SNR of the planted planets found at their period with expected SNR
    >= 7.1: the calibration for the known planets' ratio in check 1."""
    out = {}
    for arm in ("real", "white"):
        v = [r[arm]["snr"] / r["box_expected_snr"] for r in rows
             if r[arm]["found"] == "period" and r["expected_snr"] >= SNR_THRESHOLD and r["box_expected_snr"] > 0]
        out[arm] = float(np.median(v)) if v else None
    return out


def summarise_clean(rows, nulls):
    """Recovery above the threshold on the light curves whose own null search is clean (check 3
    finds nothing, or only a known planet's residual): the noise there is what the white-noise
    model has to match."""
    clean = sorted(k for k, v in nulls.items() if not v["real"]["detected"] or v["real"].get("near_known"))
    sel = [r for r in rows if r["light_curve"] in clean and r["expected_snr"] >= SNR_THRESHOLD]
    return {"light_curves": clean,
            **{arm: _binom(sum(r[arm]["found"] == "period" for r in sel), len(sel)) for arm in ("real", "white")}}


def summarise_by_star(rows):
    out = {}
    for label in sorted({r["light_curve"] for r in rows}):
        sel = [r for r in rows if r["light_curve"] == label and r["expected_snr"] >= SNR_THRESHOLD]
        out[label] = {arm: _binom(sum(r[arm]["found"] == "period" for r in sel), len(sel)) for arm in ("real", "white")}
    return out


# -------------------------------------------------------------------------------------- report
def markdown(rep):
    def pct(x):
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
    lines = ["# Validation: the transit pack on TESS light curves of known planets", "",
             f"{len(rep['recovery'])} SPOC 2-min light curves (PDCSAP, quality flags applied). Known planets: {ARCHIVE}; "
             "pi Men c from TEPCat.", "", "## 1. Recovery of the known planets", "",
             "| star | sector | scatter (ppm) | planet | transits | expected SNR | found | measured SNR | measured / expected | "
             "BLS depth / archive depth |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lc in rep["recovery"]:
        for p in lc["planets"]:
            found = p["found"] or ("not searchable" if not p["searchable"] else "no")
            if p["found"] == "period":
                found = f"yes (round {p['round']})"
            lines.append(f"| {lc['host']} | {lc['sector']} | {lc['scatter_ppm']:.0f} | {p['name']} | {p['n_transits']} | "
                         f"{p['expected_snr']:.1f} | {found} | {pct(p.get('measured_snr'))} | {pct(p.get('snr_ratio'))} | "
                         f"{pct(p.get('depth_ratio'))} |")
    extra = [(lc, u) for lc in rep["recovery"] for u in lc["unmatched_detections"]]
    lines += ["", "Detections that match no known planet:" if extra else "No detection outside the known planets.", ""]
    lines += [f"- {lc['host']} sector {lc['sector']}: P = {u['period']:.3f} d, depth {u['depth_ppm']:.0f} ppm, SNR {u['snr']:.1f}"
              + (f" - at the period of {u['near_known']} (its secondary eclipse or phase curve)" if u.get("near_known") else "")
              for lc, u in extra]
    s = rep["recovery_summary"]
    lines += ["", f"Measured / expected SNR over the found planets: median {pct(s['snr_ratio_median'])} "
              f"(range {pct(s['snr_ratio_range'][0])}-{pct(s['snr_ratio_range'][1])}); BLS / archive depth: median "
              f"{pct(s['depth_ratio_median'])}.", "",
              "## 2. Injection and recovery: real noise against its white-noise twin", "",
              f"{rep['settings']['injections']} planets per light curve, period 1-12 d, impact parameter 0-0.7, radius set "
              f"for an expected (white-noise) SNR uniform in 3-20. Found = the period within the pack's tolerance in one of up "
              f"to {rep['settings']['rounds']} search rounds (each signal found is masked before the next).", "",
              "| expected SNR | n | found, real noise | found, white noise | real - white | P/2 or 2P (real / white) | "
              "other period (real / white) |", "|---|---|---|---|---|---|---|"]
    for b in rep["injection"]["by_expected_snr"]:
        r, w = b["real"], b["white"]
        flag = " **differs**" if b["differs"] else ""
        lines.append(f"| {b['expected_snr']} | {r['n']} | {pct(r['rate'])} ± {pct(r['error'])} | {pct(w['rate'])} ± "
                     f"{pct(w['error'])} | {b['real_minus_white']:+.2f}{flag} | {r['alias']} / {w['alias']} | "
                     f"{r['other']} / {w['other']} |")
    sr = rep["injection"]["snr_ratio_median"]
    lines += ["", f"Measured / box-expected SNR of the planted planets found (expected SNR >= 7.1), median: real noise "
              f"{pct(sr['real'])}, white noise {pct(sr['white'])} - the white-noise value is what a box fit to a "
              "limb-darkened transit gives; compare the known planets' ratio with it.", "",
              "Above the detection threshold (expected SNR >= 7.1), by light curve:", "",
              "| light curve | n | found, real | found, white |", "|---|---|---|---|"]
    for label, v in rep["injection"]["by_light_curve"].items():
        lines.append(f"| {label} | {v['real']['n']} | {pct(v['real']['rate'])} | {pct(v['white']['rate'])} |")
    lines += ["", "## 3. False alarms (known planets masked)", "", "| light curve | real | white-noise twins |",
              "|---|---|---|"]
    for label, v in rep["nulls"].items():
        r = v["real"]
        real = f"yes: P = {r['period']:.3f} d, {r['depth_ppm']:.0f} ppm, SNR {r['snr']:.1f}" if r["detected"] else \
            f"no (best SNR {r['snr']:.1f})"
        if r.get("near_known"):
            real += f" - at the period of {r['near_known']}: a known planet's residual, not a new signal"
        lines.append(f"| {label} | {real} | {v['white_detected']} of {v['white_n']} |")
    lines += ["", "## Verdict", ""] + [f"- {k}: {v}" for k, v in rep["verdict"].items()]
    return "\n".join(lines) + "\n"


def verdict(rep):
    found = [p for lc in rep["recovery"] for p in lc["planets"] if p["searchable"]]
    above = [p for p in found if p["expected_snr"] >= SNR_THRESHOLD]
    inj = rep["injection"]["by_expected_snr"]
    clean = rep["injection"]["clean_light_curves"]
    differs = [b["expected_snr"] for b in inj if b["differs"]]
    real_fa = [k for k, v in rep["nulls"].items() if v["real"]["detected"] and not v["real"].get("near_known")]
    residual = [k for k, v in rep["nulls"].items() if v["real"]["detected"] and v["real"].get("near_known")]
    white_fa = sum(v["white_detected"] for v in rep["nulls"].values())
    white_n = sum(v["white_n"] for v in rep["nulls"].values())
    return {
        "known planets found": (f"{sum(p['found'] == 'period' for p in above)} of {len(above)} searchable planets with "
                                f"expected SNR >= {SNR_THRESHOLD} (by light curve)"),
        "white-noise SNR as a predictor": (
            f"known planets: measured / box-expected median {rep['recovery_summary']['snr_ratio_median']:.2f}; planted "
            f"planets: {rep['injection']['snr_ratio_median']['real'] or float('nan'):.2f} in real noise, "
            f"{rep['injection']['snr_ratio_median']['white'] or float('nan'):.2f} in white noise"
            if rep["recovery_summary"]["snr_ratio_median"] is not None else "no planet found"),
        "recovery in real vs white noise": ("differs beyond 2 sigma at expected SNR " + ", ".join(differs)) if differs
        else "agrees within 2 sigma in every SNR bin",
        "recovery on the light curves without a false alarm": (
            f"real {clean['real']['rate']:.2f} ± {clean['real']['error']:.2f}, white {clean['white']['rate']:.2f} ± "
            f"{clean['white']['error']:.2f} (expected SNR >= {SNR_THRESHOLD}, n = {clean['real']['n']}; "
            f"{', '.join(clean['light_curves'])})"),
        "false alarms": (f"real light curves: {len(real_fa)} of {len(rep['nulls'])}"
                         + (f" ({', '.join(real_fa)})" if real_fa else "")
                         + (f", plus {len(residual)} at a known planet's period ({', '.join(residual)})" if residual else "")
                         + f"; white-noise twins: {white_fa} of {white_n}"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="folder with TESS SPOC *_lc.fits files (searched recursively)")
    parser.add_argument("--injections", type=int, default=12, help="planted planets per light curve")
    parser.add_argument("--white-nulls", type=int, default=5, help="white-noise twins per light curve for false alarms")
    parser.add_argument("--rounds", type=int, default=3, help="search rounds per planted planet (each found signal masked)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", default=".")
    args = parser.parse_args(argv)
    lcs = load_lightcurves(args.data)
    if not lcs:
        raise SystemExit(f"no TESS light curves of the known hosts under {args.data}")
    labels = [f"{lc['host']} s{lc['sector']}" for lc in lcs]
    sigmas = [scatter(lc["time"], lc["flux"]) for lc in lcs]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rec = list(pool.map(recover, lcs))
    bgs = [masked_background(lc, r) for lc, r in zip(lcs, rec)]
    inj_tasks = [(args.seed + 1000 * j + i, bgs[j], sigmas[j], labels[j], args.rounds) for j in range(len(lcs))
                 for i in range(args.injections)]
    null_tasks = [(args.seed + 500_000 + 1000 * j, bgs[j], sigmas[j], labels[j], "real", lcs[j]["tic"])
                  for j in range(len(lcs))]
    null_tasks += [(args.seed + 600_000 + 1000 * j + i, bgs[j], sigmas[j], labels[j], "white", lcs[j]["tic"])
                   for j in range(len(lcs)) for i in range(args.white_nulls)]
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        injections = list(pool.map(_inject, inj_tasks, chunksize=2))
        nulls = list(pool.map(_null, null_tasks, chunksize=2))
    found = [p for lc in rec for p in lc["planets"] if p["found"] == "period"]
    ratios = [p["snr_ratio"] for p in found if p.get("snr_ratio")]
    from scidatasim.provenance import numerical_environment
    rep = {"name": "transits on TESS", "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "settings": {"injections": args.injections, "rounds": args.rounds, "white_nulls": args.white_nulls,
                        "seed": args.seed},
           "known_planets": {str(k): {"host": v[0], "planets": [dict(zip(("name", "period", "t0_bjd", "depth_ppm", "duration_h",
                                                                          "rp_over_rstar", "source"), p)) for p in v[1]]}
                             for k, v in HOSTS.items()},
           "files": [{"label": lab, **lc["provenance"], "flagged": lc["flagged"]} for lab, lc in zip(labels, lcs)],
           "recovery": rec,
           "recovery_summary": {"snr_ratio_median": float(np.median(ratios)) if ratios else None,
                                "snr_ratio_range": [float(min(ratios)), float(max(ratios))] if ratios else [None, None],
                                "depth_ratio_median": float(np.median([p["depth_ratio"] for p in found])) if found else None},
           "injection": {"by_expected_snr": summarise_injections(injections), "by_light_curve": summarise_by_star(injections),
                         "snr_ratio_median": snr_ratios(injections),
                         "runs": injections},
           "nulls": {lab: {"real": next(n for n in nulls if n["light_curve"] == lab and n["arm"] == "real"),
                           "white_detected": sum(n["detected"] for n in nulls if n["light_curve"] == lab and n["arm"] == "white"),
                           "white_n": sum(1 for n in nulls if n["light_curve"] == lab and n["arm"] == "white")}
                     for lab in labels},
           "numerics": numerical_environment()}
    rep["injection"]["clean_light_curves"] = summarise_clean(injections, rep["nulls"])
    rep["verdict"] = verdict(rep)
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, "validation_tess")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=float)
    text = markdown(rep)
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
