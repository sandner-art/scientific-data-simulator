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

4. **Trimming.** Checks 1 and 3 again with ``--trim`` days dropped after the start and each gap
   (``transit_search(trim=...)``), where instruments settle.
5. **The pack's nuisance options against the real failures.** The pack with ``spots`` at the
   RMS and rotation period measured on AU Mic (and a quiet twin), and with ``ramps`` of the size
   measured after pi Men sector 4's restarts: planted planets found and false alarms, beside the
   real light curves' rates.

6. **The second baseline** (``transit_search_robust``: local-quadratic detrending, red-noise SNR)
   on checks 1-3, beside the first.
7. **Instrument systematics.** The signal after TESS's momentum dumps and the red noise of each
   light curve; the pack with its ``dumps`` option and with AR(1) correlated noise at the measured
   levels, searched by both baselines.

Writes ``validation_tess.{json,md}`` to ``--out``.

    python tools/validate_tess.py --data C:/opendata/tess/lc --out validation          # about an hour on 2 cores
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
        raw = fits.getdata(path, 1)
        tq, q = np.asarray(raw["TIME"], float), np.asarray(raw["QUALITY"], np.int64)
        dt = np.sort(tq[((q & 32) > 0) & np.isfinite(tq)])  # momentum dumps (reaction wheel desaturation)
        dumps = dt[np.concatenate([[True], np.diff(dt) > 0.1])] if dt.size else dt
        out.append({"tic": tic, "host": HOSTS[tic][0], "sector": sector, "time": t, "flux": y,
                    "provenance": s.provenance, "flagged": s.meta.get("flagged"), "dumps": dumps})
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
BASELINES = ("median", "robust")  # transit_search, transit_search_robust


def iterative_search(t, y, sigma, max_rounds, trim=0.0, baseline="median"):
    """The baseline run again and again, each signal it finds masked (one duration either side of
    its mid-transits), until it finds nothing or ``max_rounds`` is reached: a second planet, or a
    planted one under a stronger real signal, is found the way an analyst would find it."""
    from scidatasim.experiments.transits.model import transit_search, transit_search_robust
    search = transit_search_robust if baseline == "robust" else transit_search
    t, y = np.asarray(t, float), np.asarray(y, float)
    mask = np.isfinite(t) & np.isfinite(y)
    rounds = []
    for _ in range(max_rounds):
        if mask.sum() < 100:
            break
        r = search(t[mask], y[mask], sigma, trim=trim)
        rounds.append({k: r[k] for k in ("period", "depth_ppm", "duration", "t0", "snr", "detected")})
        if not r["detected"]:
            break
        mask &= phase_distance(t, r["period"], r["t0"]) > r["duration"]
    return rounds


def recover(lc, extra_rounds=3, trim=0.0, baseline="median"):
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
    rounds = iterative_search(t, y, sigma, sum(p["searchable"] for p in planets) + extra_rounds, trim, baseline)
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
    for arm, flux, base in (("real", real, "median"), ("white", white, "median"),
                            ("real robust", real, "robust"), ("white robust", white, "robust")):
        found = iterative_search(time, flux, sigma, rounds, 0.0, base)
        hits = [(k, classify(r["period"], period, tol)) for k, r in enumerate(found) if r["detected"]]
        k, how = next(((k, h) for k, h in hits if h == "period"), next(((k, h) for k, h in hits if h), (None, None)))
        r = found[k] if k is not None else found[0]
        row[arm] = {"detected": bool(hits), "found": how, "round": None if k is None else k + 1, "snr": r["snr"],
                    "period": r["period"], "rounds_detected": len(hits)}
    return row


def _null(task):
    from scidatasim.experiments.transits.model import transit_search
    seed, bg, sigma, label, arm, tic = task[:6]
    trim = task[6] if len(task) > 6 else 0.0
    baseline = task[7] if len(task) > 7 else "median"
    tp = np.arange(bg.size) * CADENCE
    flux = bg if arm.startswith("real") else np.where(np.isfinite(bg), 1.0 + np.random.default_rng(seed).normal(0, sigma, bg.size),
                                              np.nan)
    from scidatasim.experiments.transits.model import transit_search_robust
    r = (transit_search_robust if baseline == "robust" else transit_search)(tp, flux, sigma, trim=trim)
    return {"light_curve": label, "arm": arm, "trim": trim, "baseline": baseline, "seed": seed, "detected": r["detected"], "snr": r["snr"],
            "period": r["period"], "depth_ppm": r["depth_ppm"],
            "near_known": near_known(tic, r["period"]) if r["detected"] else None}


def _binom(k, n):
    p = k / n if n else float("nan")
    return {"n": n, "k": k, "rate": p, "error": float(np.sqrt(p * (1 - p) / n)) if n else float("nan")}


def summarise_injections(rows, suffix=""):
    out = []
    for lo, hi in SNR_BINS:
        sel = [r for r in rows if lo <= r["expected_snr"] < hi]
        line = {"expected_snr": f"{lo:g}-{hi:g}"}
        for arm in ("real", "white"):
            key = arm + suffix
            line[arm] = _binom(sum(r[key]["found"] == "period" for r in sel), len(sel))
            line[arm]["alias"] = sum(r[key]["found"] == "alias" for r in sel)
            line[arm]["other"] = sum(bool(r[key]["detected"]) and r[key]["found"] is None for r in sel)
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


def summarise_by_star(rows, suffix=""):
    out = {}
    for label in sorted({r["light_curve"] for r in rows}):
        sel = [r for r in rows if r["light_curve"] == label and r["expected_snr"] >= SNR_THRESHOLD]
        out[label] = {arm: _binom(sum(r[arm + suffix]["found"] == "period" for r in sel), len(sel))
                      for arm in ("real", "white")}
    return out


# ------------------------------------------- 4-5. trimming, and the pack's nuisances against the data
def _recover_robust(lc):
    return recover(lc, baseline="robust")


def _recover_trimmed(args):
    lc, trim = args
    return recover(lc, trim=trim)


def activity(lc):
    """RMS of the relative flux and the rotation period (highest Lomb-Scargle peak, 0.5-15 d, on
    30-minute means)."""
    from scidatasim.baselines.periodogram import frequency_grid, lomb_scargle
    ok = np.isfinite(lc["flux"])
    t, y = lc["time"][ok], lc["flux"][ok]
    bins = np.floor((t - t[0]) / (30 / 1440)).astype(int)
    tb = np.bincount(bins, t) / np.maximum(np.bincount(bins), 1)
    yb = np.bincount(bins, y) / np.maximum(np.bincount(bins), 1)
    keep = np.bincount(bins) > 0
    r = lomb_scargle(tb[keep], yb[keep], frequency=frequency_grid(tb[keep], 0.5, 15.0))
    return {"rms": float(np.std(y - np.median(y))), "period": float(1.0 / r["best_frequency"])}


def ramp_size(lc, after=1.0, min_gap=0.5):
    """Largest |median deviation| in 0.1-day steps within ``after`` days of each restart, and the
    time it takes to fall below a third of it: the amplitude and timescale of the worst ramp."""
    from scidatasim.nuisance import gap_edges
    ok = np.isfinite(lc["flux"])
    t, y = lc["time"][ok], lc["flux"][ok] / np.median(lc["flux"][ok])
    best = (0.0, 0.1)
    for s in gap_edges(t, min_gap):
        dev = []
        for a in np.arange(0.0, after, 0.1):
            m = (t >= s + a) & (t < s + a + 0.1)
            dev.append(float(np.median(y[m]) - 1.0) if m.sum() > 10 else 0.0)
        dev = np.abs(dev)
        if dev.size and dev.max() > best[0]:
            below = np.nonzero(dev < dev.max() / 3)[0]
            later = below[below > int(np.argmax(dev))]
            best = (float(dev.max()), 0.1 * float(later[0] if later.size else len(dev)) / np.log(3))
    return {"amplitude": best[0], "timescale": max(best[1], 0.05)}


def _pack_planet(rng, sigma, time):
    """A planted planet with an expected white-noise SNR uniform in 7.1-20 on ``time``."""
    from scidatasim.experiments.transits.model import Planet, a_over_rstar
    period = float(np.exp(rng.uniform(np.log(1.0), np.log(12.0))))
    t0, b, target = float(rng.uniform(0, period)), float(rng.uniform(0, 0.7)), float(rng.uniform(SNR_THRESHOLD, 20.0))
    rp = 0.02
    for _ in range(3):
        f = Planet(period, t0, rp, a_over_rstar(period), b).flux(time)
        snr = float(np.sqrt(np.sum((1 - f) ** 2)) / sigma)
        if snr <= 0:
            break
        rp = float(min(0.2, rp * np.sqrt(target / snr)))
    return {"period": period, "t0": t0, "rp": rp, "b": b}


def _pack_run(task):
    """One run of the pack (quiet, spotted or with ramps), searched as the real light curves."""
    from scidatasim.engine import simulate
    seed, extra, sigma, planted, trims, rounds = task
    rng = np.random.default_rng(seed)
    grid = np.arange(0.0, 27.4, CADENCE)
    grid = grid[(grid < 13.2) | (grid >= 14.2)]
    cfg = dict(experiment_type=EXP, seed=seed, planet=bool(planted), sigma_ppm=1e6 * sigma, **extra,
               observation={"targets": ["flux"], "operators": [{"type": "gaussian", "sigma": sigma}]})
    if planted:
        cfg["orbit"] = _pack_planet(rng, sigma, grid)
    sim = simulate(cfg)
    time, flux = (np.asarray(sim.results[k]["data"], float) for k in ("time", "flux_observed"))
    out = {"seed": seed, "planted": bool(planted)}
    planet = sim.logic.planet
    for trim in trims:
        found = iterative_search(time, flux, sigma, rounds if planted else 1, trim)
        if planted:
            tol = tolerance(planet.period, planet.duration, planet.transits_in(time))
            hit = any(r["detected"] and classify(r["period"], planet.period, tol) == "period" for r in found)
        else:
            hit = bool(found and found[0]["detected"])
        out[f"trim {trim:g}"] = hit
    return out


def pack_against_data(lcs, labels, sigmas, rec, nulls, injections, n_runs, rounds, trim, seed, pool):
    """The pack with the nuisances measured on the real light curves: AU Mic's spots (and a quiet
    twin), pi Men sector 4's ramp; planted planets found and false alarms, beside the real rates."""
    out = {"spotted": [], "ramps": []}
    tasks, keys = [], []
    for j, lc in enumerate(lcs):
        if lc["host"] != "AU Mic":
            continue
        act = activity(lc)
        spots = {"spots": {"amplitude": act["rms"], "period": act["period"]}}
        for kind, extra in (("spotted", spots), ("quiet twin", {})):
            for i in range(n_runs):
                tasks.append((seed + 10_000 * j + 100 * (kind == "quiet twin") + i, extra, sigmas[j], True, (0.0, trim),
                              rounds))
                keys.append((labels[j], kind, "planted", act))
            for i in range(max(2, n_runs // 2)):
                tasks.append((seed + 10_000 * j + 500 + 100 * (kind == "quiet twin") + i, extra, sigmas[j], False,
                              (0.0, trim), rounds))
                keys.append((labels[j], kind, "null", act))
    for j, lc in enumerate(lcs):
        if not (lc["host"] == "pi Men" and lc["sector"] == 4):
            continue
        ramp = ramp_size(lc)
        extra = {"ramps": {"amplitude": ramp["amplitude"], "timescale": ramp["timescale"]}}
        for i in range(n_runs):
            tasks.append((seed + 20_000 + i, extra, sigmas[j], False, (0.0, trim), rounds))
            keys.append((labels[j], "ramps", "null", ramp))
    runs = list(pool.map(_pack_run, tasks, chunksize=1))
    groups = {}
    for (label, kind, what, measured), r in zip(keys, runs):
        g = groups.setdefault((label, kind), {"light_curve": label, "kind": kind, "measured": measured})
        for k, v in r.items():
            if k.startswith("trim"):
                g.setdefault(what, {}).setdefault(k, []).append(v)
    for g in groups.values():
        for what in ("planted", "null"):
            if what in g:
                g[what] = {k: _binom(int(sum(v)), len(v)) for k, v in g[what].items()}
        label = g["light_curve"]
        sel = [r for r in injections if r["light_curve"] == label and r["expected_snr"] >= SNR_THRESHOLD]
        g["real_planted_found"] = _binom(sum(r["real"]["found"] == "period" for r in sel), len(sel))
        g["real_false_alarm"] = bool(nulls[label]["real"]["detected"] and not nulls[label]["real"].get("near_known"))
        g["real_false_alarm_trimmed"] = bool(nulls[label]["real_trimmed"]["detected"]
                                             and not nulls[label]["real_trimmed"].get("near_known"))
        (out["ramps"] if g["kind"] == "ramps" else out["spotted"]).append(g)
    return out


# ----------------------------------------------- 6-7. the second baseline, and instrument systematics
DUMP_BINS = np.round(np.arange(-0.3, 0.6001, 0.05), 3)


def dump_profile(t, flat, dumps):
    """Median detrended flux (ppm) in 0.05-day bins from 0.3 d before to 0.6 d after the dumps,
    and the largest |value| within 0.3 d after them, in units of its own error."""
    t, flat = np.asarray(t, float), np.asarray(flat, float)
    ok = np.isfinite(flat)
    t, f = t[ok], flat[ok] - 1.0
    sw = 1.4826 * np.median(np.abs(f - np.median(f)))
    prof, err = [], []
    for a, b in zip(DUMP_BINS[:-1], DUMP_BINS[1:]):
        vals = np.concatenate([f[(t >= d + a) & (t < d + b)] for d in dumps]) if len(dumps) else np.zeros(0)
        prof.append(float(np.median(vals)) if vals.size > 20 else float("nan"))
        err.append(float(1.2533 * sw / np.sqrt(vals.size)) if vals.size > 20 else float("nan"))
    prof, err = np.array(prof), np.array(err)
    after = (DUMP_BINS[:-1] >= 0) & (DUMP_BINS[:-1] < 0.3) & np.isfinite(prof)
    i = int(np.nanargmax(np.where(after, np.abs(prof), -1))) if after.any() else 0
    return {"profile_ppm": [None if not np.isfinite(v) else 1e6 * v for v in prof],
            "peak_ppm": float(1e6 * prof[i]) if after.any() else 0.0,
            "peak_significance": float(abs(prof[i]) / err[i]) if after.any() and err[i] > 0 else 0.0}


def systematics_of(lc, bg):
    """Momentum-dump signature and red noise of a light curve with its planets masked; and the
    null search with the half day after each dump dropped."""
    from scidatasim.experiments.transits.model import detrend, red_noise, transit_search
    ok = np.isfinite(bg)
    t, y = lc["time"][ok], bg[ok]
    flat = detrend(t, y, 0.5)
    sw, sr = red_noise(t, flat, (0.04, 0.08, 0.12, 0.2))
    near = np.zeros(t.size, bool)
    for d in lc["dumps"]:
        near |= (t >= d - 0.05) & (t < d + 0.3)
    sigma = scatter(lc["time"], lc["flux"])
    masked = transit_search(t[~near], y[~near], sigma)
    return {"dumps": int(len(lc["dumps"])),
            "dump_interval": float(np.median(np.diff(lc["dumps"]))) if len(lc["dumps"]) > 1 else None,
            **dump_profile(t, flat, lc["dumps"]), "white_ppm": 1e6 * sw, "red_ppm": {k: 1e6 * v for k, v in sr.items()},
            "null_without_dumps": {"detected": masked["detected"], "snr": masked["snr"], "period": masked["period"]}}


def _ar1_sigma_for(red, length, cadence=CADENCE, d=0.08, seed=0):
    """AR(1) stationary sigma that gives red noise ``red`` at duration ``d`` for correlation
    ``length`` (numerically: the ratio is computed once on a unit process)."""
    from scidatasim.experiments.transits.model import red_noise
    rng = np.random.default_rng(seed)
    n = int(27.0 / cadence)
    rho = np.exp(-cadence / length)
    z = rng.normal(size=n)
    x = np.empty(n)
    x[0] = z[0]
    scale = np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + scale * z[i]
    unit = red_noise(np.arange(n) * cadence, 1.0 + 1e-3 * x, (d,))[1][d] / 1e-3
    return red / unit if unit > 0 else 0.0


def _mirror_run(task):
    """A quiet star from the pack with white noise and AR(1) correlated noise at a light curve's
    measured levels, or with its momentum dumps; null search by both baselines."""
    from scidatasim.engine import simulate
    from scidatasim.experiments.transits.model import transit_search, transit_search_robust
    seed, sigma, ar_sigma, length, dumps = task
    ops = [{"type": "gaussian", "sigma": sigma}]
    if ar_sigma > 0:
        ops.append({"type": "ar1", "sigma": ar_sigma, "length": length})
    cfg = dict(experiment_type=EXP, seed=seed, planet=False, sigma_ppm=1e6 * sigma,
               observation={"targets": ["flux"], "operators": ops})
    if dumps:
        cfg["dumps"] = dumps
    sim = simulate(cfg)
    time, flux = (np.asarray(sim.results[k]["data"], float) for k in ("time", "flux_observed"))
    return {"median": bool(transit_search(time, flux, sigma)["detected"]),
            "robust": bool(transit_search_robust(time, flux, sigma)["detected"])}


def systematics_check(lcs, labels, sigmas, bgs, nulls, n_runs, seed, pool, length=0.05):
    rows, tasks, keys = [], [], []
    for j, lc in enumerate(lcs):
        info = systematics_of(lc, bgs[j])
        info["light_curve"] = labels[j]
        info["real_false_alarm"] = {b: bool(nulls[labels[j]][b]["detected"] and not nulls[labels[j]][b].get("near_known"))
                                    for b in ("real", "real robust")}
        info["ar1_sigma_ppm"] = 1e6 * _ar1_sigma_for(info["red_ppm"][0.08] / 1e6, length)
        rows.append(info)
        if lc["host"] == "AU Mic":
            continue  # spots, not the instrument: check 5
        for i in range(n_runs):
            tasks.append((seed + 1000 * j + i, sigmas[j], info["ar1_sigma_ppm"] / 1e6, length, None))
            keys.append((j, "correlated noise"))
    big = max((r for r, lc in zip(rows, lcs) if lc["host"] != "AU Mic"), key=lambda r: abs(r["peak_ppm"]))
    dumps = {"interval": big["dump_interval"] or 2.5, "amplitude": abs(big["peak_ppm"]) / 1e6, "timescale": 0.1}
    jq = labels.index(big["light_curve"])
    for i in range(n_runs):
        tasks.append((seed + 900_000 + i, sigmas[jq], 0.0, length, dumps))
        keys.append((jq, "dumps"))
    runs = list(pool.map(_mirror_run, tasks, chunksize=1))
    for (j, kind), r in zip(keys, runs):
        g = rows[j].setdefault("pack", {}).setdefault(kind, {"median": [], "robust": []})
        for b in ("median", "robust"):
            g[b].append(r[b])
    for row in rows:
        for kind, g in row.get("pack", {}).items():
            row["pack"][kind] = {b: _binom(int(sum(v)), len(v)) for b, v in g.items()}
    return {"rows": rows, "ar1_length_d": length, "dumps_simulated": {"light_curve": big["light_curve"], **dumps}}


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
    tr = rep["trimmed"]
    lines += ["", f"## 4. Trimming {tr['days']:g} d after the start and each gap", "",
              "| light curve | false alarm | false alarm, trimmed | known planets found | found, trimmed |",
              "|---|---|---|---|---|"]
    by_label = {f"{r['host']} s{r['sector']}": r for r in tr["recovery"]}
    for lc in rep["recovery"]:
        label = f"{lc['host']} s{lc['sector']}"
        v = rep["nulls"][label]

        def fa(r):
            if not r["detected"]:
                return "no"
            return "known planet's period" if r.get("near_known") else f"yes (SNR {r['snr']:.0f})"

        def n_found(rr):
            ps = [p for p in rr["planets"] if p["searchable"] and p["expected_snr"] >= SNR_THRESHOLD]
            return f"{sum(p['found'] == 'period' for p in ps)} of {len(ps)}"
        lines.append(f"| {label} | {fa(v['real'])} | {fa(v['real_trimmed'])} | {n_found(lc)} | {n_found(by_label[label])} |")
    pk = rep["pack_against_data"]
    lines += ["", "## 5. The pack's nuisance options against the real failures", "",
              "Spots: the pack with `spots` at the RMS and rotation period measured on each AU Mic sector, white noise at "
              f"its scatter, planets planted at expected SNR 7.1-20, and a quiet twin without spots; searched as is and "
              f"with {tr['days']:g} d trimmed after restarts. Real: the injections of check 2 (not trimmed) and the null "
              "searches of checks 3 and 4.", "",
              "| light curve | measured RMS, period | arm | planted found | found, trimmed | false alarms | "
              "false alarms, trimmed |", "|---|---|---|---|---|---|---|"]
    spotted = {g["light_curve"]: g for g in pk["spotted"] if g["kind"] == "spotted"}
    quiet = {g["light_curve"]: g for g in pk["spotted"] if g["kind"] == "quiet twin"}
    key_t = f"trim {tr['days']:g}"

    def frac(b):
        return f"{b['k']}/{b['n']}"
    for label, g in spotted.items():
        m = g["measured"]
        for arm, h in (("spotted pack", g), ("quiet twin", quiet[label])):
            lines.append(f"| {label} | {100 * m['rms']:.2f} %, {m['period']:.2f} d | {arm} | {frac(h['planted']['trim 0'])} | "
                         f"{frac(h['planted'][key_t])} | {frac(h['null']['trim 0'])} | {frac(h['null'][key_t])} |")
        lines.append(f"| {label} | | real | {frac(g['real_planted_found'])} | - | "
                     f"{'yes' if g['real_false_alarm'] else 'no'} | {'yes' if g['real_false_alarm_trimmed'] else 'no'} |")
    for g in pk["ramps"]:
        m = g["measured"]
        a, b = g["null"]["trim 0"], g["null"][f"trim {tr['days']:g}"]
        real = rep["nulls"][g["light_curve"]]
        lines += ["", f"Ramps: the pack with `ramps` of the size measured after {g['light_curve']}'s restarts "
                  f"({1e6 * m['amplitude']:.0f} ppm, timescale {m['timescale']:.2f} d) on a quiet star: false alarms "
                  f"{a['k']} of {a['n']} untrimmed, {b['k']} of {b['n']} trimmed {tr['days']:g} d; the real light curve: "
                  f"{'yes' if real['real']['detected'] else 'no'} untrimmed, "
                  f"{'yes' if real['real_trimmed']['detected'] else 'no'} trimmed."]
    rb = rep.get("robust")
    if rb:
        lines += ["", "## 6. The second baseline: local-quadratic detrending and red-noise SNR", "",
                  "`transit_search_robust`, beside the first baseline (running median, white-noise SNR).", "",
                  "| light curve | known planets found (1st / 2nd) | false alarm (1st / 2nd) | planted found above SNR 7.1, "
                  "real (1st / 2nd) | white (1st / 2nd) |", "|---|---|---|---|---|"]
        by2 = {f"{r['host']} s{r['sector']}": r for r in rb["recovery"]}

        def found_of(rr):
            ps = [p for p in rr["planets"] if p["searchable"] and p["expected_snr"] >= SNR_THRESHOLD]
            return sum(p["found"] == "period" for p in ps), len(ps)

        def fa2(r):
            if not r["detected"]:
                return "no"
            return "known period" if r.get("near_known") else f"SNR {r['snr']:.0f}"
        for lc in rep["recovery"]:
            label = f"{lc['host']} s{lc['sector']}"
            a, n = found_of(lc)
            b, _ = found_of(by2[label])
            v1, v2 = rep["injection"]["by_light_curve"][label], rb["by_light_curve"][label]
            lines.append(f"| {label} | {a} / {b} of {n} | {fa2(rep['nulls'][label]['real'])} / "
                         f"{fa2(rep['nulls'][label]['real_robust'])} | {v1['real']['k']} / {v2['real']['k']} of {v1['real']['n']} | "
                         f"{v1['white']['k']} / {v2['white']['k']} of {v1['white']['n']} |")
        lines += ["", "| expected SNR | n | 2nd baseline: found, real noise | found, white noise |", "|---|---|---|---|"]
        for bl in rb["by_expected_snr"]:
            lines.append(f"| {bl['expected_snr']} | {bl['real']['n']} | {pct(bl['real']['rate'])} | {pct(bl['white']['rate'])} |")
        wf = sum(v["white_robust_detected"] for v in rep["nulls"].values())
        lines += ["", f"White-noise twins, 2nd baseline: {wf} false alarms in "
                  f"{sum(v['white_n'] for v in rep['nulls'].values())} runs."]
    sy = rep.get("systematics")
    if sy:
        lines += ["", "## 7. Instrument systematics: momentum dumps and correlated noise", "",
                  "Measured on each light curve with its planets masked: the median detrended flux after the momentum dumps "
                  "TESS flags (largest 0.05-d bin within 0.3 d, and its significance), the white and red noise (Pont et al. "
                  "2006) at 0.08 d, and the null search with 0.35 d around each dump dropped. Then the pack, quiet star, with "
                  f"white noise and AR(1) noise (correlation length {sy['ar1_length_d']:g} d) at that light curve's measured "
                  "levels: false alarms by the two baselines, beside the real ones.", "",
                  "| light curve | dumps (interval) | after-dump peak | white / red at 0.08 d (ppm) | real null without dumps | "
                  "real false alarm (1st / 2nd) | pack with its correlated noise: false alarms (1st / 2nd) |",
                  "|---|---|---|---|---|---|---|"]
        for r in sy["rows"]:
            pk = (r.get("pack") or {}).get("correlated noise")
            pk_txt = f"{pk['median']['k']}/{pk['median']['n']} / {pk['robust']['k']}/{pk['robust']['n']}" if pk else "- (spots)"
            nd = r["null_without_dumps"]
            lines.append(f"| {r['light_curve']} | {r['dumps']} ({r['dump_interval'] or 0:.2f} d) | {r['peak_ppm']:+.0f} ppm "
                         f"({r['peak_significance']:.1f} sigma) | {r['white_ppm']:.0f} / {r['red_ppm'][0.08]:.0f} | "
                         f"{'yes' if nd['detected'] else 'no'} (SNR {nd['snr']:.0f}) | "
                         f"{'yes' if r['real_false_alarm']['real'] else 'no'} / {'yes' if r['real_false_alarm']['real robust'] else 'no'}"
                         f" | {pk_txt} |")
        ds = sy["dumps_simulated"]
        g = next(r for r in sy["rows"] if r["light_curve"] == ds["light_curve"])["pack"]["dumps"]
        lines += ["", f"The pack with `dumps` as large as the largest measured ({ds['light_curve']}: {1e6 * ds['amplitude']:.0f} ppm "
                  f"every {ds['interval']:.2f} d, decaying over {ds['timescale']:g} d) on a quiet star with that light curve's "
                  f"white noise: false alarms {g['median']['k']} of {g['median']['n']} (1st baseline), {g['robust']['k']} of "
                  f"{g['robust']['n']} (2nd)."]
    lines += ["", "## Verdict", ""] + [f"- {k}: {v}" for k, v in rep["verdict"].items()]
    return "\n".join(lines) + "\n"


def _pooled(by_lc, arm):
    k = sum(v[arm]["k"] for v in by_lc.values())
    n = sum(v[arm]["n"] for v in by_lc.values())
    return k / n if n else float("nan")


def _pooled_pack(sy, baseline):
    gs = [r["pack"]["correlated noise"][baseline] for r in sy["rows"] if (r.get("pack") or {}).get("correlated noise")]
    n = sum(g["n"] for g in gs)
    return sum(g["k"] for g in gs) / n if n else float("nan")


def verdict(rep):
    found = [p for lc in rep["recovery"] for p in lc["planets"] if p["searchable"]]
    above = [p for p in found if p["expected_snr"] >= SNR_THRESHOLD]
    inj = rep["injection"]["by_expected_snr"]
    clean = rep["injection"]["clean_light_curves"]
    tr = rep["trimmed"]["days"]
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
        "trimming after restarts": (
            f"false alarms {len(real_fa)} -> "
            f"{sum(1 for v in rep['nulls'].values() if v['real_trimmed']['detected'] and not v['real_trimmed'].get('near_known'))}"
            f" of {len(rep['nulls'])} with {rep['trimmed']['days']:g} d trimmed"),
        "the pack's spots": "; ".join(
            f"{g['light_curve']}: planted planets found {g['planted']['trim 0']['rate']:.2f} (quiet twin "
            f"{q['planted']['trim 0']['rate']:.2f}, real {g['real_planted_found']['rate']:.2f}), "
            f"{g['planted'][f'trim {tr:g}']['rate']:.2f} trimmed; false alarms {g['null']['trim 0']['rate']:.2f} "
            f"(real {'yes' if g['real_false_alarm'] else 'no'}), {g['null'][f'trim {tr:g}']['rate']:.2f} trimmed "
            f"(real {'yes' if g['real_false_alarm_trimmed'] else 'no'})"
            for g in rep["pack_against_data"]["spotted"] if g["kind"] == "spotted"
            for q in rep["pack_against_data"]["spotted"] if q["kind"] == "quiet twin" and q["light_curve"] == g["light_curve"]),
        "the pack's ramps": "; ".join(
            f"false alarms {g['null']['trim 0']['rate']:.2f} untrimmed, "
            f"{g['null'][next(k for k in g['null'] if k != 'trim 0')]['rate']:.2f} trimmed"
            for g in rep["pack_against_data"]["ramps"]),
        **({"second baseline": (
            f"known planets {sum(p['found'] == 'period' for r in rep['robust']['recovery'] for p in r['planets'] if p['searchable'] and p['expected_snr'] >= SNR_THRESHOLD)}"
            f" of {len(above)} (1st: {sum(p['found'] == 'period' for p in above)}); real false alarms "
            f"{sum(1 for v in rep['nulls'].values() if v['real_robust']['detected'] and not v['real_robust'].get('near_known'))}"
            f" of {len(rep['nulls'])} (1st: {len(real_fa)}); planted planets found above SNR 7.1 in real noise "
            f"{_pooled(rep['robust']['by_light_curve'], 'real'):.2f} (1st: {_pooled(rep['injection']['by_light_curve'], 'real'):.2f}),"
            f" in white noise {_pooled(rep['robust']['by_light_curve'], 'white'):.2f} "
            f"(1st: {_pooled(rep['injection']['by_light_curve'], 'white'):.2f})")} if rep.get("robust") else {}),
        **({"instrument systematics": (
            "momentum dumps: largest after-dump signal "
            f"{max(abs(r['peak_ppm']) for r in rep['systematics']['rows'] if not r['light_curve'].startswith('AU Mic')):.0f} ppm; "
            f"dropping the data after dumps removes {sum(1 for r in rep['systematics']['rows'] if r['real_false_alarm']['real'] and not r['null_without_dumps']['detected'])}"
            f" of {sum(1 for r in rep['systematics']['rows'] if r['real_false_alarm']['real'])} real false alarms; the pack with "
            "each quiet light curve's correlated noise gives false alarms in "
            f"{_pooled_pack(rep['systematics'], 'median'):.2f} of runs (1st baseline) and {_pooled_pack(rep['systematics'], 'robust'):.2f}"
            " (2nd)")} if rep.get("systematics") else {}),
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
    parser.add_argument("--trim", type=float, default=1.0,
                        help="days dropped after the start and each gap in check 4 (the remedy for ramps)")
    parser.add_argument("--pack-runs", type=int, default=12, help="pack runs per arm in check 5")
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
    null_tasks += [(args.seed + 500_000 + 1000 * j, bgs[j], sigmas[j], labels[j], "real, trimmed", lcs[j]["tic"], args.trim)
                   for j in range(len(lcs))]
    null_tasks += [(args.seed + 500_000 + 1000 * j, bgs[j], sigmas[j], labels[j], "real robust", lcs[j]["tic"], 0.0,
                    "robust") for j in range(len(lcs))]
    null_tasks += [(args.seed + 600_000 + 1000 * j + i, bgs[j], sigmas[j], labels[j], "white robust", lcs[j]["tic"], 0.0,
                    "robust") for j in range(len(lcs)) for i in range(args.white_nulls)]

    def arm_of(lab, arm):
        return next(n for n in nulls if n["light_curve"] == lab and n["arm"] == arm)
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        injections = list(pool.map(_inject, inj_tasks, chunksize=2))
        nulls = list(pool.map(_null, null_tasks, chunksize=2))
        rec_trim = list(pool.map(_recover_trimmed, [(lc, args.trim) for lc in lcs]))
        rec_robust = list(pool.map(_recover_robust, lcs))
        null_view = {lab: {"real": arm_of(lab, "real"), "real_trimmed": arm_of(lab, "real, trimmed"),
                           "real robust": arm_of(lab, "real robust")} for lab in labels}
        pack = pack_against_data(lcs, labels, sigmas, rec, null_view, injections, args.pack_runs, args.rounds, args.trim,
                                 args.seed + 700_000, pool)
        systematics = systematics_check(lcs, labels, sigmas, bgs, null_view, args.pack_runs, args.seed + 800_000, pool)
    found = [p for lc in rec for p in lc["planets"] if p["found"] == "period"]
    ratios = [p["snr_ratio"] for p in found if p.get("snr_ratio")]
    from scidatasim.provenance import numerical_environment
    rep = {"name": "transits on TESS", "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "settings": {"injections": args.injections, "rounds": args.rounds, "white_nulls": args.white_nulls,
                        "trim": args.trim, "pack_runs": args.pack_runs, "seed": args.seed},
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
                           "white_n": sum(1 for n in nulls if n["light_curve"] == lab and n["arm"] == "white"),
                           "real_trimmed": arm_of(lab, "real, trimmed"), "real_robust": arm_of(lab, "real robust"),
                           "white_robust_detected": sum(n["detected"] for n in nulls
                                                        if n["light_curve"] == lab and n["arm"] == "white robust")}
                     for lab in labels},
           "trimmed": {"days": args.trim, "recovery": rec_trim},
           "robust": {"recovery": rec_robust, "by_expected_snr": summarise_injections(injections, " robust"),
                      "by_light_curve": summarise_by_star(injections, " robust")},
           "systematics": systematics,
           "pack_against_data": pack,
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
