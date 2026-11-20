# scidatasim/observation.py
"""Observation operators: turn a model's exact (latent) outputs into
realistic measurements, as recorded, configurable knobs.

The mechanism is never modified. For every target output ``x`` the engine keeps
``x`` (role ``latent``) and adds ``x_observed`` (role ``observation``). If a
sampling scheme is configured, the observed series are evaluated only at the
sampled indices and matching coordinates (``time_observed``...) are added.

Configuration::

    observation:
      targets: [prey_population, predator_population]  # default: outputs with role "latent"
      sampling: {every: 5}            # or {n: 20, method: random|uniform}, or {indices: [...]}
      operators:                      # applied in order
        - {type: gaussian, snr_db: 20}          # or sigma: 2.0; snr_db uses the RMS of the whole
                                                # series (one noise level), so prefer multiplicative
                                                # noise for series spanning orders of magnitude
        - {type: multiplicative, sigma: 0.05}   # log-normal factor
        - {type: poisson, scale: 1.0}           # counts; scale = counts per unit
        - {type: quantize, step: 1.0}
        - {type: outliers, rate: 0.02, scale: 5.0}
        - {type: missing, rate: 0.1}            # NaN at random positions
        - {type: clip, low: 0.0}
      suffix: _observed

All random draws use the run's ``observation`` stream, so noise is reproducible
from the run seed and independent of the model's own randomness.

Beyond independent noise per point (operators on 1-D series unless noted):

* ``{type: ar1, snr_db: 30, rho: 0.8}`` - correlated noise: each error carries ``rho`` of the
  previous one (stationary AR(1)); ``length: 2.0`` instead sets a correlation length in coordinate
  units, for uneven spacing.
* ``{type: drift, rate: 0.01, random_walk: 0.002}`` - slow instrument drift: linear per
  coordinate unit (per index without a coordinate) and/or a random walk per point.
* ``{type: calibration, gain: 1.0, offset: 0.0, gain_sd: 0.02, offset_sd: 0.1}`` - one gain and
  offset error for the whole run (any shape).
* ``{type: censor, low: 0.5, replace: nan}`` - detection limits: values below ``low`` (or above
  ``high``) become NaN, the limit (``replace: limit``) or half the lower limit (``half``).
* ``{type: missing, rate: 0.1, mechanism: mnar, strength: 2, direction: high}`` - missingness
  that depends on the true value (``mnar``) or on the coordinate (``mar``); ``mcar`` (default) is
  the random kind. The overall rate stays ``rate``.
* ``{type: jitter, sigma: 0.05}`` - timing or position jitter: each point is measured at its
  coordinate plus N(0, sigma) but recorded at the nominal one (the latent series is interpolated
  linearly, so sample it finely). Must come first.

Fields (2-D images, or stacks such as ``(time, x, y)``; image operators act on the last two axes,
``axes: [a, b]`` to choose) are observed when named in ``targets``; sampling does not apply:

* ``{type: psf_blur, sigma: 1.5, boundary: reflect}`` - Gaussian point-spread function (pixels;
  ``boundary: wrap`` for periodic fields).
* ``{type: detector, scale: 100, read_noise: 5, gain: 2, full_well: 30000, bits: 12}`` - a camera:
  photon (Poisson) noise on ``value * scale`` electrons, Gaussian read noise, saturation, gain and
  ADC quantisation, converted back to data units (any shape).
* ``{type: bin, factor: 2}`` - pixel binning: block averages (the field gets smaller).
* ``{type: vignetting, strength: 0.3, power: 2}`` - brightness falling off towards the corners.
* ``{type: shift, dx: 0.3, dy: -0.2}`` or ``{type: shift, sd: 0.5}`` - sub-pixel misregistration.

Real noise (injection into real data; series and fields):

* ``{type: real_noise, source: lc.fits, time: TIME, column: PDCSAP_FLUX, quality: QUALITY,
  normalize: median, combine: multiply}`` - the simulated signal in a real background: a measured
  series (or image, ``field: true``) with the signal absent or removed. ``align: time`` (the default
  when ``time`` is given) takes, for each point, the background sample nearest its coordinate after
  a random shift (``shift`` to fix it; ``max_gap``, default half the cadence, leaves gaps as NaN);
  ``align: index`` takes consecutive samples from a random ``offset``. Fields take a window of the
  background image. ``combine: add`` (residuals, default) or ``multiply`` (relative flux ~1);
  ``scale`` multiplies the background's deviations. The file's path, SHA-256 and the offset or
  shift go into the run's record (see :mod:`scidatasim.ingest`). Choose the model's time step equal
  to the background's cadence, so each point gets its own real sample. The offset or shift stays in
  the private record: published together with the background file, it would let a solver subtract
  the noise exactly.

Nuisances measured on real data (:mod:`scidatasim.nuisance`; 1-D series, in coordinate units, per
index without a coordinate; ``combine: add`` (default) or ``multiply``, where the amplitude is
relative):

* ``{type: quasi_periodic, amplitude: 0.01, period: 4.9, lifetime: 15, harmonics: [1, 0.5, 0.25]}``
  - variability that repeats but evolves: starspots, seasonal cycles that change from year to
  year, vibration (RMS ``amplitude``; ``lifetime`` defaults to three periods).
* ``{type: flares, rate: 0.5, amplitude: [0.001, 0.02], index: 2, rise: 0.003, decay: 0.02}`` -
  sudden rises with exponential decays at random times (stellar flares, bursts, spikes).
* ``{type: ramps, amplitude: 0.005, timescale: 0.3, min_gap: 0.5, at_start: true, sign: -1}`` -
  an exponential settling after every gap in the coordinate longer than ``min_gap`` and at the
  start (thermal settling after a restart, sensor warm-up); ``amplitude: [low, high]`` draws one
  per restart, ``sign: random`` a random sign.
* ``{type: efficiency, shape: logistic, threshold: 20, width: 1.5}`` - a selection efficiency of
  the coordinate (``logistic``, ``erf``, or ``two_body`` with ``thresholds``, ``turn_on``,
  ``pair_pt``, ``eta_max``, ``rapidity_sd``: both daughters of a decay above momentum thresholds),
  multiplying the series (expected counts before ``poisson``; a survey's completeness).
* ``{type: response, sigma: 0.05, tail: {alpha: 1.2, n: 3, side: low}}`` - the detector response
  of a spectrum: what is true in one bin is spread over its neighbours by a Gaussian, or with
  ``tail`` a Crystal Ball (radiative losses, incomplete charge collection; ``side: high`` for
  tailing). ``relative: 0.01`` instead of ``sigma`` makes the width proportional to the coordinate;
  ``mixture: [[0.6, 0.75], [0.4, 1.4]]`` sums responses of those weights and width factors.

``gaussian``, ``multiplicative``, ``poisson``, ``quantize``, ``outliers``, ``missing`` (MCAR),
``calibration``, ``censor``, ``detector`` and ``clip`` work on any shape (dead pixels are
``missing``, hot pixels ``outliers``).
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .utils import DataDescriptor, DataType, Role

Info = Dict[str, Any]
ValueOperator = Callable[[np.ndarray, np.random.Generator, Dict[str, Any]], Tuple[np.ndarray, Info]]


def _rms(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else 0.0


def _noise_sigma(x, p, name):
    if ("sigma" in p) == ("snr_db" in p):
        raise ValueError(f"{name} operator needs exactly one of 'sigma' or 'snr_db'")
    if "sigma" in p:
        sigma = float(p["sigma"])
    else:
        sigma = _rms(x) / (10.0 ** (float(p["snr_db"]) / 20.0))
    if sigma < 0:
        raise ValueError(f"{name} sigma must be >= 0")
    return sigma


def _gaussian(x, rng, p):
    sigma = _noise_sigma(x, p, "gaussian")
    return x + rng.normal(0.0, sigma, size=x.shape), {"sigma": sigma}


def _multiplicative(x, rng, p):
    sigma = float(p["sigma"])
    if sigma < 0:
        raise ValueError("multiplicative sigma must be >= 0")
    # mean-one log-normal factor
    factor = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=x.shape)
    return x * factor, {"sigma": sigma}


def _poisson(x, rng, p):
    scale = float(p.get("scale", 1.0))
    if scale <= 0:
        raise ValueError("poisson scale must be > 0")
    lam = np.clip(np.nan_to_num(x, nan=0.0), 0.0, None) * scale
    out = rng.poisson(lam).astype(float) / scale
    out[np.isnan(x)] = np.nan
    return out, {"scale": scale}


def _quantize(x, rng, p):
    step = float(p["step"])
    if step <= 0:
        raise ValueError("quantize step must be > 0")
    return np.round(x / step) * step, {"step": step}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _missing(x, rng, p, ctx):
    rate = float(p["rate"])
    if not 0.0 <= rate <= 1.0:
        raise ValueError("missing rate must be in [0, 1]")
    mechanism = p.get("mechanism", "mcar")
    info: Info = {"rate": rate, "mechanism": mechanism}
    if mechanism == "mcar":
        mask = rng.random(x.shape) < rate
    elif mechanism in ("mar", "mnar"):
        if x.ndim != 1:
            raise ValueError("missing: mechanism 'mar' and 'mnar' apply to 1-D series")
        direction = p.get("direction", "high")
        if direction not in ("high", "low"):
            raise ValueError("missing direction must be 'high' or 'low'")
        strength = float(p.get("strength", 2.0))
        if mechanism == "mnar":  # depends on the true value: e.g. signals too weak or too strong
            score = np.asarray(ctx["latent"], float)
        else:  # depends on where the point is: e.g. late measurements are lost more often
            coord = ctx.get("coordinate")
            score = np.arange(x.size, dtype=float) if coord is None else np.asarray(coord, float)
        finite = np.isfinite(score)
        z = np.zeros(x.shape)
        if finite.any():
            spread = float(np.std(score[finite])) or 1.0
            z[finite] = (score[finite] - float(np.mean(score[finite]))) / spread
        z *= strength if direction == "high" else -strength
        if rate in (0.0, 1.0):
            prob = np.full(x.shape, rate)
        else:  # intercept so that the expected missing share is ``rate``
            lo, hi = -60.0, 60.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if _sigmoid(mid + z).mean() < rate else (lo, mid)
            prob = _sigmoid(0.5 * (lo + hi) + z)
        mask = rng.random(x.shape) < prob
        info.update(strength=strength, direction=direction)
    else:
        raise ValueError("missing mechanism must be 'mcar', 'mar' or 'mnar'")
    out = x.astype(float, copy=True)
    out[mask] = np.nan
    info["n_missing"] = int(mask.sum())
    return out, info


def _ar1(x, rng, p, ctx):
    if x.ndim != 1:
        raise ValueError("ar1 applies to 1-D series")
    sigma = _noise_sigma(x, p, "ar1")
    if ("rho" in p) == ("length" in p):
        raise ValueError("ar1 operator needs exactly one of 'rho' or 'length'")
    n = x.size
    if "rho" in p:
        rho0 = float(p["rho"])
        if not -1.0 < rho0 < 1.0:
            raise ValueError("ar1 rho must be in (-1, 1)")
        rho = np.full(max(n - 1, 0), rho0)
        info: Info = {"sigma": sigma, "rho": rho0}
    else:
        length = float(p["length"])
        coord = ctx.get("coordinate")
        if coord is None:
            raise ValueError("ar1 'length' needs the series' coordinate (its x_axis)")
        if length <= 0:
            raise ValueError("ar1 length must be > 0")
        rho = np.exp(-np.abs(np.diff(np.asarray(coord, float))) / length)
        info = {"sigma": sigma, "length": length}
    z = rng.normal(size=n)
    err = np.empty(n)
    if n:
        err[0] = sigma * z[0]
    for i in range(1, n):  # stationary: every error has standard deviation sigma
        err[i] = rho[i - 1] * err[i - 1] + sigma * np.sqrt(1.0 - rho[i - 1] ** 2) * z[i]
    return x + err, info


def _drift(x, rng, p, ctx):
    if x.ndim != 1:
        raise ValueError("drift applies to 1-D series")
    rate, walk = float(p.get("rate", 0.0)), float(p.get("random_walk", 0.0))
    if walk < 0:
        raise ValueError("drift random_walk must be >= 0")
    if not rate and not walk:
        raise ValueError("drift needs 'rate' and/or 'random_walk'")
    coord = ctx.get("coordinate")
    position = np.arange(x.size, dtype=float) if coord is None else np.asarray(coord, float)
    offset = rate * (position - position[0]) if x.size else np.zeros(0)
    if walk:
        offset = offset + np.cumsum(rng.normal(0.0, walk, size=x.size))
    return x + offset, {"rate": rate, "random_walk": walk, "per": "index" if coord is None else "coordinate unit",
                        "final_offset": float(offset[-1]) if x.size else 0.0}


def _calibration(x, rng, p):
    gain_sd, offset_sd = float(p.get("gain_sd", 0.0)), float(p.get("offset_sd", 0.0))
    if gain_sd < 0 or offset_sd < 0:
        raise ValueError("calibration gain_sd and offset_sd must be >= 0")
    gain = float(p.get("gain", 1.0)) + (rng.normal(0.0, gain_sd) if gain_sd else 0.0)
    offset = float(p.get("offset", 0.0)) + (rng.normal(0.0, offset_sd) if offset_sd else 0.0)
    return gain * x + offset, {"gain": float(gain), "offset": float(offset)}


def _censor(x, rng, p):
    low, high = p.get("low"), p.get("high")
    if low is None and high is None:
        raise ValueError("censor needs 'low' and/or 'high'")
    replace = p.get("replace", "nan")
    if replace not in ("nan", "limit", "half"):
        raise ValueError("censor replace must be 'nan', 'limit' or 'half'")
    if replace == "half" and low is None:
        raise ValueError("censor replace 'half' needs 'low'")
    out = x.astype(float, copy=True)
    with np.errstate(invalid="ignore"):
        below = np.zeros(x.shape, bool) if low is None else x < float(low)
        above = np.zeros(x.shape, bool) if high is None else x > float(high)
    if replace == "nan":
        out[below | above] = np.nan
    else:
        out[below] = float(low) / 2.0 if replace == "half" else (float(low) if low is not None else np.nan)
        out[above] = float(high) if high is not None else np.nan
    return out, {"low": low, "high": high, "replace": replace, "n_below": int(below.sum()),
                 "n_above": int(above.sum()), "indices_below": np.flatnonzero(below).tolist(),
                 "indices_above": np.flatnonzero(above).tolist()}


def _jitter(x, rng, p, ctx):
    if x.ndim != 1:
        raise ValueError("jitter applies to 1-D series")
    sigma = float(p["sigma"])
    if sigma < 0:
        raise ValueError("jitter sigma must be >= 0")
    coord, x_full, latent_full = ctx.get("coordinate"), ctx.get("x_full"), ctx.get("latent_full")
    if coord is None or x_full is None:
        raise ValueError("jitter needs the series' coordinate (its x_axis)")
    x_full = np.asarray(x_full, float)
    if x_full.size > 1 and not np.all(np.diff(x_full) > 0):
        raise ValueError("jitter needs an increasing coordinate")
    shift = rng.normal(0.0, sigma, size=np.shape(coord))
    values = np.interp(np.asarray(coord, float) + shift, x_full, np.asarray(latent_full, float))
    return values, {"sigma": sigma, "rms_shift": float(np.sqrt(np.mean(shift ** 2))) if shift.size else 0.0}


# ------------------------------------------------------------------ fields (images)
def _image_axes(x, p):
    if x.ndim < 2:
        raise ValueError("image operators apply to fields with at least 2 dimensions")
    axes = tuple(int(a) % x.ndim for a in p.get("axes", (-2, -1)))
    if len(axes) != 2 or axes[0] == axes[1]:
        raise ValueError("axes must name two different axes")
    return axes


def _psf_blur(x, rng, p):
    from scipy.ndimage import gaussian_filter
    axes = _image_axes(x, p)
    sigma = p["sigma"]
    sy, sx = (float(sigma), float(sigma)) if np.isscalar(sigma) else (float(sigma[0]), float(sigma[1]))
    if sy < 0 or sx < 0:
        raise ValueError("psf_blur sigma must be >= 0")
    boundary = p.get("boundary", "reflect")
    if boundary not in ("reflect", "wrap", "nearest", "constant"):
        raise ValueError("psf_blur boundary must be reflect, wrap, nearest or constant")
    widths = [0.0] * x.ndim
    widths[axes[0]], widths[axes[1]] = sy, sx
    return gaussian_filter(x.astype(float), sigma=widths, mode=boundary), {"sigma": [sy, sx], "boundary": boundary}


def _detector(x, rng, p):
    scale, gain = float(p.get("scale", 1.0)), float(p.get("gain", 1.0))
    read_noise, bias = float(p.get("read_noise", 0.0)), float(p.get("bias", 0.0))
    full_well, bits = p.get("full_well"), p.get("bits")
    if scale <= 0 or gain <= 0 or read_noise < 0:
        raise ValueError("detector needs scale > 0, gain > 0 and read_noise >= 0")
    if bits is not None and not 1 <= int(bits) <= 32:
        raise ValueError("detector bits must be in [1, 32]")
    electrons = rng.poisson(np.clip(np.nan_to_num(x, nan=0.0), 0.0, None) * scale).astype(float)
    electrons += rng.normal(0.0, read_noise, size=x.shape) if read_noise else 0.0
    saturated = np.zeros(x.shape, bool)
    if full_well is not None:
        saturated |= electrons >= float(full_well)
        electrons = np.minimum(electrons, float(full_well))
    adu = electrons / gain + bias
    if bits is not None:
        top = 2.0 ** int(bits) - 1
        adu = np.clip(np.round(adu), 0.0, top)
        saturated |= adu >= top
    out = (adu - bias) * gain / scale  # back in the field's own units
    out[np.isnan(x)] = np.nan
    return out, {"scale": scale, "gain": gain, "read_noise": read_noise, "bias": bias, "full_well": full_well,
                 "bits": None if bits is None else int(bits), "n_saturated": int(saturated.sum())}


def _bin(x, rng, p):
    axes = _image_axes(x, p)
    k = int(p["factor"])
    if k < 1:
        raise ValueError("bin factor must be >= 1")
    moved = np.moveaxis(x.astype(float), axes, (-2, -1))
    ny, nx = moved.shape[-2] // k, moved.shape[-1] // k
    if ny == 0 or nx == 0:
        raise ValueError("bin factor is larger than the field")
    cropped = moved[..., : ny * k, : nx * k]
    binned = cropped.reshape(*cropped.shape[:-2], ny, k, nx, k).mean(axis=(-3, -1))
    return np.moveaxis(binned, (-2, -1), axes), {"factor": k, "cropped": [int(moved.shape[-2] - ny * k),
                                                                          int(moved.shape[-1] - nx * k)]}


def _vignetting(x, rng, p):
    axes = _image_axes(x, p)
    strength, power = float(p.get("strength", 0.3)), float(p.get("power", 2.0))
    if not 0.0 <= strength < 1.0 or power <= 0:
        raise ValueError("vignetting needs 0 <= strength < 1 and power > 0")
    ny, nx = x.shape[axes[0]], x.shape[axes[1]]
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, ny), np.linspace(-1.0, 1.0, nx), indexing="ij")
    factor = 1.0 - strength * (np.sqrt(xx ** 2 + yy ** 2) / np.sqrt(2.0)) ** power  # 1 at the centre
    shape = [1] * x.ndim
    shape[axes[0]], shape[axes[1]] = ny, nx
    return x * factor.reshape(shape), {"strength": strength, "power": power,
                                       "corner_factor": float(1.0 - strength)}


def _shift(x, rng, p):
    from scipy.ndimage import shift as nd_shift
    axes = _image_axes(x, p)
    if "sd" in p:
        sd = float(p["sd"])
        if sd < 0:
            raise ValueError("shift sd must be >= 0")
        dy, dx = (float(v) for v in rng.normal(0.0, sd, size=2))
    else:
        dy, dx = float(p.get("dy", 0.0)), float(p.get("dx", 0.0))
    offsets = [0.0] * x.ndim
    offsets[axes[0]], offsets[axes[1]] = dy, dx
    boundary = p.get("boundary", "nearest")
    return nd_shift(x.astype(float), offsets, order=1, mode=boundary), {"dy": dy, "dx": dx, "boundary": boundary}


def _outliers(x, rng, p):
    rate, scale = float(p["rate"]), float(p.get("scale", 5.0))
    if not 0.0 <= rate <= 1.0:
        raise ValueError("outliers rate must be in [0, 1]")
    finite = x[np.isfinite(x)]
    spread = float(np.std(finite)) if finite.size else 0.0
    spread = spread if spread > 0 else max(abs(_rms(x)), 1.0)
    mask = rng.random(x.shape) < rate
    signs = rng.choice([-1.0, 1.0], size=x.shape)
    out = x.astype(float, copy=True)
    out[mask] += signs[mask] * scale * spread
    return out, {"rate": rate, "scale": scale, "magnitude": scale * spread,
                 "indices": np.flatnonzero(mask).tolist()}


def _clip(x, rng, p):
    low, high = p.get("low"), p.get("high")
    return np.clip(x, low if low is not None else -np.inf, high if high is not None else np.inf), \
        {"low": low, "high": high}


def _background(p) -> Tuple[np.ndarray, Optional[np.ndarray], Info]:
    """The real background: inline ``values`` (and ``times``), a field file, or a series file."""
    if "values" in p:
        v = np.asarray(p["values"], float)
        t = None if p.get("times") is None else np.asarray(p["times"], float)
        return v, t, {"source": "inline", "n": int(v.size)}
    if "source" not in p:
        raise ValueError("real_noise needs 'source' (a file) or 'values'")
    from . import ingest
    if p.get("field"):
        f = ingest.read_field(p["source"], fmt=p.get("format"), key=p.get("key"))
        return np.asarray(f["data"], float), None, {"source": f["provenance"]}
    s = ingest.read_series(p["source"], value=p.get("column", "value"), time=p.get("time"),
                           error=p.get("error"), quality=p.get("quality"),
                           good_quality=p.get("good_quality", (0,)), normalize=p.get("normalize"),
                           fmt=p.get("format"), key=p.get("key"))
    info = {"source": s.provenance, "column": s.meta["value"], "n": len(s)}
    if s.meta.get("flagged") is not None:
        info["flagged"] = s.meta["flagged"]
    return s.value, (s.time if p.get("time") else None), info


def _real_noise(x, rng, p, ctx):
    """A simulated signal in real noise: ``x`` combined with a real background (a measured series
    or image with the signal absent or removed). The noise, its correlations and its gaps (NaN in
    the background) are the real ones; the truth is still the latent ``x``."""
    combine = p.get("combine", "add")
    if combine not in ("add", "multiply"):
        raise ValueError("real_noise combine must be 'add' or 'multiply'")
    scale = float(p.get("scale", 1.0))
    bg, bg_t, info = _background(p)
    info.update(combine=combine, scale=scale)
    if x.ndim >= 2:  # a field: a window of the background image with the same shape
        if bg.ndim != x.ndim or any(b < s for b, s in zip(bg.shape, x.shape)):
            raise ValueError(f"real_noise: background shape {bg.shape} cannot hold a {x.shape} window")
        start = p.get("offset")
        start = [int(rng.integers(0, b - s + 1)) for b, s in zip(bg.shape, x.shape)] if start is None \
            else [int(o) for o in (start if isinstance(start, (list, tuple)) else [start] * x.ndim)]
        window = bg[tuple(slice(o, o + s) for o, s in zip(start, x.shape))]
        info["offset"] = start
    else:
        align = p.get("align", "time" if bg_t is not None else "index")
        info["align"] = align
        if align == "index":  # consecutive background samples, one per point
            if bg.size < x.size:
                raise ValueError(f"real_noise: background has {bg.size} points, the series {x.size}")
            start = int(p["offset"]) if p.get("offset") is not None else int(rng.integers(0, bg.size - x.size + 1))
            window = bg[start:start + x.size]
            info["offset"] = start
        elif align == "time":  # the background sample nearest each point's coordinate, after a shift
            coord = ctx.get("coordinate")
            if coord is None or bg_t is None:
                raise ValueError("real_noise align 'time' needs the series' coordinate and background times")
            order = np.argsort(bg_t)
            bt, bv = bg_t[order], bg[order]
            cadence = float(np.median(np.diff(bt))) if bt.size > 1 else 1.0
            span = float(np.max(coord) - np.min(coord))
            if p.get("shift") is not None:
                shift = float(p["shift"])
            else:
                lo, hi = bt[0] - np.min(coord), bt[-1] - np.min(coord) - span
                if hi < lo:
                    raise ValueError(f"real_noise: background spans {bt[-1] - bt[0]:g}, the series {span:g}")
                shift = float(rng.uniform(lo, hi))
            target = np.asarray(coord, float) + shift
            k = np.clip(np.searchsorted(bt, target), 1, bt.size - 1)
            k = np.where(np.abs(bt[k - 1] - target) <= np.abs(bt[k] - target), k - 1, k)
            max_gap = float(p.get("max_gap", 0.5 * cadence))
            window = np.where(np.abs(bt[k] - target) <= max_gap, bv[k], np.nan)
            info.update(shift=shift, cadence=cadence, max_gap=max_gap)
        else:
            raise ValueError("real_noise align must be 'index' or 'time'")
    window = np.asarray(window, float)
    if combine == "add":
        out = x + scale * window
    else:  # relative flux: the background is ~1, its deviations scaled
        out = x * (1.0 + scale * (window - 1.0))
    info["n_gaps"] = int(np.sum(~np.isfinite(window)))
    return out, info


def _position(x, ctx):
    coord = ctx.get("coordinate")
    return np.arange(x.size, dtype=float) if coord is None else np.asarray(coord, float)


def _combine(x, extra, p, name):
    how = p.get("combine", "add")
    if how == "add":
        return x + extra
    if how == "multiply":
        return x * (1.0 + extra)
    raise ValueError(f"{name} combine must be 'add' or 'multiply'")


def _quasi_periodic(x, rng, p, ctx):
    from .nuisance import quasi_periodic
    if x.ndim != 1:
        raise ValueError("quasi_periodic applies to 1-D series")
    period = float(p["period"])
    lifetime = float(p.get("lifetime", 3.0 * period))
    harmonics = [float(v) for v in p.get("harmonics", (1.0, 0.5, 0.25))]
    extra = quasi_periodic(_position(x, ctx), rng, float(p["amplitude"]), period, lifetime, harmonics)
    return _combine(x, extra, p, "quasi_periodic"), {
        "amplitude": float(p["amplitude"]), "period": period, "lifetime": lifetime, "harmonics": harmonics,
        "combine": p.get("combine", "add"), "realized_rms": float(np.std(extra)) if x.size else 0.0}


def _flares(x, rng, p, ctx):
    from .nuisance import flares
    if x.ndim != 1:
        raise ValueError("flares applies to 1-D series")
    amp = tuple(float(v) for v in p.get("amplitude", (1e-3, 2e-2)))
    extra, events = flares(_position(x, ctx), rng, float(p["rate"]), amp, float(p.get("index", 2.0)),
                           float(p.get("rise", 0.003)), float(p.get("decay", 0.02)))
    return _combine(x, extra, p, "flares"), {"rate": float(p["rate"]), "amplitude": list(amp),
                                              "combine": p.get("combine", "add"), "flares": events}


def _ramps(x, rng, p, ctx):
    from .nuisance import restart_ramps
    if x.ndim != 1:
        raise ValueError("ramps applies to 1-D series")
    amp = p["amplitude"]
    amp = [float(v) for v in amp] if isinstance(amp, (list, tuple)) else float(amp)
    sign = p.get("sign", -1.0)
    sign = None if sign == "random" else float(sign)
    extra, events = restart_ramps(_position(x, ctx), rng, amp, float(p.get("timescale", 0.3)),
                                  float(p.get("min_gap", 0.5)), bool(p.get("at_start", True)), sign)
    return _combine(x, extra, p, "ramps"), {"timescale": float(p.get("timescale", 0.3)),
                                             "combine": p.get("combine", "add"), "ramps": events}


def selection_efficiency(coordinate, p: Dict[str, Any]) -> np.ndarray:
    """A selection efficiency of the coordinate (see the ``efficiency`` operator)."""
    from .nuisance import erf_efficiency, logistic_efficiency, two_body_efficiency
    shape = p.get("shape", "logistic")
    c = np.asarray(coordinate, float)
    if shape in ("logistic", "erf"):
        f = logistic_efficiency if shape == "logistic" else erf_efficiency
        return f(c, float(p.get("threshold", 20.0)), float(p.get("width", 1.5)), float(p.get("plateau", 1.0)))
    if shape == "two_body":
        keys = ("thresholds", "turn_on", "pair_pt", "eta_max", "rapidity_sd", "daughter_mass", "n", "seed")
        kw = {k: p[k] for k in keys if k in p}
        return float(p.get("plateau", 1.0)) * two_body_efficiency(c, **kw)
    raise ValueError("efficiency shape must be 'logistic', 'erf' or 'two_body'")


def _efficiency(x, rng, p, ctx):
    if x.ndim != 1:
        raise ValueError("efficiency applies to 1-D series")
    eff = selection_efficiency(_position(x, ctx), p)
    return x * eff, {"shape": p.get("shape", "logistic"), "min": float(eff.min()) if x.size else 0.0,
                     "max": float(eff.max()) if x.size else 0.0}


def _response(x, rng, p, ctx):
    from .nuisance import response_matrix
    if x.ndim != 1:
        raise ValueError("response applies to 1-D series")
    pos = _position(x, ctx)
    if ("sigma" in p) == ("relative" in p):
        raise ValueError("response needs exactly one of 'sigma' or 'relative'")
    sigma = float(p["sigma"]) if "sigma" in p else float(p["relative"]) * np.abs(pos)
    R = response_matrix(pos, sigma, p.get("tail"), p.get("mixture"))
    ok = np.isfinite(x)
    out = R[:, ok] @ x[ok]
    return out, {"sigma": p.get("sigma"), "relative": p.get("relative"), "tail": p.get("tail"),
                 "mixture": p.get("mixture"),
                 "kept_share": float(np.sum(out) / np.sum(x[ok])) if np.sum(x[ok]) else None}


VALUE_OPERATORS: Dict[str, ValueOperator] = {
    "gaussian": _gaussian,
    "multiplicative": _multiplicative,
    "poisson": _poisson,
    "quantize": _quantize,
    "outliers": _outliers,
    "clip": _clip,
    "calibration": _calibration,
    "censor": _censor,
    "psf_blur": _psf_blur,
    "detector": _detector,
    "bin": _bin,
    "vignetting": _vignetting,
    "shift": _shift,
}
#: Operators that also need the context: ``func(values, rng, params, ctx)`` with ``ctx`` holding
#: ``coordinate`` (the observed points' x values, or None), ``latent`` (the exact values there),
#: ``x_full`` and ``latent_full`` (the whole series before sampling).
CONTEXT_OPERATORS: Dict[str, Callable[..., Tuple[np.ndarray, Info]]] = {
    "missing": _missing,
    "ar1": _ar1,
    "drift": _drift,
    "jitter": _jitter,
    "real_noise": _real_noise,
    "quasi_periodic": _quasi_periodic,
    "flares": _flares,
    "ramps": _ramps,
    "efficiency": _efficiency,
    "response": _response,
}
#: Operators that only make sense first (they start again from the exact values).
FIRST_ONLY = {"jitter"}


def operator_types() -> List[str]:
    return sorted(set(VALUE_OPERATORS) | set(CONTEXT_OPERATORS))


def register_operator(name: str, func: Callable[..., Tuple[np.ndarray, Info]], context: bool = False) -> None:
    """Register a custom operator ``func(values, rng, params) -> (values, info)``, or with
    ``context=True`` ``func(values, rng, params, ctx)`` (see :data:`CONTEXT_OPERATORS`)."""
    if name in VALUE_OPERATORS or name in CONTEXT_OPERATORS:
        raise ValueError(f"Operator '{name}' is already registered")
    (CONTEXT_OPERATORS if context else VALUE_OPERATORS)[name] = func


def validate_observation_config(obs: Any) -> Dict[str, Any]:
    if obs is None:
        return {}
    if not isinstance(obs, dict):
        raise ValueError("'observation' must be a mapping")
    unknown = set(obs) - {"targets", "sampling", "operators", "suffix"}
    if unknown:
        raise ValueError(f"Unknown observation keys: {sorted(unknown)}")
    for i, op in enumerate(obs.get("operators") or []):
        if not isinstance(op, dict) or "type" not in op:
            raise ValueError(f"observation.operators[{i}] must be a mapping with a 'type'")
        if op["type"] not in VALUE_OPERATORS and op["type"] not in CONTEXT_OPERATORS:
            raise ValueError(f"observation.operators[{i}]: unknown type '{op['type']}'. "
                             f"Available: {operator_types()}")
        if op["type"] in FIRST_ONLY and i != 0:
            raise ValueError(f"observation.operators[{i}]: '{op['type']}' must be the first operator "
                             "(it starts from the exact values)")
    sampling = obs.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, dict) or not ({"every", "n", "indices"} & set(sampling)):
            raise ValueError("observation.sampling needs one of 'every', 'n' or 'indices'")
    targets = obs.get("targets")
    if targets is not None and not (isinstance(targets, list) and all(isinstance(t, str) for t in targets)):
        raise ValueError("observation.targets must be a list of output names")
    return obs


def _sample_indices(length: int, sampling: Optional[Dict[str, Any]], rng) -> Tuple[Optional[np.ndarray], Info]:
    if not sampling:
        return None, {}
    if "indices" in sampling:
        idx = np.asarray(sampling["indices"], dtype=int)
        if idx.size and (idx.min() < 0 or idx.max() >= length):
            raise ValueError("observation.sampling.indices out of range")
        return np.unique(idx), {"method": "indices"}
    if "every" in sampling:
        every = int(sampling["every"])
        offset = int(sampling.get("offset", 0))
        if every < 1:
            raise ValueError("observation.sampling.every must be >= 1")
        if not 0 <= offset < length:
            raise ValueError(f"observation.sampling.offset must be in [0, {length})")
        return np.arange(offset, length, every), {"method": "every", "every": every, "offset": offset}
    n = int(sampling["n"])
    if not 1 <= n <= length:
        raise ValueError(f"observation.sampling.n must be in [1, {length}]")
    method = sampling.get("method", "uniform")
    if method == "uniform":
        idx = np.unique(np.round(np.linspace(0, length - 1, n)).astype(int))
    elif method == "random":
        idx = np.sort(rng.choice(length, size=n, replace=False))
    else:
        raise ValueError("observation.sampling.method must be 'uniform' or 'random'")
    return idx, {"method": method, "n": n}


def apply_observation(results: Dict[str, Dict[str, Any]], obs: Dict[str, Any],
                      rng: np.random.Generator) -> Tuple[Dict[str, Dict[str, Any]], Info]:
    """Return ``(new_results, realized)`` where ``new_results`` holds the added
    observation/coordinate entries and ``realized`` records what was applied
    (including noise levels derived from SNRs and outlier positions)."""
    obs = validate_observation_config(obs)
    if not obs:
        return {}, {}
    suffix = obs.get("suffix", "_observed")
    targets = obs.get("targets")
    if targets is None:
        # Default: every 1-D numeric latent series (fields of higher dimension, e.g.
        # PDE solutions, need explicit targets and are not supported by the
        # operators yet).
        targets = [name for name, info in results.items() if info["descriptor"].role == Role.LATENT
                   and isinstance(info["data"], np.ndarray) and info["data"].ndim == 1
                   and np.issubdtype(info["data"].dtype, np.number)]
        if not targets:
            raise ValueError("observation.targets not given and no output is a 1-D latent series")
    for name in targets:
        if name not in results:
            raise KeyError(f"observation target '{name}' is not an output of the model")
        data = results[name]["data"]
        if not (isinstance(data, np.ndarray) and data.ndim >= 1 and np.issubdtype(data.dtype, np.number)):
            raise TypeError(f"observation target '{name}' must be a numeric array (a series or a field)")
    if obs.get("sampling") and any(results[t]["data"].ndim != 1 for t in targets):
        raise ValueError("observation.sampling applies to 1-D series only, not to fields")

    lengths = {len(results[t]["data"]) for t in targets}
    if obs.get("sampling") and len(lengths) != 1:
        raise ValueError("sampling requires all observation targets to have the same length")
    indices, sampling_info = _sample_indices(lengths.pop() if len(lengths) == 1 else 0,
                                             obs.get("sampling"), rng)

    added: Dict[str, Dict[str, Any]] = {}
    realized: Info = {"suffix": suffix, "targets": {}, "sampling": sampling_info or None}

    coordinate_map: Dict[str, str] = {}
    if indices is not None:
        added["observation_index"] = {
            "data": indices,
            "descriptor": DataDescriptor("observation_index", DataType.NDARRAY, shape=indices.shape,
                                         units="index", group="observation", role=Role.AUXILIARY),
        }
        for t in targets:
            x_name = results[t]["descriptor"].x_axis
            if x_name and x_name in results and x_name not in coordinate_map:
                x_info = results[x_name]
                new_name = f"{x_name}{suffix}"
                x_obs = np.asarray(x_info["data"])[indices]
                added[new_name] = {
                    "data": x_obs,
                    "descriptor": x_info["descriptor"].replace(name=new_name, shape=x_obs.shape,
                                                               role=Role.COORDINATE, plot_type=None),
                }
                coordinate_map[x_name] = new_name

    for t in targets:
        exact = np.asarray(results[t]["data"], dtype=float)
        values = exact if indices is None else exact[indices]
        desc = results[t]["descriptor"]
        x_full = None
        if exact.ndim == 1 and desc.x_axis and desc.x_axis in results:
            candidate = np.asarray(results[desc.x_axis]["data"])
            if candidate.shape == exact.shape and np.issubdtype(candidate.dtype, np.number):
                x_full = candidate.astype(float)
        ctx = {"coordinate": None if x_full is None else (x_full if indices is None else x_full[indices]),
               "latent": values.copy(), "x_full": x_full, "latent_full": exact}
        steps: List[Info] = []
        for op in obs.get("operators") or []:
            params = {k: v for k, v in op.items() if k != "type"}
            if op["type"] in CONTEXT_OPERATORS:
                values, info = CONTEXT_OPERATORS[op["type"]](values, rng, params, ctx)
            else:
                values, info = VALUE_OPERATORS[op["type"]](values, rng, params)
            steps.append({"type": op["type"], **info})
        new_name = f"{t}{suffix}"
        added[new_name] = {
            "data": values,
            "descriptor": desc.replace(name=new_name, shape=values.shape, role=Role.OBSERVATION,
                                       plot_type="scatter" if values.ndim == 1 else desc.plot_type,
                                       x_axis=coordinate_map.get(desc.x_axis, desc.x_axis)),
        }
        realized["targets"][t] = {"observed_as": new_name, "operators": steps}
    return added, copy.deepcopy(realized)
