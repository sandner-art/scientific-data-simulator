# simulator/observation.py
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


def _gaussian(x, rng, p):
    if ("sigma" in p) == ("snr_db" in p):
        raise ValueError("gaussian operator needs exactly one of 'sigma' or 'snr_db'")
    if "sigma" in p:
        sigma = float(p["sigma"])
    else:
        sigma = _rms(x) / (10.0 ** (float(p["snr_db"]) / 20.0))
    if sigma < 0:
        raise ValueError("gaussian sigma must be >= 0")
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


def _missing(x, rng, p):
    rate = float(p["rate"])
    if not 0.0 <= rate <= 1.0:
        raise ValueError("missing rate must be in [0, 1]")
    mask = rng.random(x.shape) < rate
    out = x.astype(float, copy=True)
    out[mask] = np.nan
    return out, {"rate": rate, "n_missing": int(mask.sum())}


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


VALUE_OPERATORS: Dict[str, ValueOperator] = {
    "gaussian": _gaussian,
    "multiplicative": _multiplicative,
    "poisson": _poisson,
    "quantize": _quantize,
    "missing": _missing,
    "outliers": _outliers,
    "clip": _clip,
}


def register_operator(name: str, func: ValueOperator) -> None:
    """Register a custom value operator ``func(values, rng, params) -> (values, info)``."""
    if name in VALUE_OPERATORS:
        raise ValueError(f"Operator '{name}' is already registered")
    VALUE_OPERATORS[name] = func


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
        if op["type"] not in VALUE_OPERATORS:
            raise ValueError(f"observation.operators[{i}]: unknown type '{op['type']}'. "
                             f"Available: {sorted(VALUE_OPERATORS)}")
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
        targets = [name for name, info in results.items() if info["descriptor"].role == Role.LATENT]
        if not targets:
            raise ValueError("observation.targets not given and no output has role 'latent'")
    for name in targets:
        if name not in results:
            raise KeyError(f"observation target '{name}' is not an output of the model")
        data = results[name]["data"]
        if not (isinstance(data, np.ndarray) and data.ndim == 1 and np.issubdtype(data.dtype, np.number)):
            raise TypeError(f"observation target '{name}' must be a 1-D numeric array")

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
        values = np.asarray(results[t]["data"], dtype=float)
        if indices is not None:
            values = values[indices]
        steps: List[Info] = []
        for op in obs.get("operators") or []:
            params = {k: v for k, v in op.items() if k != "type"}
            values, info = VALUE_OPERATORS[op["type"]](values, rng, params)
            steps.append({"type": op["type"], **info})
        desc = results[t]["descriptor"]
        new_name = f"{t}{suffix}"
        added[new_name] = {
            "data": values,
            "descriptor": desc.replace(name=new_name, shape=values.shape, role=Role.OBSERVATION,
                                       plot_type="scatter",
                                       x_axis=coordinate_map.get(desc.x_axis, desc.x_axis)),
        }
        realized["targets"][t] = {"observed_as": new_name, "operators": steps}
    return added, copy.deepcopy(realized)
