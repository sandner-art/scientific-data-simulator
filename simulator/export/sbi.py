# simulator/export/sbi.py
"""Simulation-based inference (SBI) export.

SBI libraries (e.g. ``sbi``, BayesFlow) need two things: a prior over the
parameters theta and simulated data x for draws of theta. This module provides:

* :class:`BatchSimulator` - ``x = simulator(theta)`` for an ``(n, d)`` array of
  parameters, running the model in memory with independent, recorded seeds;
* :func:`generate_training_data` - draw theta from a box prior, simulate, and
  save ``theta.npy``/``x.npy`` plus a JSON description (prior, layout of x,
  seeds, base config) that makes the set reproducible;
* :func:`training_data_from_runs` - the same arrays from runs already on disk
  (e.g. a design), read back from their records;
* :func:`to_sbi_prior` - a ``sbi`` ``BoxUniform`` prior (needs torch + sbi).

``x`` concatenates the flattened observation outputs in a fixed order; its
layout is recorded as ``x_layout`` (name, start, length). Missing values stay
NaN: most SBI methods need them removed or imputed (``nan_policy``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..config import apply_overrides
from ..engine import simulate
from ..persistence import load_experiment_record
from ..rng import derive_seeds, new_seed, validate_seed
from ..utils import Role

Prior = Dict[str, Sequence[float]]


def _observation_names(results, observation: Optional[Sequence[str]]) -> List[str]:
    if observation:
        missing = [n for n in observation if n not in results]
        if missing:
            raise KeyError(f"Unknown observation outputs: {missing}")
        return list(observation)
    names = sorted(n for n, i in results.items() if i["descriptor"].role is Role.OBSERVATION)
    if not names:
        raise ValueError("No outputs with role 'observation'; configure observation: or pass observation=[...]")
    return names


def _flatten(results, names) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    parts, layout, start = [], [], 0
    for name in names:
        values = np.ravel(np.asarray(results[name]["data"], dtype=float))
        parts.append(values)
        layout.append({"name": name, "start": start, "length": int(values.size)})
        start += values.size
    return np.concatenate(parts) if parts else np.zeros(0), layout


def _apply_nan_policy(x: np.ndarray, policy: str) -> np.ndarray:
    if policy == "keep":
        return x
    if policy == "error":
        if np.isnan(x).any():
            raise ValueError("x contains NaN (missing observations); use nan_policy='keep' or 'zero'")
        return x
    if policy == "zero":
        return np.nan_to_num(x, nan=0.0)
    raise ValueError("nan_policy must be 'keep', 'error' or 'zero'")


class BatchSimulator:
    """Callable ``theta (n, d) -> x (n, k)`` over a base configuration.

    Each simulation gets its own seed, derived from ``seed`` and a call counter,
    so repeated calls give fresh noise and the whole sequence is reproducible.
    Seeds used are kept in :attr:`seeds`.
    """

    def __init__(self, base_config: Dict[str, Any], parameters: Sequence[str],
                 observation: Optional[Sequence[str]] = None, seed: Optional[int] = None,
                 nan_policy: str = "keep"):
        if not parameters:
            raise ValueError("parameters must name at least one config key")
        self.base_config = dict(base_config)
        self.base_config.pop("bundle", None)
        self.parameters = list(parameters)
        self.observation = list(observation) if observation else None
        self.seed = new_seed() if seed is None else validate_seed(seed)
        self.nan_policy = nan_policy
        self.seeds: List[int] = []
        self.x_layout: Optional[List[Dict[str, Any]]] = None
        self._calls = 0

    def simulate_one(self, theta: Sequence[float], seed: int) -> np.ndarray:
        values = dict(zip(self.parameters, (float(v) for v in theta)))
        config = apply_overrides(self.base_config, values)
        config["seed"] = int(seed)
        sim = simulate(config)
        names = _observation_names(sim.results, self.observation)
        x, layout = _flatten(sim.results, names)
        if self.x_layout is None:
            self.x_layout, self.observation = layout, names
        elif layout != self.x_layout:
            raise ValueError("Observation layout changed between simulations "
                             "(different lengths); sbi needs fixed-size x")
        return _apply_nan_policy(x, self.nan_policy)

    def __call__(self, theta: Any) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        if theta.shape[1] != len(self.parameters):
            raise ValueError(f"theta must have {len(self.parameters)} columns ({self.parameters})")
        seeds = derive_seeds(self.seed, self._calls + len(theta))[self._calls:]
        self._calls += len(theta)
        self.seeds.extend(seeds)
        return np.stack([self.simulate_one(row, s) for row, s in zip(theta, seeds)])


def sample_box_prior(prior: Prior, n: int, seed: Optional[int] = None) -> np.ndarray:
    """``n`` draws from independent uniform priors ``{name: [low, high]}`` (in dict order)."""
    rng = np.random.default_rng(seed)
    lows = np.array([float(b[0]) for b in prior.values()])
    highs = np.array([float(b[1]) for b in prior.values()])
    if np.any(lows >= highs):
        raise ValueError("prior bounds must satisfy low < high")
    return lows + rng.random((int(n), len(prior))) * (highs - lows)


def generate_training_data(base_config: Dict[str, Any], prior: Prior, n: int, seed: Optional[int] = None,
                           observation: Optional[Sequence[str]] = None, output_dir: Optional[str] = None,
                           nan_policy: str = "keep") -> Dict[str, Any]:
    """Draw ``n`` parameter sets from a box prior, simulate them, optionally save.

    Returns a dict with ``theta``, ``x``, ``parameter_names``, ``x_layout``,
    ``seeds`` and ``seed``. With ``output_dir``, writes ``theta.npy``, ``x.npy``
    and ``sbi_dataset.json``.
    """
    seed = new_seed() if seed is None else validate_seed(seed)
    prior_seed, sim_seed = derive_seeds(seed, 2)
    theta = sample_box_prior(prior, n, prior_seed)
    simulator = BatchSimulator(base_config, list(prior), observation, sim_seed, nan_policy)
    x = simulator(theta)
    data = {"theta": theta, "x": x, "parameter_names": list(prior), "x_layout": simulator.x_layout,
            "seeds": simulator.seeds, "seed": seed}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "theta.npy"), theta, allow_pickle=False)
        np.save(os.path.join(output_dir, "x.npy"), x, allow_pickle=False)
        with open(os.path.join(output_dir, "sbi_dataset.json"), "w", encoding="utf-8") as f:
            json.dump({"prior": {"type": "box_uniform", "bounds": {k: list(map(float, v)) for k, v in prior.items()}},
                       "parameter_names": list(prior), "x_layout": simulator.x_layout, "n": int(n),
                       "seed": seed, "simulation_seeds": simulator.seeds, "nan_policy": nan_policy,
                       "base_config": simulator.base_config}, f, indent=2, default=str)
    return data


def load_training_data(directory: str) -> Dict[str, Any]:
    with open(os.path.join(directory, "sbi_dataset.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["theta"] = np.load(os.path.join(directory, "theta.npy"))
    meta["x"] = np.load(os.path.join(directory, "x.npy"))
    return meta


def training_data_from_runs(output_dir: str, experiment_ids: Sequence[str], parameters: Sequence[str],
                            observation: Optional[Sequence[str]] = None,
                            nan_policy: str = "keep") -> Dict[str, Any]:
    """``theta``/``x`` from runs already on disk (e.g. the rows of a design).

    theta comes from each run's ground-truth parameters (so a law shift is
    respected), x from its observation outputs.
    """
    thetas, xs, layout, names = [], [], None, None
    for experiment_id in experiment_ids:
        record = load_experiment_record(output_dir, experiment_id)
        params = record.ground_truth.get("parameters") or record.config
        thetas.append([float(params[p]) for p in parameters])
        names = names or _observation_names(record.output_data, observation)
        x, this_layout = _flatten(record.output_data, names)
        if layout is not None and this_layout != layout:
            raise ValueError("Runs have different observation layouts; sbi needs fixed-size x")
        layout = this_layout
        xs.append(_apply_nan_policy(x, nan_policy))
    return {"theta": np.array(thetas), "x": np.stack(xs), "parameter_names": list(parameters),
            "x_layout": layout, "experiment_ids": list(experiment_ids)}


def to_sbi_prior(prior: Prior):
    """``sbi.utils.BoxUniform`` for a box prior (requires torch and sbi)."""
    try:
        import torch
        from sbi.utils import BoxUniform
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("to_sbi_prior needs 'torch' and 'sbi' (pip install sbi)") from e
    low = torch.tensor([float(b[0]) for b in prior.values()])
    high = torch.tensor([float(b[1]) for b in prior.values()])
    return BoxUniform(low=low, high=high)
