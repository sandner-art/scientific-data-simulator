# simulator/doe.py
"""Design of experiments: design tables, seeded replicates and design runs.

A *design table* is a DataFrame with one row per run. Columns are config keys;
dotted keys address nested config entries (``observation.operators.0.snr_db``).
:func:`run_design` adds ``replicate`` and ``seed`` columns, runs every row
through :class:`SimulatorEngine` and returns the table joined with run IDs and
scalar outputs (the "tidy simulation" pattern: one row per run).
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union

import numpy as np
import pandas as pd

from .base import ExperimentLogic
from .config import apply_overrides, get_by_path, set_by_path, to_python as _to_python  # noqa: F401 (re-exported)
from .bundle import public_question
from .engine import SimulatorEngine
from .rng import derive_seeds, new_seed, validate_seed

logger = logging.getLogger(__name__)

DESIGN_TYPES = ("full_factorial", "lhs", "sobol", "halton", "random")


# ---------------------------------------------------------------------------- design tables
def generate_parameter_combinations(param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """All combinations (full factorial) of the given parameter levels."""
    if not param_ranges:
        raise ValueError("param_ranges cannot be empty.")
    keys = list(param_ranges.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*param_ranges.values())]


def _parse_bounds(name: str, spec: Any):
    """``[low, high]`` or ``{"low": .., "high": .., "scale": "linear"|"log", "integer": bool}``."""
    if isinstance(spec, Mapping):
        low, high = spec["low"], spec["high"]
        scale = spec.get("scale", "linear")
        integer = bool(spec.get("integer", False))
    elif isinstance(spec, Sequence) and len(spec) == 2:
        (low, high), scale, integer = spec, "linear", False
    else:
        raise ValueError(f"Parameter '{name}': continuous designs need [low, high] or "
                         f"{{low, high, scale, integer}}, got {spec!r}")
    if not low < high:
        raise ValueError(f"Parameter '{name}': low must be < high")
    if integer and (float(low) != int(low) or float(high) != int(high)):
        raise ValueError(f"Parameter '{name}': integer parameters need integer bounds")
    if scale not in ("linear", "log"):
        raise ValueError(f"Parameter '{name}': scale must be 'linear' or 'log'")
    if integer and scale == "log":
        raise ValueError(f"Parameter '{name}': integer parameters must use a linear scale")
    if scale == "log" and low <= 0:
        raise ValueError(f"Parameter '{name}': log scale needs low > 0")
    return float(low), float(high), scale, integer


def create_doe_table(param_ranges: Dict[str, Any], design_type: str = 'full_factorial',
                     n_samples: Optional[int] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Creates a design-of-experiments table.

    Args:
        param_ranges: For ``full_factorial``: name -> list of levels.
            For ``lhs``, ``sobol``, ``halton`` and ``random``: name -> ``[low, high]``
            or ``{"low", "high", "scale": "linear"|"log", "integer": bool}``.
        design_type: One of :data:`DESIGN_TYPES`.
        n_samples: Number of rows for sampled designs (powers of two are
            recommended for ``sobol``).
        seed: Seed for sampled designs (scrambling / sampling). Required for a
            reproducible table; recorded by :func:`run_design`.
    """
    if not param_ranges:
        raise ValueError("param_ranges cannot be empty for DOE table creation.")
    if design_type == 'full_factorial':
        return pd.DataFrame(generate_parameter_combinations(param_ranges))
    if design_type not in DESIGN_TYPES:
        raise ValueError(f"Unsupported design_type: {design_type}. Use one of {DESIGN_TYPES}.")
    if not n_samples or n_samples < 1:
        raise ValueError(f"design_type '{design_type}' requires n_samples >= 1")

    from scipy.stats import qmc

    names = list(param_ranges)
    bounds = [_parse_bounds(n, param_ranges[n]) for n in names]
    d = len(names)
    rng = np.random.default_rng(seed)
    if design_type == "lhs":
        unit = qmc.LatinHypercube(d=d, seed=rng).random(n_samples)
    elif design_type == "sobol":
        unit = qmc.Sobol(d=d, scramble=True, seed=rng).random(n_samples)
    elif design_type == "halton":
        unit = qmc.Halton(d=d, scramble=True, seed=rng).random(n_samples)
    else:  # random
        unit = rng.random((n_samples, d))

    columns = {}
    for j, (name, (low, high, scale, integer)) in enumerate(zip(names, bounds)):
        u = unit[:, j]
        if integer:  # uniform over the integers low..high (inclusive)
            values = np.minimum(np.floor(low + u * (high - low + 1)), high).astype(int)
        elif scale == "log":
            values = np.exp(np.log(low) + u * (np.log(high) - np.log(low)))
        else:
            values = low + u * (high - low)
        columns[name] = values
    return pd.DataFrame(columns)


def add_replicates_and_seeds(doe_table: pd.DataFrame, replicates: int = 1,
                             base_seed: Optional[int] = None,
                             seed_mode: str = "per_row") -> pd.DataFrame:
    """Repeat each design point ``replicates`` times and assign seeds.

    ``seed_mode``:
      * ``per_row``   - every run gets its own seed derived from ``base_seed``.
      * ``per_replicate`` - all design points share the seed of their replicate
        index (common random numbers: differences between design points are
        then not confounded with different random draws).
    """
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    if seed_mode not in ("per_row", "per_replicate"):
        raise ValueError("seed_mode must be 'per_row' or 'per_replicate'")
    base_seed = new_seed() if base_seed is None else validate_seed(base_seed)
    n_points = len(doe_table)
    table = doe_table.loc[doe_table.index.repeat(replicates)].reset_index(drop=True)
    table.insert(0, "design_point", np.repeat(np.arange(n_points), replicates))
    table.insert(1, "replicate", np.tile(np.arange(replicates), n_points))
    if seed_mode == "per_row":
        seeds = derive_seeds(base_seed, len(table))
    else:
        rep_seeds = derive_seeds(base_seed, replicates)
        seeds = [rep_seeds[r] for r in table["replicate"]]
    table["seed"] = np.array(seeds, dtype=np.uint64).astype(object)  # exact Python ints
    table.attrs["base_seed"] = base_seed
    table.attrs["seed_mode"] = seed_mode
    return table


# ---------------------------------------------------------------------------- running designs
@dataclass
class DesignRun:
    """Result of :func:`run_design`."""
    table: pd.DataFrame                       # design + replicate + seed + experiment_id + scalar outputs
    results: List[Optional[Dict[str, Dict[str, Any]]]] = field(default_factory=list)
    output_dir: str = ""
    base_seed: Optional[int] = None


_DESIGN_META_COLUMNS = ("design_point", "replicate", "seed")


def _row_overrides(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items() if k not in _DESIGN_META_COLUMNS}


def _run_one(args):
    """Worker for one design row (module-level so it can run in a subprocess)."""
    output_dir, config, cls, generate_plots, on_error = args
    engine = SimulatorEngine(output_dir=output_dir, verbose=False)
    try:
        run = engine.run_config(config, experiment_logic_class=cls, generate_plots=generate_plots)
        return {"status": "ok", "experiment_id": run.experiment_id,
                "experiment_dir": run.experiment_dir, "results": run.results, "error": None}
    except Exception as e:
        if on_error == "raise":
            raise
        return {"status": "failed", "experiment_id": None, "experiment_dir": None,
                "results": None, "error": f"{type(e).__name__}: {e}"}


def _scalar_outputs(results: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, info in (results or {}).items():
        value = info.get("data")
        if isinstance(value, (int, float, str, np.generic)) and not isinstance(value, bool):
            out[name] = _to_python(value)
    return out


def run_design(base_config: Dict[str, Any], doe_table: pd.DataFrame,
               output_dir: str = "experiments_output",
               experiment_logic_class: Optional[Type[ExperimentLogic]] = None,
               replicates: int = 1, base_seed: Optional[int] = None, seed_mode: str = "per_row",
               generate_plots: Optional[bool] = None, n_jobs: int = 1,
               on_error: str = "raise") -> DesignRun:
    """
    Run every row of a design table through the engine.

    If ``doe_table`` has no ``seed`` column, replicates and seeds are added with
    :func:`add_replicates_and_seeds` (``base_seed`` defaults to
    ``base_config['seed']`` or a fresh, recorded seed). The design is written to
    ``<output_dir>/design.csv`` and ``design_manifest.json`` before running, and
    the joined table to ``design_results.csv`` afterwards.

    Args:
        n_jobs: >1 runs rows in worker processes. The ExperimentLogic class must
            be importable by its module path, and a calling script must guard its
            entry point with ``if __name__ == "__main__":``.
        on_error: 'raise' (default) or 'continue' (failed rows get status
            'failed' and an error message in the table).
    """
    if on_error not in ("raise", "continue"):
        raise ValueError("on_error must be 'raise' or 'continue'")
    os.makedirs(output_dir, exist_ok=True)
    if "seed" in doe_table.columns:
        table = doe_table.reset_index(drop=True).copy()
        base_seed = doe_table.attrs.get("base_seed", base_seed)
    else:
        if base_seed is None and base_config.get("seed") is not None:
            base_seed = base_config["seed"]
        table = add_replicates_and_seeds(doe_table, replicates, base_seed, seed_mode)
        base_seed = table.attrs["base_seed"]

    table.to_csv(os.path.join(output_dir, "design.csv"), index=False)
    with open(os.path.join(output_dir, "design_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"base_seed": base_seed, "seed_mode": table.attrs.get("seed_mode", seed_mode),
                   "n_runs": len(table), "replicates": replicates,
                   "experiment_logic_class": (f"{experiment_logic_class.__module__}."
                                              f"{experiment_logic_class.__qualname__}"
                                              if experiment_logic_class else base_config.get("experiment_type")),
                   "base_config": base_config, "design_columns": list(table.columns)},
                  f, indent=2, default=str)

    jobs = []
    # to_dict("records") keeps each column's dtype; iterrows() would upcast
    # mixed rows to float64 and silently round 63-bit seeds.
    for row in table.to_dict("records"):
        config = apply_overrides(base_config, _row_overrides(row))
        config["seed"] = int(row["seed"])
        jobs.append((output_dir, config, experiment_logic_class, generate_plots, on_error))

    if n_jobs and n_jobs > 1:
        # "spawn" behaves the same on Linux, macOS and Windows and avoids
        # fork-after-threads deadlocks.
        import multiprocessing
        from concurrent.futures.process import BrokenProcessPool
        try:
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
                outcomes = list(pool.map(_run_one, jobs))
        except BrokenProcessPool as e:
            raise RuntimeError(
                "A worker process failed to start. When using n_jobs > 1 from a script, "
                "put the calling code under `if __name__ == \"__main__\":` (worker processes "
                "re-import the main module), and make sure the ExperimentLogic class is "
                "importable by its module path.") from e
    else:
        outcomes = [_run_one(job) for job in jobs]

    table["experiment_id"] = [o["experiment_id"] for o in outcomes]
    table["status"] = [o["status"] for o in outcomes]
    if any(o["error"] for o in outcomes):
        table["error"] = [o["error"] for o in outcomes]
    scalars = pd.DataFrame([_scalar_outputs(o["results"]) for o in outcomes], index=table.index)
    if not scalars.empty:
        table = pd.concat([table, scalars.add_prefix("out.")], axis=1)
    table.to_csv(os.path.join(output_dir, "design_results.csv"), index=False)
    _write_question_index(output_dir, table, outcomes)
    logger.info("Design finished: %d runs (%d failed) in %s", len(table),
                int((table["status"] != "ok").sum()), output_dir)
    return DesignRun(table=table, results=[o["results"] for o in outcomes],
                     output_dir=output_dir, base_seed=base_seed)


def _write_question_index(output_dir: str, table: pd.DataFrame, outcomes: List[Dict[str, Any]]) -> None:
    """If runs produced bundles, collect their questions into one benchmark:
    ``questions.jsonl`` (public, no answers) and ``qa.jsonl`` (private, with answers),
    one line per question, keyed by experiment_id and question id."""
    public_lines, private_lines = [], []
    for row, outcome in zip(table.to_dict("records"), outcomes):
        run_dir = outcome.get("experiment_dir")
        qa_path = os.path.join(run_dir, "bundle", "qa.json") if run_dir else None
        if not qa_path or not os.path.exists(qa_path):
            continue
        with open(qa_path, encoding="utf-8") as f:
            items = json.load(f)
        figure = None
        manifest_path = os.path.join(run_dir, "bundle", "bundle.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                fig = json.load(f).get("figure")
            figure = (os.path.relpath(os.path.join(run_dir, "bundle", fig["output"]), output_dir)
                      .replace(os.sep, "/") if fig else None)  # portable: written into the manifest
        for item in items:
            base = {"experiment_id": outcome["experiment_id"], "design_point": row.get("design_point"),
                    "replicate": row.get("replicate"), "figure": figure}
            private_lines.append({**base, **item})
            public_lines.append({**base, **public_question(item)})
    if not private_lines:
        return
    for name, lines in (("questions.jsonl", public_lines), ("qa.jsonl", private_lines)):
        with open(os.path.join(output_dir, name), "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, default=_to_python) + "\n")


def run_parameter_sweep(experiment_logic_class: Type[ExperimentLogic], base_config: Dict[str, Any],
                        param_ranges: Dict[str, List[Any]],
                        output_dir: str = "experiments_output",
                        output_transform: str = 'list',
                        generate_plots: Optional[bool] = None,
                        replicates: int = 1,
                        base_seed: Optional[int] = None,
                        n_jobs: int = 1) -> Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Full-factorial parameter sweep (convenience wrapper around :func:`run_design`).

    Returns:
        If ``output_transform == 'list'``: a list of dicts with keys
        'params', 'results', 'record_id' and 'seed' (one per run, in design order).
        If ``'nested'``: a dict mapping "k1=v1, k2=v2" to the results dict
        (only meaningful with ``replicates == 1``).
    """
    if not (isinstance(experiment_logic_class, type) and issubclass(experiment_logic_class, ExperimentLogic)):
        raise TypeError("experiment_logic_class must be a subclass of ExperimentLogic")
    if output_transform not in ('list', 'nested'):
        raise ValueError("Invalid output_transform value. Must be 'list' or 'nested'.")
    if not param_ranges:
        raise ValueError("param_ranges cannot be empty.")

    design = run_design(base_config, create_doe_table(param_ranges), output_dir=output_dir,
                        experiment_logic_class=experiment_logic_class, replicates=replicates,
                        base_seed=base_seed, generate_plots=generate_plots, n_jobs=n_jobs)
    names = list(param_ranges)
    rows = design.table.to_dict("records")
    if output_transform == 'list':
        return [{'params': {k: _to_python(row[k]) for k in names}, 'results': res,
                 'record_id': row['experiment_id'], 'seed': int(row['seed'])}
                for row, res in zip(rows, design.results)]
    nested: Dict[str, Dict[str, Any]] = {}
    for row, res in zip(rows, design.results):
        nested[", ".join(f"{k}={_to_python(row[k])}" for k in names)] = res
    return nested


def append_results_to_doe_table(doe_table: pd.DataFrame, results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Appends flattened simulation results to a DOE table (one row per run).

    ``results`` is the 'list' output of :func:`run_parameter_sweep`. Arrays and
    lists become ``<name>_<i>`` columns, DataFrames ``<name>_<col>_<i>``,
    dicts ``<name>_<key>``; scalars keep their name.
    """
    if not isinstance(doe_table, pd.DataFrame):
        raise TypeError("doe_table must be a pandas DataFrame")
    if not all('results' in r for r in results):
        raise ValueError("results must contain 'results' key, ensure output_transform='list'")
    if len(doe_table) != len(results):
        raise ValueError("The number of results must match the number of rows in the DOE table.")

    flattened_results = []
    for res in (r['results'] for r in results):
        flat_res: Dict[str, Any] = {}
        for data_name, data_info in res.items():
            data = data_info['data']
            if isinstance(data, (np.ndarray, list)):
                for i, val in enumerate(np.ravel(data) if isinstance(data, np.ndarray) else data):
                    flat_res[f"{data_name}_{i}"] = val
            elif isinstance(data, pd.DataFrame):
                for i, row in enumerate(data.to_dict(orient='records')):
                    for key, val in row.items():
                        flat_res[f"{data_name}_{key}_{i}"] = val
            elif isinstance(data, dict):
                for key, val in data.items():
                    flat_res[f"{data_name}_{key}"] = val
            else:
                flat_res[data_name] = data
        flattened_results.append(flat_res)

    results_df = pd.DataFrame(flattened_results)
    return pd.concat([doe_table.reset_index(drop=True), results_df], axis=1)
