# simulator/export/the_well.py
"""Export field simulations in the format of The Well (PolymathicAI).

The Well (Ohana et al., NeurIPS 2024 Datasets and Benchmarks) stores each
dataset as self-describing HDF5 files that its ``the_well`` package reads with
``the_well.data.WellDataset``. :func:`export_the_well` writes runs of a field
model (for example :class:`experiments.heat.logic.HeatExperiment`) in that
layout::

    <output_dir>/
      data/train/<dataset>[_<param>_<value>...].hdf5   one file per parameter set
      data/valid/...                                   (only the splits that get runs)
      data/test/...
      stats.yaml          mean, std, mean_delta, std_delta, rms, rms_delta per field
                          (from the train split; read with use_normalization=True)
      README.md           dataset card
      sds_export.json     provenance: file -> trajectory -> experiment id, seed, fidelity

The schema follows the loader (``the_well/data/datasets.py``) and the package's
reference writer (``the_well/utils/dummy_data.py``):

* file attributes ``dataset_name``, ``grid_type`` (``cartesian``), ``n_spatial_dims``,
  ``n_trajectories``, ``simulation_parameters`` (names) and one attribute per
  simulation parameter with its value;
* group ``dimensions`` (attribute ``spatial_dims``) with datasets ``time`` and one
  per spatial dimension (``x``, ``y``...), each with ``sample_varying`` and
  ``time_varying`` attributes;
* group ``boundary_conditions`` with one subgroup per axis (``x_periodic``...):
  attributes ``associated_dims``, ``associated_fields``, ``bc_type`` (``PERIODIC``,
  ``WALL`` or ``OPEN``), ``sample_varying``, ``time_varying`` and a boolean ``mask``
  over the axis that marks the boundary points (first and last index);
* group ``scalars`` (attribute ``field_names``): the simulation parameters as
  constant scalars (``time_varying = sample_varying = False``);
* groups ``t0_fields``, ``t1_fields``, ``t2_fields`` (attribute ``field_names``);
  scalar fields go to ``t0_fields`` with shape ``(n_trajectories, n_time, *spatial)``
  and attributes ``dim_varying`` (one ``True`` per spatial axis),
  ``sample_varying = True`` and ``time_varying = True``.

As in The Well, a file holds the trajectories of one parameter set (the runs
differ in their initial conditions, i.e. their seeds) and the parameters that
vary between files are its ``simulation_parameters``. Each run becomes one
trajectory. Spatial grids must agree across all runs; time grids within a file.

Unlike the Hugging Face export, nothing is withheld: The Well is for training
surrogate models, so the files contain the exact (latent) fields. The fidelity of
every run (``reference``/``audit``...) is recorded in ``sds_export.json``, the file
attribute ``sds_fidelity`` and the README; audit runs are flagged. Output is
deterministic: the same runs give byte-identical files.

Writing needs ``h5py`` and ``pyyaml``; reading needs ``the_well`` (and PyTorch)::

    from the_well.data import WellDataset
    ds = WellDataset(path="<output_dir>", well_split_name="train", n_steps_input=4, n_steps_output=1)
    sample = ds[0]   # input_fields (4, nx[, ny], n_fields), output_fields, constant_scalars, ...

``WellDataModule`` accepts only dataset names of the official collection, so use
``WellDataset`` with ``path=`` for exported data.
"""
from __future__ import annotations

import json
import math
import numbers
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..utils import Role
from .huggingface import Source, _fidelity, _load_runs, _model, _Run

SPLITS = ("train", "valid", "test")
BC_TYPES = ("PERIODIC", "WALL", "OPEN")
#: Files written by the exporter; ``overwrite=True`` replaces only these.
MANAGED_PATHS = ("data", "stats.yaml", "README.md", "sds_export.json")


@dataclass
class WellExport:
    """Summary of an export."""
    output_dir: str
    dataset_name: str
    files: Dict[str, List[str]] = field(default_factory=dict)          # split -> relative file paths
    n_trajectories: Dict[str, int] = field(default_factory=dict)       # split -> trajectories
    simulation_parameters: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    spatial_dims: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- helpers
def _require_h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError("The Well export needs h5py: pip install 'scientific-data-simulator[hdf5]'") from None
    return h5py


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def _strings(values: Sequence[str]):
    h5py = _require_h5py()
    return np.array(list(values), dtype=h5py.string_dtype())


def _format_value(value: Any) -> str:
    text = f"{value:.10g}" if isinstance(value, float) else str(value)
    return "".join(c if (c.isalnum() or c in "-.") else "_" for c in text)


def _inferable(model: Optional[str]) -> Tuple[str, ...]:
    if not model:
        return ()
    try:
        from ..engine import resolve_experiment_type
        return tuple(getattr(resolve_experiment_type({"experiment_type": model}), "inferable_parameters", ()) or ())
    except Exception:  # model not importable here: fall back to the varying parameters
        return ()


def _output(run: _Run, name: str) -> Tuple[np.ndarray, Any]:
    entry = run.record.output_data.get(name)
    if entry is None:
        raise KeyError(f"Run {run.experiment_id} has no output '{name}'")
    data = entry["data"]
    if not isinstance(data, np.ndarray) or not np.issubdtype(np.asarray(data).dtype, np.number):
        raise TypeError(f"Output '{name}' of run {run.experiment_id} is not a numeric array")
    return np.asarray(data), entry["descriptor"]


def _spatial_dims(run: _Run, field_name: str, time_coordinate: str) -> List[str]:
    data, _ = _output(run, field_name)
    n = data.ndim - 1
    coords = [name for name, info in run.record.output_data.items()
              if info["descriptor"].role is Role.COORDINATE and name != time_coordinate]
    preferred = [d for d in ("x", "y", "z") if d in coords]
    if len(preferred) >= n:
        return preferred[:n]
    raise ValueError(f"Cannot infer the {n} spatial coordinate(s) of '{field_name}' from coordinate outputs "
                     f"{coords}; pass spatial_dims=[...]")


def _check_splits(names) -> None:
    unknown = set(names) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown split(s) {sorted(unknown)}; The Well uses {SPLITS}")


def _assign_splits(source: Optional[Source], split: str,
                   splits: Optional[Mapping[str, Any]]) -> Tuple[List[Tuple[str, _Run]], Optional[Dict[str, float]]]:
    """``[(split, run)]`` and, for fractional splits, the fractions (applied per parameter set later)."""
    if splits is None:
        if source is None:
            raise ValueError("give source (with split=...) or splits={...}")
        _check_splits([split])
        return [(split, run) for run in _load_runs(source)], None
    _check_splits(splits)
    if splits and all(_is_number(v) for v in splits.values()):
        if source is None:
            raise ValueError("fractional splits need a source")
        fractions = {k: float(v) for k, v in splits.items()}
        if any(v < 0 for v in fractions.values()) or not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
            raise ValueError("split fractions must be >= 0 and sum to 1")
        return [("", run) for run in _load_runs(source)], fractions
    if source is not None:
        raise ValueError("give either source (with split=...) or splits={split: source}, not both")
    assigned: List[Tuple[str, _Run]] = []
    for name in SPLITS:
        if name in splits:
            assigned.extend((name, run) for run in _load_runs(splits[name]))
    seen: Dict[str, str] = {}
    for name, run in assigned:
        if run.experiment_id in seen:
            raise ValueError(f"Run {run.experiment_id} is in split '{seen[run.experiment_id]}' and '{name}'")
        seen[run.experiment_id] = name
    return assigned, None


def _parameter_names(runs: List[_Run], parameters: Optional[Sequence[str]]) -> List[str]:
    params = [r.ground_truth.get("parameters") or {} for r in runs]
    if parameters is not None:
        names = list(parameters)
        for name in names:
            for run, p in zip(runs, params):
                if not _is_number(p.get(name)):
                    raise ValueError(f"Parameter '{name}' of run {run.experiment_id} is not a number: {p.get(name)!r}")
        return names
    keys = sorted(set().union(*params)) if params else []
    names: List[str] = []
    inferable = set()
    for run in runs:
        inferable.update(_inferable(_model(run)))
    for key in keys:
        values = [p.get(key) for p in params]
        distinct = {json.dumps(v, sort_keys=True, default=str) for v in values}
        configured = all(key in (r.raw.get("config") or {}) for r in runs)  # skip derived values (dx, ...)
        if all(_is_number(v) for v in values):
            if key in inferable or (len(distinct) > 1 and configured):
                names.append(key)
        elif len(distinct) > 1 and all(isinstance(v, (str, bool, type(None))) for v in values):
            raise ValueError(f"Parameter '{key}' differs between runs ({sorted(distinct)}) and is not a number; "
                             "export such runs as separate datasets")
    return names


def _bc_types(boundary_conditions: Union[str, Mapping[str, str]], dims: List[str]) -> Dict[str, str]:
    if isinstance(boundary_conditions, str):
        types = {d: boundary_conditions for d in dims}
    else:
        types = dict(boundary_conditions)
        if set(types) != set(dims):
            raise ValueError(f"boundary_conditions must name every spatial dimension {dims}")
    types = {d: str(t).upper() for d, t in types.items()}
    bad = {t for t in types.values() if t not in BC_TYPES}
    if bad:
        raise ValueError(f"Unknown boundary condition type(s) {sorted(bad)}; use {BC_TYPES}")
    return types


# ---------------------------------------------------------------------------- writing
def _write_file(path: str, dataset_name: str, runs: List[_Run], fields: List[str], dims: List[str],
                coords: Dict[str, np.ndarray], time: np.ndarray, param_names: List[str],
                param_values: Tuple[float, ...], bcs: Dict[str, str], dtype: np.dtype,
                units: Dict[str, Optional[str]]) -> None:
    h5py = _require_h5py()
    opts = {"track_times": False}
    with h5py.File(path, "w", libver="earliest") as f:
        f.attrs["dataset_name"] = dataset_name
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = len(dims)
        f.attrs["n_trajectories"] = len(runs)
        f.attrs["simulation_parameters"] = _strings(param_names)
        for name, value in zip(param_names, param_values):
            f.attrs[name] = value
        f.attrs["sds_experiment_ids"] = _strings([r.experiment_id for r in runs])
        f.attrs["sds_seeds"] = np.array([int(r.ground_truth.get("seed") or 0) for r in runs], dtype=np.int64)
        f.attrs["sds_fidelity"] = _strings([_fidelity(r) or "unknown" for r in runs])

        g = f.create_group("dimensions")
        g.attrs["spatial_dims"] = _strings(dims)
        ds = g.create_dataset("time", data=time.astype(np.float64), **opts)
        ds.attrs["sample_varying"] = False
        ds.attrs["time_varying"] = True
        if units.get("time"):
            ds.attrs["units"] = units["time"]
        for d in dims:
            ds = g.create_dataset(d, data=coords[d].astype(np.float64), **opts)
            ds.attrs["sample_varying"] = False
            ds.attrs["time_varying"] = False
            if units.get(d):
                ds.attrs["units"] = units[d]

        g = f.create_group("boundary_conditions")
        for d in dims:
            sub = g.create_group(f"{d}_{bcs[d].lower()}")
            sub.attrs["associated_dims"] = _strings([d])
            sub.attrs["associated_fields"] = _strings([])
            sub.attrs["bc_type"] = bcs[d]
            sub.attrs["sample_varying"] = False
            sub.attrs["time_varying"] = False
            mask = np.zeros(len(coords[d]), dtype=bool)
            mask[0] = mask[-1] = True
            sub.create_dataset("mask", data=mask, **opts)

        g = f.create_group("scalars")
        g.attrs["field_names"] = _strings(param_names)
        for name, value in zip(param_names, param_values):
            ds = g.create_dataset(name, data=np.array(value, dtype=np.float64), **opts)
            ds.attrs["time_varying"] = False
            ds.attrs["sample_varying"] = False

        g = f.create_group("t0_fields")
        g.attrs["field_names"] = _strings(fields)
        for name in fields:
            data = np.stack([_output(r, name)[0] for r in runs]).astype(dtype)
            ds = g.create_dataset(name, data=data, **opts)
            ds.attrs["dim_varying"] = np.ones(len(dims), dtype=bool)
            ds.attrs["sample_varying"] = True
            ds.attrs["time_varying"] = True
            if units.get(name):
                ds.attrs["units"] = units[name]
        for order in (1, 2):
            f.create_group(f"t{order}_fields").attrs["field_names"] = _strings([])


def _field_stats(arrays: List[np.ndarray]) -> Dict[str, float]:
    """mean/std/rms of the values and of the one-step time differences."""
    values = np.concatenate([a.astype(np.float64).ravel() for a in arrays])
    deltas = [np.diff(a.astype(np.float64), axis=0).ravel() for a in arrays if a.shape[0] > 1]
    deltas = np.concatenate(deltas) if deltas else np.zeros(1)
    return {"mean": float(values.mean()), "std": float(values.std()), "rms": float(np.sqrt(np.mean(values ** 2))),
            "mean_delta": float(deltas.mean()), "std_delta": float(deltas.std()),
            "rms_delta": float(np.sqrt(np.mean(deltas ** 2)))}


def _readme(export: WellExport, entries: List[Dict[str, Any]], units: Dict[str, Optional[str]],
            bcs: Dict[str, str], shape: Tuple[int, ...], description: Optional[str]) -> str:
    models = sorted({e["model"] or "unknown" for e in entries})
    fidelity: Dict[str, int] = {}
    for e in entries:
        fidelity[e["fidelity"] or "unknown"] = fidelity.get(e["fidelity"] or "unknown", 0) + 1
    lines = [f"# {export.dataset_name}", "",
             description or "Simulated fields exported from Scientific Data Simulator in The Well format.", "",
             "## Contents", "",
             f"- Model(s): {', '.join(f'`{m}`' for m in models)}",
             f"- Fields (`t0_fields`): " + ", ".join(f"`{n}` [{units.get(n) or '-'}]" for n in export.fields),
             f"- Spatial dimensions: {', '.join(f'`{d}` ({bcs[d].lower()})' for d in export.spatial_dims)}; "
             f"resolution {' x '.join(str(s) for s in shape)}",
             f"- Simulation parameters (constant scalars): "
             + (", ".join(f"`{p}`" for p in export.simulation_parameters) or "none"),
             "- Trajectories per split: " + ", ".join(f"{s}: {n}" for s, n in export.n_trajectories.items()),
             "- Fidelity of the runs: " + ", ".join(f"{k}: {v}" for k, v in sorted(fidelity.items())), "",
             "The fields are the exact (latent) model outputs: nothing is withheld, as the dataset is meant "
             "for training and evaluating surrogate models. Each trajectory is one simulator run; "
             "`sds_export.json` maps every trajectory to its experiment id, seed and fidelity, so it can be "
             "regenerated from the simulator.", ""]
    if fidelity.get("audit"):
        lines += ["**Warning:** some runs are `audit` models (deliberately flawed, e.g. explicit finite "
                  "differences with discretisation error or instability). Do not use them as ground truth "
                  "for the physical system.", ""]
    lines += ["## Loading", "", "```python", "from the_well.data import WellDataset",
              "ds = WellDataset(path=\"<this directory>\", well_split_name=\"train\",",
              "                 n_steps_input=4, n_steps_output=1)",
              "sample = ds[0]  # input_fields, output_fields, constant_scalars, boundary_conditions, grids",
              "```", "",
              "`stats.yaml` holds per-field normalisation statistics computed from the train split "
              "(`use_normalization=True, normalization_type=ZScoreNormalization`).", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------- public API
def export_the_well(source: Optional[Source], output_dir: str, dataset_name: str,
                    fields: Sequence[str] = ("u",), split: str = "train",
                    splits: Optional[Mapping[str, Any]] = None,
                    parameters: Optional[Sequence[str]] = None,
                    spatial_dims: Optional[Sequence[str]] = None, time_coordinate: str = "time",
                    boundary_conditions: Union[str, Mapping[str, str]] = "PERIODIC",
                    dtype: str = "float32", description: Optional[str] = None,
                    overwrite: bool = False) -> WellExport:
    """Write runs of a field model as a dataset in The Well's HDF5 format.

    Args:
        source: A run directory, a design directory (of runs), or a list of them.
            ``None`` when ``splits`` maps split names to sources.
        output_dir: Dataset directory to create (``<output_dir>/data/<split>/*.hdf5``).
        dataset_name: Value of the ``dataset_name`` file attribute and file-name stem.
        fields: Outputs to export; each must have shape ``(n_time, *spatial)``.
        split: Split for all runs of ``source`` (``train``, ``valid`` or ``test``).
        splits: Either ``{split: source}`` (explicit sources per split) or
            ``{split: fraction}`` (fractions summing to 1, applied to the runs of
            each parameter set in their deterministic order: design point,
            replicate, run directory).
        parameters: Simulation parameters (numeric ground-truth parameters) stored
            as file attributes and constant scalars; runs are grouped into one file
            per parameter set. Default: the model's ``inferable_parameters`` plus
            every numeric configuration value that varies between the runs
            (derived parameters such as ``dx`` are not included).
        spatial_dims: Names of the spatial coordinate outputs, in axis order
            (default: ``x``, ``y``, ``z`` as present).
        boundary_conditions: ``PERIODIC``, ``WALL`` or ``OPEN`` for every axis, or a
            mapping per spatial dimension.
        dtype: Storage type of the fields (``float32`` as in The Well, or ``float64``).
        overwrite: Replace an existing export (only the files this exporter writes).
    """
    _require_h5py()
    import yaml

    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise ValueError("dtype must be a floating-point type")
    fields = list(fields)
    if not fields:
        raise ValueError("fields is empty")
    if any(os.path.exists(os.path.join(output_dir, p)) for p in MANAGED_PATHS):
        if not overwrite:
            raise FileExistsError(f"'{output_dir}' already contains an export; pass overwrite=True")
        for p in MANAGED_PATHS:
            target = os.path.join(output_dir, p)
            if os.path.isdir(target):
                shutil.rmtree(target)
            elif os.path.exists(target):
                os.remove(target)

    assigned, fractions = _assign_splits(source, split, splits)
    if not assigned:
        raise ValueError("no runs to export")
    runs = [run for _, run in assigned]

    # -- grid and fields (shared by every run)
    first = runs[0]
    dims = list(spatial_dims) if spatial_dims is not None else _spatial_dims(first, fields[0], time_coordinate)
    if not 1 <= len(dims) <= 3:
        raise ValueError("The Well supports 1 to 3 spatial dimensions")
    coords = {d: _output(first, d)[0] for d in dims}
    shape = tuple(len(coords[d]) for d in dims)
    units: Dict[str, Optional[str]] = {"time": _output(first, time_coordinate)[1].units}
    for d in dims:
        units[d] = _output(first, d)[1].units
    for name in fields:
        units[name] = _output(first, name)[1].units
    for run in runs:
        for d in dims:
            if not np.array_equal(_output(run, d)[0], coords[d]):
                raise ValueError(f"Run {run.experiment_id} has a different '{d}' grid; The Well needs one "
                                 "spatial grid per dataset")
        n_time = len(_output(run, time_coordinate)[0])
        for name in fields:
            data, desc = _output(run, name)
            if data.shape != (n_time,) + shape:
                raise ValueError(f"Output '{name}' of run {run.experiment_id} has shape {data.shape}, "
                                 f"expected {(n_time,) + shape} (time, {', '.join(dims)})")
            if desc.units != units[name]:
                raise ValueError(f"Output '{name}' has units {desc.units!r} in run {run.experiment_id} "
                                 f"and {units[name]!r} elsewhere")
    bcs = _bc_types(boundary_conditions, dims)

    # -- parameter sets -> files
    param_names = _parameter_names(runs, parameters)
    groups: Dict[Tuple[float, ...], List[Tuple[str, _Run]]] = {}
    for split_name, run in assigned:
        p = run.ground_truth.get("parameters") or {}
        key = tuple(float(p[n]) if not isinstance(p[n], numbers.Integral) else int(p[n]) for n in param_names)
        groups.setdefault(key, []).append((split_name, run))
    by_file: Dict[Tuple[str, Tuple[float, ...]], List[_Run]] = {}
    if fractions is not None and max(len(m) for m in groups.values()) == 1:
        # One trajectory per parameter set (e.g. every run draws its own parameters): split the
        # parameter sets instead, in a fixed pseudo-random order (not by value, which would make
        # the test split an extrapolation split).
        import hashlib
        keys = sorted(groups, key=lambda k: hashlib.sha256(repr(k).encode()).hexdigest())
        acc, start = 0.0, 0
        for name in SPLITS:
            acc += fractions.get(name, 0.0)
            end = int(round(acc * len(keys)))
            for key in keys[start:end]:
                by_file.setdefault((name, key), []).append(groups[key][0][1])
            start = max(start, end)
        groups = {}
    for key in sorted(groups):
        members = groups[key]
        if fractions is not None:
            n = len(members)
            bounds, acc = [], 0.0
            for name in SPLITS:
                acc += fractions.get(name, 0.0)
                bounds.append((name, int(round(acc * n))))
            start = 0
            for name, end in bounds:
                for _, run in members[start:end]:
                    by_file.setdefault((name, key), []).append(run)
                start = max(start, end)
        else:
            for name, run in members:
                by_file.setdefault((name, key), []).append(run)

    export = WellExport(os.path.abspath(output_dir), dataset_name, simulation_parameters=param_names,
                        fields=fields, spatial_dims=dims)
    entries: List[Dict[str, Any]] = []
    used_names: Dict[str, int] = {}  # "<split>/<stem>" -> files with that name
    train_arrays: Dict[str, List[np.ndarray]] = {n: [] for n in fields}
    all_arrays: Dict[str, List[np.ndarray]] = {n: [] for n in fields}
    for (split_name, key), members in sorted(by_file.items(), key=lambda kv: (SPLITS.index(kv[0][0]), kv[0][1])):
        time = _output(members[0], time_coordinate)[0]
        for run in members:
            if not np.array_equal(_output(run, time_coordinate)[0], time):
                raise ValueError(f"Runs {members[0].experiment_id} and {run.experiment_id} share the parameters "
                                 f"{dict(zip(param_names, key))} but not the time grid; add the parameter that "
                                 "differs (e.g. dt or n_steps) to parameters=[...]")
        stem = "_".join([dataset_name] + [f"{n}_{_format_value(v)}" for n, v in zip(param_names, key)])
        count = used_names.get(f"{split_name}/{stem}", 0)
        used_names[f"{split_name}/{stem}"] = count + 1
        file_name = f"{stem}.hdf5" if count == 0 else f"{stem}__{count + 1}.hdf5"
        rel = os.path.join("data", split_name, file_name)
        os.makedirs(os.path.join(output_dir, "data", split_name), exist_ok=True)
        _write_file(os.path.join(output_dir, rel), dataset_name, members, fields, dims, coords, time,
                    param_names, key, bcs, dtype, units)
        export.files.setdefault(split_name, []).append(rel.replace(os.sep, "/"))
        export.n_trajectories[split_name] = export.n_trajectories.get(split_name, 0) + len(members)
        for index, run in enumerate(members):
            entries.append({"split": split_name, "file": rel.replace(os.sep, "/"), "trajectory": index,
                            "experiment_id": run.experiment_id, "seed": run.ground_truth.get("seed"),
                            "model": _model(run), "fidelity": _fidelity(run),
                            "simulation_parameters": dict(zip(param_names, key))})
            for name in fields:
                arr = _output(run, name)[0].astype(dtype)
                all_arrays[name].append(arr)
                if split_name == "train":
                    train_arrays[name].append(arr)

    source_arrays = train_arrays if any(train_arrays[n] for n in fields) else all_arrays
    stats: Dict[str, Dict[str, float]] = {k: {} for k in ("mean", "std", "mean_delta", "std_delta", "rms", "rms_delta")}
    for name in fields:
        for k, v in _field_stats(source_arrays[name]).items():
            stats[k][name] = v
    with open(os.path.join(output_dir, "stats.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=True)

    with open(os.path.join(output_dir, "sds_export.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "the_well", "dataset_name": dataset_name, "fields": fields, "spatial_dims": dims,
                   "boundary_conditions": bcs, "simulation_parameters": param_names, "dtype": dtype.name,
                   "units": units,
                   "stats_from": "train" if source_arrays is train_arrays else "all splits",
                   "files": export.files, "trajectories": entries}, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(_readme(export, entries, units, bcs, shape, description))
    return export
