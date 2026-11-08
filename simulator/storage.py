# simulator/storage.py
"""Data files for experiment outputs, with SHA-256 checksums.

Arrays and DataFrames are written next to ``experiment_record.json`` instead of
being inlined in it; the record keeps a manifest entry per output::

    "storage": {"path": "data/observation/y_observed.npy", "format": "npy",
                "sha256": "...", "bytes": 16128}

Files are grouped by role so that observations can be distributed without the
ground truth::

    data/ground_truth/   parameter, latent, label
    data/observation/    observation
    data/coordinate/     coordinate
    data/auxiliary/      everything else

Configuration (``storage:`` in the experiment config)::

    storage:
      format: npy        # npy (default) | hdf5 (single data.h5, needs h5py) | inline (old JSON-only layout)
      dataframes: auto   # auto (parquet if pyarrow is installed, else csv) | parquet | csv

Scalars, strings and lists stay inline in the JSON record.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .provenance import file_sha256
from .utils import GROUND_TRUTH_ROLES, DataDescriptor, Role

FORMATS = ("npy", "hdf5", "inline")
DATAFRAME_FORMATS = ("auto", "parquet", "csv")


def role_directory(role: Optional[Role]) -> str:
    if role in GROUND_TRUTH_ROLES:
        return "ground_truth"
    if role is Role.OBSERVATION:
        return "observation"
    if role is Role.COORDINATE:
        return "coordinate"
    return "auxiliary"


def resolve_storage_config(storage: Optional[Dict[str, Any]]) -> Dict[str, str]:
    storage = dict(storage or {})
    unknown = set(storage) - {"format", "dataframes"}
    if unknown:
        raise ValueError(f"Unknown storage keys: {sorted(unknown)}")
    fmt = storage.get("format", "npy")
    if fmt not in FORMATS:
        raise ValueError(f"storage.format must be one of {FORMATS}")
    dfs = storage.get("dataframes", "auto")
    if dfs not in DATAFRAME_FORMATS:
        raise ValueError(f"storage.dataframes must be one of {DATAFRAME_FORMATS}")
    if dfs == "auto":
        try:
            import pyarrow  # noqa: F401
            dfs = "parquet"
        except ImportError:
            dfs = "csv"
    if fmt == "hdf5":
        try:
            import h5py  # noqa: F401
        except ImportError:
            raise ImportError("storage.format 'hdf5' needs h5py: pip install 'scientific-data-simulator[hdf5]'") from None
    return {"format": fmt, "dataframes": dfs}


def _safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


class _FileNames:
    """Unique file names per run. Sanitising can map different output names to
    the same file ("a b" and "a_b"), and some filesystems ignore case, so
    collisions are resolved deterministically with a numeric suffix."""

    def __init__(self):
        self._used = set()

    def __call__(self, sub: str, name: str, ext: str) -> str:
        stem, n = _safe_name(name), 1
        while True:
            candidate = os.path.join(sub, f"{stem}.{ext}" if n == 1 else f"{stem}__{n}.{ext}")
            if candidate.lower() not in self._used:
                self._used.add(candidate.lower())
                return candidate
            n += 1


def _hdf5_storable(arr: np.ndarray) -> bool:
    return arr.dtype.kind not in "OUSV"  # objects, unicode/bytes strings, void: keep inline


def _entry(experiment_dir: str, rel_path: str, fmt: str, **extra: Any) -> Dict[str, Any]:
    full = os.path.join(experiment_dir, rel_path)
    return {"path": rel_path.replace(os.sep, "/"), "format": fmt,
            "sha256": file_sha256(full), "bytes": os.path.getsize(full), **extra}


def write_outputs(output_data: Dict[str, Dict[str, Any]], experiment_dir: str,
                  storage: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[Dict[str, Any]]]:
    """Write array/DataFrame outputs to files. Returns ``{name: manifest entry or None}``
    (None = keep the value inline in the JSON record)."""
    cfg = resolve_storage_config(storage)
    manifest: Dict[str, Optional[Dict[str, Any]]] = {name: None for name in output_data}
    if cfg["format"] == "inline":
        return manifest

    arrays = {n: i for n, i in output_data.items() if isinstance(i["data"], np.ndarray)}
    frames = {n: i for n, i in output_data.items() if isinstance(i["data"], pd.DataFrame)}
    file_name = _FileNames()

    for name, info in frames.items():
        sub = os.path.join("data", role_directory(info["descriptor"].role))
        os.makedirs(os.path.join(experiment_dir, sub), exist_ok=True)
        ext = cfg["dataframes"]
        rel = file_name(sub, name, ext)
        df = info["data"]
        # keep a meaningful index; drop the default 0..n-1 RangeIndex
        keep_index = not (isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1)
        if ext == "parquet":
            df.to_parquet(os.path.join(experiment_dir, rel), index=keep_index)
        else:
            df.to_csv(os.path.join(experiment_dir, rel), index=keep_index)
        manifest[name] = _entry(experiment_dir, rel, ext, index=keep_index)

    if cfg["format"] == "npy":
        for name, info in arrays.items():
            if info["data"].dtype == object:
                continue  # not representable without pickle: keep inline
            sub = os.path.join("data", role_directory(info["descriptor"].role))
            os.makedirs(os.path.join(experiment_dir, sub), exist_ok=True)
            rel = file_name(sub, name, "npy")
            np.save(os.path.join(experiment_dir, rel), info["data"], allow_pickle=False)
            manifest[name] = _entry(experiment_dir, rel, "npy")
    elif any(_hdf5_storable(i["data"]) for i in arrays.values()):  # hdf5: one file, one group per role
        import h5py
        rel = "data.h5"
        datasets = {}
        with h5py.File(os.path.join(experiment_dir, rel), "w") as h5:
            for name, info in arrays.items():
                if not _hdf5_storable(info["data"]):
                    continue  # strings/objects stay inline in the JSON record
                dataset = f"{role_directory(info['descriptor'].role)}/{name.replace('/', '_')}"
                ds = h5.create_dataset(dataset, data=info["data"])
                ds.attrs["descriptor"] = json.dumps(info["descriptor"].to_dict())
                datasets[name] = dataset
        entry = _entry(experiment_dir, rel, "hdf5")
        for name, dataset in datasets.items():
            manifest[name] = {**entry, "dataset": dataset}
    return manifest


class ChecksumError(ValueError):
    """A data file does not match the checksum recorded in its record."""


def read_output(experiment_dir: str, entry: Dict[str, Any], descriptor: DataDescriptor,
                verify: bool = True, mmap: bool = False) -> Any:
    path = os.path.join(experiment_dir, entry["path"])
    if verify:
        actual = file_sha256(path)
        if actual != entry["sha256"]:
            raise ChecksumError(f"Checksum mismatch for '{descriptor.name}' ({entry['path']}): "
                                f"expected {entry['sha256']}, got {actual}")
    fmt = entry["format"]
    if fmt == "npy":
        return np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path, index_col=0 if entry.get("index") else None)
    if fmt == "hdf5":
        import h5py
        with h5py.File(path, "r") as h5:
            return h5[entry["dataset"]][()]
    raise ValueError(f"Unknown storage format '{fmt}'")
