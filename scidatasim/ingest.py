"""Ingest: read external tables and series (real data) with a provenance record.

Real data enter the simulator in three places: as a background for :mod:`observation`'s
``real_noise`` operator (a simulated signal in real noise), as the outputs of the
``external`` experiment (real measurements put through the same questions, bundles and estimators
as simulated ones), and in validation reports that compare the two.

Formats: CSV/TSV/TXT (and ECSV: '#' header lines are skipped), Parquet (pyarrow), JSON/JSONL,
NPY/NPZ, HDF5 (h5py; ``key`` names the dataset or group), FITS (astropy; ``key`` names the HDU,
default the first table). Compressed CSV (``.gz``, ``.bz2``, ``.zip``, ``.xz``) is read by pandas.

Every read returns, with the data, a provenance record: the absolute path, SHA-256, size, format,
rows and columns read, and the time of reading. Runs that use external data store it, so a result
names the exact file it came from.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TEXT = (".csv", ".tsv", ".txt", ".dat", ".ecsv")
COMPRESSED = (".gz", ".bz2", ".zip", ".xz")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(path: str, fmt: str, **extra) -> Dict[str, Any]:
    path = os.path.abspath(path)
    return {"path": path, "sha256": sha256_file(path), "bytes": os.path.getsize(path), "format": fmt,
            "read_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}


def _format(path: str, fmt: Optional[str]) -> str:
    if fmt:
        return fmt.lower()
    name = path.lower()
    for ext in COMPRESSED:
        if name.endswith(ext):
            name = name[: -len(ext)]
    ext = os.path.splitext(name)[1]
    return {".csv": "csv", ".tsv": "tsv", ".txt": "csv", ".dat": "whitespace", ".ecsv": "ecsv",
            ".parquet": "parquet", ".pq": "parquet", ".json": "json", ".jsonl": "jsonl", ".npy": "npy",
            ".npz": "npz", ".h5": "hdf5", ".hdf5": "hdf5", ".hdf": "hdf5", ".fits": "fits", ".fit": "fits",
            ".fts": "fits"}.get(ext, "csv")


def _fits(path: str, key) -> pd.DataFrame:
    try:
        from astropy.io import fits
        from astropy.table import Table
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("reading FITS needs astropy: pip install 'scidatasim[astro]'") from exc
    with fits.open(path, memmap=False) as hdul:
        if key is None:
            key = next((i for i, h in enumerate(hdul) if isinstance(h, (fits.BinTableHDU, fits.TableHDU))), None)
            if key is None:
                key = next(i for i, h in enumerate(hdul) if h.data is not None)
        hdu = hdul[key]
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            table = Table(hdu.data)
            return pd.DataFrame({c: list(np.asarray(table[c])) if np.ndim(table[c]) > 1 else
                                 np.asarray(table[c]).astype(np.asarray(table[c]).dtype.newbyteorder("="))
                                 for c in table.colnames})
        data = np.asarray(hdu.data)
        return pd.DataFrame({"value": [data]}) if data.ndim > 1 else pd.DataFrame({"value": data})


def _hdf5(path: str, key) -> pd.DataFrame:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("reading HDF5 needs h5py: pip install h5py") from exc
    with h5py.File(path, "r") as f:
        obj = f[key] if key else f
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            if data.dtype.names:
                return pd.DataFrame({n: data[n] for n in data.dtype.names})
            return pd.DataFrame({"value": data}) if data.ndim == 1 else pd.DataFrame({"value": [data]})
        cols = {}
        for name, item in obj.items():  # a group: its 1-D datasets of equal length are columns
            if isinstance(item, h5py.Dataset) and item.ndim == 1:
                cols[name] = item[()]
        if not cols:
            raise ValueError(f"{path}: no 1-D datasets in group '{key or '/'}'; name one with key=")
        n = min(len(v) for v in cols.values())
        return pd.DataFrame({k: v[:n] for k, v in cols.items() if len(v) == n})


def read_table(path: str, fmt: Optional[str] = None, key=None, columns: Optional[Sequence[str]] = None,
               **kwargs) -> "Table":
    """Read a file into a :class:`Table` (a DataFrame with its provenance). ``columns`` keeps those
    (and checks they exist); ``kwargs`` go to the pandas reader for text formats."""
    fmt = _format(path, fmt)
    if fmt in ("csv", "tsv", "ecsv", "whitespace"):
        opts = dict(kwargs)
        if fmt == "tsv":
            opts.setdefault("sep", "\t")
        elif fmt == "whitespace":
            opts.setdefault("sep", r"\s+")
        elif fmt == "ecsv":
            opts.setdefault("sep", None)
            opts.setdefault("engine", "python")
        opts.setdefault("comment", "#")
        frame = pd.read_csv(path, **opts)
    elif fmt == "parquet":
        frame = pd.read_parquet(path, columns=list(columns) if columns else None)
    elif fmt in ("json", "jsonl"):
        frame = pd.read_json(path, lines=fmt == "jsonl")
    elif fmt == "npy":
        data = np.load(path, allow_pickle=False)
        frame = pd.DataFrame({n: data[n] for n in data.dtype.names}) if data.dtype.names else \
            pd.DataFrame({"value": data} if data.ndim == 1 else {f"c{i}": data[:, i] for i in range(data.shape[1])})
    elif fmt == "npz":
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if key:
            arrays = {key: arrays[key]}
        one_d = {k: v for k, v in arrays.items() if v.ndim == 1}
        if one_d:
            n = max(len(v) for v in one_d.values())
            frame = pd.DataFrame({k: v for k, v in one_d.items() if len(v) == n})
        else:
            frame = pd.DataFrame({k: [v] for k, v in arrays.items()})
    elif fmt == "hdf5":
        frame = _hdf5(path, key)
    elif fmt == "fits":
        frame = _fits(path, key)
    else:
        raise ValueError(f"unknown format '{fmt}'")
    if columns:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise KeyError(f"{path}: no column(s) {missing}; columns are {list(frame.columns)}")
        frame = frame[list(columns)]
    prov = provenance(path, fmt, key=key, rows=int(len(frame)), columns=[str(c) for c in frame.columns])
    return Table(frame, prov)


@dataclass
class Table:
    frame: pd.DataFrame
    provenance: Dict[str, Any]


@dataclass
class Series:
    """A measured 1-D series: ``time`` (or any coordinate), ``value``, optional ``error`` and a
    ``mask`` of points that are usable (finite and not flagged), with the file's provenance."""
    time: np.ndarray
    value: np.ndarray
    error: Optional[np.ndarray]
    provenance: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def mask(self) -> np.ndarray:
        ok = np.isfinite(self.time) & np.isfinite(self.value)
        return ok if self.error is None else ok & np.isfinite(self.error)

    @property
    def cadence(self) -> float:
        """Median spacing of the usable time stamps."""
        t = np.sort(self.time[self.mask])
        return float(np.median(np.diff(t))) if t.size > 1 else float("nan")

    def __len__(self) -> int:
        return int(self.time.size)


def read_series(path: str, value: str, time: Optional[str] = None, error: Optional[str] = None,
                quality: Optional[str] = None, good_quality: Sequence[int] = (0,), normalize: Optional[str] = None,
                fmt: Optional[str] = None, key=None, **kwargs) -> Series:
    """A series from a file: ``value`` column, ``time`` column (default: the row index), optional
    ``error`` and ``quality`` columns (rows whose quality is not in ``good_quality`` become NaN, as
    in TESS/Kepler light curves). ``normalize``: ``"median"`` divides value and error by the
    median of the usable values (relative flux), ``"subtract_median"`` subtracts it (residuals)."""
    wanted = [c for c in (time, value, error, quality) if c]
    table = read_table(path, fmt=fmt, key=key, columns=wanted, **kwargs)
    f = table.frame
    t = np.asarray(f[time], float) if time else np.arange(len(f), dtype=float)
    v = np.asarray(f[value], float).copy()
    e = np.asarray(f[error], float).copy() if error else None
    meta: Dict[str, Any] = {"value": value, "time": time, "error": error}
    if quality:
        bad = ~np.isin(np.asarray(f[quality]), list(good_quality))
        v[bad] = np.nan
        meta["flagged"] = int(bad.sum())
    if normalize:
        ok = np.isfinite(v)
        med = float(np.median(v[ok])) if ok.any() else float("nan")
        if normalize == "median":
            v = v / med
            e = None if e is None else e / med
        elif normalize == "subtract_median":
            v = v - med
        else:
            raise ValueError("normalize must be 'median' or 'subtract_median'")
        meta.update(normalize=normalize, median=med)
    return Series(t, v, e, table.provenance, meta)


def read_field(path: str, fmt: Optional[str] = None, key=None) -> Dict[str, Any]:
    """A 2-D (or N-D) array from NPY, NPZ (``key``), HDF5 (``key``) or a FITS image HDU (``key``),
    with its provenance: ``{"data": array, "provenance": {...}}``."""
    fmt = _format(path, fmt)
    if fmt == "npy":
        data = np.load(path, allow_pickle=False)
    elif fmt == "npz":
        with np.load(path, allow_pickle=False) as z:
            data = z[key or z.files[0]]
    elif fmt == "hdf5":
        import h5py
        with h5py.File(path, "r") as f:
            data = f[key][()] if key else next(v[()] for v in f.values() if isinstance(v, h5py.Dataset))
    elif fmt == "fits":
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            hdu = hdul[key] if key is not None else next(h for h in hdul if h.data is not None and h.data.ndim >= 2)
            data = np.asarray(hdu.data, dtype=float)
    else:
        raise ValueError(f"read_field reads npy, npz, hdf5 or fits, not '{fmt}'")
    data = np.asarray(data)
    return {"data": data, "provenance": provenance(path, fmt, key=key, shape=list(data.shape))}


def available_formats() -> List[str]:
    """Formats readable in this environment (optional libraries checked)."""
    out = ["csv", "tsv", "ecsv", "whitespace", "json", "jsonl", "npy", "npz"]
    for fmt, module in (("parquet", "pyarrow"), ("hdf5", "h5py"), ("fits", "astropy")):
        try:
            __import__(module)
            out.append(fmt)
        except ImportError:
            pass
    return out
