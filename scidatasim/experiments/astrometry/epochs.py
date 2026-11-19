"""Gaia DR4 ``epoch_astrometry`` tables: read them into :class:`~.model.Epochs`, and write simulated
sources in the same layout so a pipeline can be run on simulated files before the real ones exist.

Layout (DR4 draft data model, issue D, 2026): one row per source and field-of-view transit; per
CCD, arrays of 10 values (index 0 is the sky mapper, 1-9 the astrometric field AF1-AF9):

* ``obs_time_tcb`` - long[10], ns since J2010.0 (TCB at Gaia); ``obs_time_bary_corr`` - ns, one
  per transit, added to get barycentric time
* ``scan_pos_angle`` - double[10], degrees
* ``centroid_pos_al`` - double[10], mas: the along-scan local-plane coordinate w relative to the
  reference point (``ra0``, ``dec0``)
* ``centroid_pos_error_al`` - float[10], mas
* ``used_by_agis_al`` - bool[10]
* ``parallax_factor_al`` - one per transit
* ``agis_source_excess_noise`` - one per source (repeated on every row), mas

Weights: AGIS used sigma^2 = centroid_pos_error_al^2 + agis_source_excess_noise^2. The source
excess noise absorbs unmodelled motion such as an orbit, so for an orbit search the measurement
errors alone are the natural weights (``weights="measurement"``, the default here);
``weights="agis"`` reproduces the AGIS weighting.

Known issue (data model): a few CCD values of ``centroid_pos_al`` are grossly wrong (beyond the
window, some above 1e9 mas). They are dropped here when they lie more than ``gross_limit`` mas from
a robust five-parameter fit.

Reading ECSV/CSV needs only pandas; VOTable and FITS need astropy (``pip install astropy``).
"""
from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from .model import DAYS_PER_YEAR, REF_EPOCH, TCB_ORIGIN, Epochs

ARRAY_COLUMNS = ("obs_time_tcb", "scan_pos_angle", "centroid_pos_al", "centroid_pos_error_al", "used_by_agis_al")
N_CCD = 10
NS_PER_YEAR = 1e9 * 86400.0 * DAYS_PER_YEAR
#: Along-scan window half-width is a few hundred mas; values this far from a robust
#: five-parameter fit are the grossly deviating centroids of the data model's known issue.
GROSS_LIMIT_MAS = 1000.0


def tcb_ns_to_years(ns) -> np.ndarray:
    """``obs_time_tcb`` (ns since J2010.0) to Julian years from the DR4 reference epoch J2017.5."""
    return np.asarray(ns, dtype=np.float64) / NS_PER_YEAR + (TCB_ORIGIN - REF_EPOCH)


def years_to_tcb_ns(t) -> np.ndarray:
    return np.round((np.asarray(t, float) - (TCB_ORIGIN - REF_EPOCH)) * NS_PER_YEAR).astype(np.int64)


# ---------------------------------------------------------------------------- reading
def _cell_array(value, dtype) -> np.ndarray:
    """An array cell as numpy: already a sequence (astropy, parquet), or text such as '[1, 2]',
    '(1, 2)' or '{1,2}' (CSV/ECSV)."""
    if isinstance(value, str):
        text = value.strip().strip("[](){}")
        items = [v.strip() for v in text.replace(";", ",").split(",") if v.strip()]
        if dtype is bool:
            return np.array([v.lower() in ("true", "t", "1") for v in items], dtype=bool)
        return np.array([float("nan") if v.lower() in ("nan", "null", "--", "") else v for v in items],
                        dtype=float if dtype is not np.int64 else object).astype(dtype)
    arr = np.ma.filled(np.ma.asarray(value), False if dtype is bool else 0 if dtype is np.int64 else np.nan)
    return np.asarray(arr).astype(dtype)


def _read_text(handle, name: str) -> pd.DataFrame:
    text = handle.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]  # ECSV header is '#'-prefixed
    return pd.read_csv(io.StringIO("\n".join(lines)), sep=None if name.endswith(".ecsv") else ",", engine="python")


def _read_one(path_or_handle, name: str) -> pd.DataFrame:
    lower = name.lower()
    if lower.endswith((".csv", ".ecsv", ".txt")):
        if isinstance(path_or_handle, (str, os.PathLike)):
            with open(path_or_handle, "rb") as f:
                return _read_text(f, lower)
        return _read_text(path_or_handle, lower)
    if lower.endswith(".parquet"):
        return pd.read_parquet(path_or_handle)
    try:
        from astropy.table import Table
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(f"reading {name} needs astropy (pip install astropy); ECSV/CSV need only pandas") from exc
    table = Table.read(path_or_handle, format="votable" if lower.endswith((".xml", ".vot", ".votable")) else None)
    return pd.DataFrame({c: list(table[c]) if table[c].ndim > 1 else np.asarray(table[c]) for c in table.colnames})


def read_epoch_astrometry(source: Union[str, os.PathLike, pd.DataFrame, Iterable]) -> pd.DataFrame:
    """Epoch astrometry rows from a file (CSV/ECSV/Parquet; VOTable/FITS with astropy), a ZIP of
    such files as DataLink returns them, a list of files, or a DataFrame. Array columns become
    numpy arrays; column names are lower-cased."""
    if isinstance(source, pd.DataFrame):
        frames = [source.copy()]
    elif isinstance(source, (str, os.PathLike)) and str(source).lower().endswith(".zip"):
        frames = []
        with zipfile.ZipFile(source) as zf:
            for member in sorted(zf.namelist()):
                if member.endswith("/"):
                    continue
                with zf.open(member) as handle:
                    data = handle.read()
                frames.append(_read_one(io.BytesIO(data), member))
    elif isinstance(source, (str, os.PathLike)):
        frames = [_read_one(source, str(source))]
    else:
        return pd.concat([read_epoch_astrometry(s) for s in source], ignore_index=True)
    frame = pd.concat(frames, ignore_index=True)
    frame.columns = [str(c).lower() for c in frame.columns]
    missing = {"source_id", "parallax_factor_al", *ARRAY_COLUMNS} - set(frame.columns)
    if missing:
        raise KeyError(f"not a DR4 epoch_astrometry table: missing columns {sorted(missing)}")
    kinds = {"obs_time_tcb": np.int64, "used_by_agis_al": bool}
    for col in ARRAY_COLUMNS:
        frame[col] = [_cell_array(v, kinds.get(col, float)) for v in frame[col]]
    return frame


def source_epochs(frame: pd.DataFrame, source_id: int, weights: str = "measurement",
                  barycentric: bool = True, gross_limit: float = GROSS_LIMIT_MAS):
    """One source's CCD-level measurements as ``(Epochs, w, info)``: the CCDs AGIS used in AL,
    with barycentric times (if the correction column is present and ``barycentric``), scan angles
    in radians, the transit's parallax factor, and weights as chosen (see the module docstring).
    ``info`` reports the number of CCDs, how many were dropped as gross outliers, and the reference
    point and excess noise."""
    if weights not in ("measurement", "agis"):
        raise ValueError("weights must be 'measurement' or 'agis'")
    rows = frame[frame["source_id"] == source_id]
    if rows.empty:
        raise KeyError(f"source_id {source_id} is not in the table")
    t, theta, f_al, w, err = [], [], [], [], []
    for _, row in rows.iterrows():
        used = np.asarray(row["used_by_agis_al"], bool)
        ns = np.asarray(row["obs_time_tcb"], np.int64)
        values = np.asarray(row["centroid_pos_al"], float)
        used = used & np.isfinite(values) & np.isfinite(np.asarray(row["centroid_pos_error_al"], float))
        if not used.any():
            continue
        bary = float(row["obs_time_bary_corr"]) if barycentric and "obs_time_bary_corr" in row \
            and np.isfinite(row["obs_time_bary_corr"]) else 0.0
        t.append(tcb_ns_to_years(ns[used] + bary))
        theta.append(np.radians(np.asarray(row["scan_pos_angle"], float)[used]))
        f_al.append(np.full(int(used.sum()), float(row["parallax_factor_al"])))
        w.append(values[used])
        err.append(np.asarray(row["centroid_pos_error_al"], float)[used])
    if not t:
        raise ValueError(f"source_id {source_id} has no CCD used by AGIS in AL")
    t, theta, f_al, w, err = (np.concatenate(v) for v in (t, theta, f_al, w, err))
    first = rows.iloc[0]
    excess = float(first.get("agis_source_excess_noise", 0.0) or 0.0)
    sigma = np.sqrt(err ** 2 + excess ** 2) if weights == "agis" else err
    ep = Epochs(t, theta, f_al, sigma)
    keep = _not_gross(ep, w, gross_limit)
    info = {"n_ccd": int(t.size), "n_gross": int((~keep).sum()), "n_transits": int(len(rows)),
            "ra0": float(first.get("ra0", np.nan)), "dec0": float(first.get("dec0", np.nan)),
            "agis_source_excess_noise": excess, "weights": weights}
    return Epochs(t[keep], theta[keep], f_al[keep], sigma[keep]), w[keep], info


def _not_gross(ep: Epochs, w: np.ndarray, limit: float) -> np.ndarray:
    """CCDs within ``limit`` mas of a robust five-parameter fit (least absolute deviations, by
    iteratively reweighted least squares) - only gross errors are removed, not orbits."""
    from .fit import five_parameter_design
    design = five_parameter_design(ep)
    weights = np.ones_like(w)
    finite = np.abs(w) < 1e8
    params = np.zeros(design.shape[1])
    for _ in range(30):
        sw = np.where(finite, weights, 0.0)
        params, *_ = np.linalg.lstsq(design * sw[:, None], w * sw, rcond=None)
        r = np.abs(w - design @ params)
        weights = 1.0 / np.sqrt(np.maximum(r, 1e-3))
    return finite & (np.abs(w - design @ params) <= limit)


# ---------------------------------------------------------------------------- writing
def epoch_table(source_id: int, ep: Epochs, w: np.ndarray, ra0: float = 0.0, dec0: float = 0.0,
                excess_noise: float = 0.0, transit_gap_days: float = 0.02) -> pd.DataFrame:
    """Simulated CCD measurements ``(ep, w)`` in the DR4 layout: CCDs closer in time than
    ``transit_gap_days`` form one transit (up to 9 AF CCDs; the sky-mapper slot 0 is unused)."""
    order = np.argsort(ep.t)
    t, theta, f_al, sigma, w = ep.t[order], ep.theta[order], ep.f_al[order], ep.sigma[order], np.asarray(w)[order]
    breaks = np.flatnonzero(np.diff(t) * DAYS_PER_YEAR > transit_gap_days) + 1
    rows = []
    for k, idx in enumerate(np.split(np.arange(t.size), breaks)):
        for chunk in np.array_split(idx, int(np.ceil(idx.size / (N_CCD - 1)))):
            slots = np.arange(1, chunk.size + 1)
            ns = np.zeros(N_CCD, np.int64)
            angle, pos, err = (np.full(N_CCD, np.nan) for _ in range(3))
            used = np.zeros(N_CCD, bool)
            ns[slots] = years_to_tcb_ns(t[chunk])
            angle[slots] = np.degrees(theta[chunk])
            pos[slots] = w[chunk]
            err[slots] = sigma[chunk]
            used[slots] = True
            rows.append({"source_id": int(source_id), "transit_id": int(source_id) * 10000 + len(rows),
                         "ra0": ra0, "dec0": dec0, "agis_source_excess_noise": excess_noise,
                         "obs_time_tcb": ns, "obs_time_bary_corr": 0.0, "scan_pos_angle": angle,
                         "centroid_pos_al": pos, "centroid_pos_error_al": err, "used_by_agis_al": used,
                         "parallax_factor_al": float(np.mean(f_al[chunk]))})
    return pd.DataFrame(rows)


def write_ecsv(frame: pd.DataFrame, path: str) -> None:
    """Write a table from :func:`epoch_table` as ECSV-like text (array cells as JSON lists), readable
    by :func:`read_epoch_astrometry`."""
    out = frame.copy()
    for col in ARRAY_COLUMNS:
        out[col] = [json.dumps([None if (isinstance(v, float) and np.isnan(v)) else
                                (bool(v) if isinstance(v, np.bool_) else v.item() if hasattr(v, "item") else v)
                                for v in arr]) for arr in out[col]]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# %ECSV 1.0\n# simulated by scidatasim (DR4 epoch_astrometry layout)\n")
        out.to_csv(f, index=False)


def sources(frame: pd.DataFrame) -> np.ndarray:
    return np.unique(frame["source_id"].to_numpy())


def summary(frame: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Counts a fetch can be checked against: sources, transits, CCDs used."""
    used = sum(int(np.asarray(v, bool).sum()) for v in frame["used_by_agis_al"])
    return {"sources": int(frame["source_id"].nunique()), "transits": int(len(frame)), "ccd_used_al": used}
