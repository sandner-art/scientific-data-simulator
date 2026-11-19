"""Gaia DR4 after the release: the selection, the fetch and the analysis pre-registered in
10.5281/zenodo.24370371 (record made with scidatasim 0.7.0, commit b30a7e9).

Everything here calls the registered method (:func:`.fit.detect`, :func:`.fit.orbit_noncentrality`,
:func:`.fit.excess_noise`, :func:`.epochs.source_epochs`) unchanged. What the record leaves open is
fixed here and reported as such by the analysis (``CHOICES``): seeds for the null replicates and
the injected orbits, how DR3 orbits are carried to DR4, how known companions are matched.

Network access goes through a ``client`` with astroquery's ``Gaia`` interface
(``launch_job_async``, ``load_data``), so the fetch can be tested without the archive.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .model import DAYS_PER_YEAR, Orbit

DR4, DR3 = "gaiadr4", "gaiadr3"
#: DR3 orbits give ``t_periastron`` in days from J2016.0; DR4 counts from J2017.5.
DR3_TO_DR4_DAYS = -1.5 * DAYS_PER_YEAR
KNOWN_BLACK_HOLES = {"Gaia BH1": 4373465352415301632, "Gaia BH2": 5870569352746779008,
                     "Gaia BH3": 4318465066420528000}  # Gaia DR3 source_id
#: Seeds and rules the record does not fix (reported with the results).
CHOICES = {
    "h1_seed": "numpy default_rng(400000 + control index); replicates drawn in order",
    "injection": "AstrometricBinaryExperiment({'seed': 300000 + control index, 'companion': True}) - its orbit "
                 "(defaults: period 20-2000 d log-uniform, a0 0.06-2 mas log-uniform, e < 0.7, isotropic)",
    "dr3_orbit": "DR3 Thiele-Innes A, B, F, G (mas), period, eccentricity; t_periastron shifted from J2016.0 to "
                 "J2017.5 (minus 547.875 d)",
    "dr3_match": "gaiadr4.dr3_neighbourhood: for each DR3 source the DR4 source with the smallest angular_distance, "
                 "|magnitude_difference| < 0.5",
    "h5_final_period": "fit.detect's period_best (the Keplerian period when detected, else the periodogram peak)",
}


# ---------------------------------------------------------------------------- the record
def load_record(path: str) -> Dict[str, Any]:
    """The pre-registration record, its own hash checked (``record_sha256``)."""
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                       default=float).encode("utf-8")).hexdigest()
    if digest != record.get("record_sha256"):
        raise ValueError(f"{path}: record_sha256 does not match its content")
    return record


def hypothesis(record: Dict[str, Any], hid: str) -> Dict[str, Any]:
    for h in record["hypotheses"]:
        if h["id"] == hid:
            return h
    raise KeyError(hid)


# ---------------------------------------------------------------------------- queries
def percentile_query(record, g_lo: float, g_hi: float) -> str:
    return record["selection"]["controls"]["percentile_adql"].format(g_lo=g_lo, g_hi=g_hi)


def resolve_percentiles(table: pd.DataFrame) -> Dict[str, float]:
    """The bin's thresholds: 25th percentile of matched transits, 75th of scan-direction strengths."""
    t = table.rename(columns=str.lower)
    return {"matched_transits_p25": float(np.nanpercentile(t["astrometric_matched_transits"], 25)),
            "k1_p75": float(np.nanpercentile(t["scan_direction_strength_k1"], 75)),
            "k2_p75": float(np.nanpercentile(t["scan_direction_strength_k2"], 75)), "n": int(len(t))}


def control_query(record, g_lo: float, g_hi: float, percentiles: Dict[str, float], n: Optional[int] = None) -> str:
    size = n if n is not None else int(record["selection"]["controls"]["size"].split()[0])
    return record["selection"]["controls"]["adql"].format(
        n=size, g_lo=g_lo, g_hi=g_hi, matched_transits_p25=int(np.floor(percentiles["matched_transits_p25"])),
        k1_p75=repr(percentiles["k1_p75"]), k2_p75=repr(percentiles["k2_p75"]))


def dr3_companion_query() -> str:
    """DR3 astrometric orbits with significance > 20 and period < 1500 d that have DR4 epoch
    astrometry, with their DR4 neighbours (all candidates; :func:`best_match` picks one)."""
    return f"""SELECT o.source_id AS dr3_source_id, o.solution_type, o.period, o.period_error, o.eccentricity,
       o.t_periastron, o.a_thiele_innes, o.b_thiele_innes, o.f_thiele_innes, o.g_thiele_innes,
       o.significance, n.source_id AS source_id, n.angular_distance, n.magnitude_difference
FROM {DR3}.nss_two_body_orbit AS o
JOIN {DR4}.dr3_neighbourhood AS n ON n.dr3_source_id = o.source_id
JOIN {DR4}.all_source_flags AS f ON f.source_id = n.source_id
WHERE o.solution_type IN ('Orbital', 'AstroSpectroSB1') AND o.significance > 20 AND o.period < 1500
  AND f.has_epoch_astrometry = 'true'"""


def named_query(dr3_ids: Sequence[int]) -> str:
    ids = ", ".join(str(int(i)) for i in dr3_ids)
    return f"""SELECT n.dr3_source_id, n.source_id, n.angular_distance, n.magnitude_difference
FROM {DR4}.dr3_neighbourhood AS n WHERE n.dr3_source_id IN ({ids})"""


def auxiliary_queries() -> Dict[str, str]:
    """DR4 rows for the uploaded sample (``TAP_UPLOAD.sample`` with a ``source_id`` column)."""
    return {
        "nss_two_body_orbit": f"SELECT o.* FROM TAP_UPLOAD.sample AS s JOIN {DR4}.nss_two_body_orbit AS o "
                              "ON o.source_id = s.source_id",
        "spurious_signal": f"SELECT p.* FROM TAP_UPLOAD.sample AS s JOIN {DR4}.spurious_signal AS p "
                           "ON p.source_id = s.source_id",
        "gaia_source": f"SELECT g.source_id, g.phot_g_mean_mag, g.ruwe, g.parallax, g.parallax_error, "
                       f"g.astrometric_excess_noise FROM TAP_UPLOAD.sample AS s JOIN {DR4}.gaia_source AS g "
                       "ON g.source_id = s.source_id",
    }


def best_match(table: pd.DataFrame, max_dmag: float = 0.5) -> pd.DataFrame:
    """One DR4 source per DR3 source: the nearest neighbour with |dG| < ``max_dmag``."""
    t = table.rename(columns=str.lower)
    t = t[np.abs(t["magnitude_difference"].fillna(0.0)) < max_dmag]
    return t.sort_values(["dr3_source_id", "angular_distance"]).groupby("dr3_source_id", as_index=False).first()


def dr3_orbit(row) -> Orbit:
    """A DR3 ``nss_two_body_orbit`` row as an :class:`Orbit` on the DR4 time axis."""
    v = {k: float(row[k]) for k in ("period", "eccentricity", "t_periastron", "a_thiele_innes", "b_thiele_innes",
                                    "f_thiele_innes", "g_thiele_innes")}
    return Orbit(v["period"], v["eccentricity"], v["t_periastron"] + DR3_TO_DR4_DAYS,
                 v["a_thiele_innes"], v["b_thiele_innes"], v["f_thiele_innes"], v["g_thiele_innes"])


# ---------------------------------------------------------------------------- fetch
def _table(job_or_table) -> pd.DataFrame:
    table = job_or_table.get_results() if hasattr(job_or_table, "get_results") else job_or_table
    if hasattr(table, "to_pandas"):
        table = table.to_pandas()
    return pd.DataFrame(table).rename(columns=str.lower)


def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Fetch:
    """The registered selection and fetch, step by step, into ``out``; every query, row count and file
    (with SHA-256) goes into ``manifest.json``. Steps already done (their file exists) are skipped,
    so an interrupted fetch resumes."""

    def __init__(self, client, record: Dict[str, Any], out: str, batch: int = 1000, data_release: str = "Gaia DR4",
                 data_structure: str = "RAW", fmt: str = "ecsv", per_bin: Optional[int] = None, log=print):
        self.client, self.record, self.out, self.batch = client, record, out, int(batch)
        self.data_release, self.data_structure, self.fmt, self.per_bin, self.log = \
            data_release, data_structure, fmt, per_bin, log
        os.makedirs(out, exist_ok=True)
        self.manifest_path = os.path.join(out, "manifest.json")
        self.manifest = {"record_sha256": record.get("record_sha256"), "steps": {}, "files": {}}
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, encoding="utf-8") as f:
                self.manifest = json.load(f)

    def _save(self):
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=1)

    def _query(self, name: str, query: str, upload: Optional[str] = None) -> pd.DataFrame:
        path = os.path.join(self.out, f"{name}.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        self.log(f"query {name}")
        kwargs = {"upload_resource": upload, "upload_table_name": "sample"} if upload else {}
        table = _table(self.client.launch_job_async(query, **kwargs))
        table.to_csv(path, index=False)
        self.manifest["steps"][name] = {"query": query, "rows": int(len(table)),
                                        "retrieved_utc": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")}
        self.manifest["files"][f"{name}.csv"] = sha256_path(path)
        self._save()
        return table

    def select(self) -> pd.DataFrame:
        """Percentiles and controls per G bin, then DR3 companions and the named black holes."""
        frames = []
        for g_lo, g_hi in self.record["selection"]["controls"]["g_bins"]:
            tag = f"{g_lo:g}-{g_hi:g}"
            pct = resolve_percentiles(self._query(f"percentiles_G{tag}", percentile_query(self.record, g_lo, g_hi)))
            self.manifest["steps"].setdefault("thresholds", {})[tag] = pct
            controls = self._query(f"controls_G{tag}", control_query(self.record, g_lo, g_hi, pct, self.per_bin))
            frames.append(controls.assign(sample="control", g_bin=tag))
        comp = best_match(self._query("dr3_companions", dr3_companion_query()))
        named = best_match(self._query("named", named_query(list(KNOWN_BLACK_HOLES.values()))))
        names = {v: k for k, v in KNOWN_BLACK_HOLES.items()}
        named = named.assign(name=[names.get(int(i), "") for i in named["dr3_source_id"]])
        sample = pd.concat(frames + [comp.assign(sample="dr3_companion"), named.assign(sample="named")],
                           ignore_index=True)
        sample.to_csv(os.path.join(self.out, "sample.csv"), index=False)
        self.manifest["files"]["sample.csv"] = sha256_path(os.path.join(self.out, "sample.csv"))
        self._save()
        return sample

    def auxiliary(self, sample: pd.DataFrame) -> None:
        ids = pd.DataFrame({"source_id": sorted(set(int(i) for i in sample["source_id"]))})
        upload = os.path.join(self.out, "sample_ids.csv")
        ids.to_csv(upload, index=False)
        try:  # astroquery uploads VOTable; write one if astropy is there, else the CSV path
            from astropy.table import Table
            upload = os.path.join(self.out, "sample_ids.vot")
            Table.from_pandas(ids).write(upload, format="votable", overwrite=True)
        except ImportError:
            pass
        for name, query in auxiliary_queries().items():
            self._query(f"dr4_{name}", query, upload=upload)

    def epochs(self, sample: pd.DataFrame) -> List[str]:
        ids = sorted(set(int(i) for i in sample["source_id"]))
        files = []
        for k in range(0, len(ids), self.batch):
            name = f"epoch_astrometry_{k // self.batch:03d}.zip"
            path = os.path.join(self.out, name)
            files.append(path)
            if os.path.exists(path):
                continue
            self.log(f"epoch astrometry {k + 1}-{min(k + self.batch, len(ids))} of {len(ids)}")
            result = self.client.load_data(ids=ids[k:k + self.batch], data_release=self.data_release,
                                           retrieval_type="EPOCH_ASTROMETRY", data_structure=self.data_structure,
                                           format=self.fmt, output_file=path)
            if not os.path.exists(path):  # some astroquery versions return the tables instead of a file
                _write_tables(result, path)
            self.manifest["files"][name] = sha256_path(path)
            self._save()
        return files

    def run(self) -> Dict[str, Any]:
        sample = self.select()
        self.auxiliary(sample)
        self.epochs(sample)
        return self.manifest


def _write_tables(result, path: str) -> None:
    import zipfile
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, tables in (result or {}).items():
            for i, table in enumerate(tables if isinstance(tables, list) else [tables]):
                if hasattr(table, "colnames"):  # astropy: to_pandas() refuses multidimensional columns
                    frame = pd.DataFrame({c: list(np.asarray(table[c])) if np.ndim(table[c]) > 1
                                          else np.asarray(table[c]) for c in table.colnames})
                else:
                    frame = pd.DataFrame(table)
                for col in frame.columns:  # arrays as JSON lists, readable by epochs.read_epoch_astrometry
                    if len(frame) and isinstance(frame[col].iloc[0], (list, np.ndarray)):
                        frame[col] = [json.dumps(np.asarray(v).tolist()) for v in frame[col]]
                zf.writestr(f"{os.path.splitext(os.path.basename(name))[0]}_{i}.csv", frame.to_csv(index=False))


def iter_sources(frame: pd.DataFrame) -> Iterable:
    """(source_id, rows) per source, grouped once (``epochs.source_epochs`` filters a frame)."""
    for sid, rows in frame.groupby("source_id", sort=True):
        yield int(sid), rows
