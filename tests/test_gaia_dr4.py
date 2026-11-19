"""The Gaia DR4 post-release tools on a simulated archive: selection, fetch, analysis."""
import io
import json
import os
import re
import zipfile

import numpy as np
import pandas as pd
import pytest

from scidatasim.experiments.astrometry import dr4, dr4_analysis
from scidatasim.experiments.astrometry.epochs import epoch_table, read_epoch_astrometry, source_epochs, write_ecsv
from scidatasim.experiments.astrometry.model import Orbit, along_scan, simulated_epochs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, "docs", "preregistration", "preregistration_gaia_dr4.json")


class FakeArchive:
    """Answers the fetch's queries from a simulated sky: a few controls per G bin, DR3 orbit
    companions (with DR4 epochs that contain their orbit) and the named black holes."""

    def __init__(self, per_bin=3, n_companions=4, seed=0):
        rng = np.random.default_rng(seed)
        self.epochs, self.rows, self.queries = {}, [], []
        sid = 1000
        for g_lo, g_hi in [(8, 10), (10, 12), (12, 14), (14, 16), (16, 17)]:
            for _ in range(per_bin + 2):  # two fail the sampling cut
                sid += 1
                self.rows.append({"source_id": sid, "g": rng.uniform(g_lo, g_hi), "kind": "control",
                                  "matched": int(rng.integers(40, 80)), "k1": rng.uniform(0, 0.4),
                                  "k2": rng.uniform(0, 0.4), "random_index": int(rng.integers(0, 10 ** 6))})
                self._source(sid, rng, None)
        self.companions = []
        for k in range(n_companions):
            sid += 1
            orbit = Orbit.from_campbell(rng.uniform(80, 900), rng.uniform(0, 0.5), rng.uniform(-30, 30),
                                        rng.uniform(1.0, 2.0), 1.0, 0.7, 0.3)
            self._source(sid, rng, orbit)
            self.companions.append({"dr3_source_id": 9_000_000 + k, "solution_type": "Orbital",
                                    "period": orbit.period, "period_error": 1.0, "eccentricity": orbit.eccentricity,
                                    "t_periastron": orbit.t_periastron - dr4.DR3_TO_DR4_DAYS,
                                    "a_thiele_innes": orbit.a, "b_thiele_innes": orbit.b, "f_thiele_innes": orbit.f,
                                    "g_thiele_innes": orbit.g, "significance": 50.0, "source_id": sid,
                                    "angular_distance": 0.1, "magnitude_difference": 0.01})
            self.rows.append({"source_id": sid, "g": 12.5, "kind": "companion", "matched": 60, "k1": 0.1,
                              "k2": 0.1, "random_index": 10 ** 7})
        self.named = [{"dr3_source_id": v, "source_id": sid + 1 + i, "angular_distance": 0.2,
                       "magnitude_difference": 0.0} for i, v in enumerate(dr4.KNOWN_BLACK_HOLES.values())]
        for row in self.named:
            self._source(row["source_id"], rng, None)

    def _source(self, sid, rng, orbit):
        ep = simulated_epochs(rng, n_visits=25)
        w = along_scan(ep.t, ep.theta, ep.f_al, {"parallax": 5.0, "pmra": 3.0}, orbit) + rng.normal(0, ep.sigma)
        self.epochs[sid] = (ep, w)

    class Job:
        def __init__(self, frame):
            self.frame = frame

        def get_results(self):
            return self.frame

    def launch_job_async(self, query, upload_resource=None, upload_table_name=None):
        self.queries.append(query)
        sky = pd.DataFrame(self.rows)
        controls = sky[sky["kind"] == "control"]
        g = re.search(r"phot_g_mean_mag >= ([\d.]+) AND g.phot_g_mean_mag < ([\d.]+)", query)
        if g:
            lo, hi = float(g.group(1)), float(g.group(2))
            sel = controls[(controls["g"] >= lo) & (controls["g"] < hi)]
        if query.startswith("SELECT g.astrometric_matched_transits"):
            return self.Job(sel.rename(columns={"matched": "astrometric_matched_transits",
                                                "k1": "scan_direction_strength_k1", "k2": "scan_direction_strength_k2"}))
        if "TOP" in query:
            n = int(re.search(r"TOP (\d+)", query).group(1))
            p25 = int(re.search(r"astrometric_matched_transits >= (\d+)", query).group(1))
            k1 = float(re.search(r"scan_direction_strength_k1 <= ([\d.e-]+)", query).group(1))
            k2 = float(re.search(r"scan_direction_strength_k2 <= ([\d.e-]+)", query).group(1))
            ok = sel[(sel["matched"] >= p25) & (sel["k1"] <= k1) & (sel["k2"] <= k2)].sort_values("random_index")
            return self.Job(ok.head(n).rename(columns={"g": "phot_g_mean_mag"}))
        if "nss_two_body_orbit AS o\nJOIN" in query:
            return self.Job(pd.DataFrame(self.companions))
        if "dr3_neighbourhood AS n WHERE" in query:
            return self.Job(pd.DataFrame(self.named))
        if "gaiadr4.gaia_source AS g ON" in query:
            ids = pd.read_csv(upload_resource) if upload_resource.endswith(".csv") else None
            base = sky.rename(columns={"g": "phot_g_mean_mag"})
            if ids is not None:
                base = base[base["source_id"].isin(ids["source_id"])]
            return self.Job(base.assign(ruwe=1.0, parallax=5.0, parallax_error=0.02, astrometric_excess_noise=0.0)
                            [["source_id", "phot_g_mean_mag", "ruwe", "parallax", "parallax_error",
                              "astrometric_excess_noise"]])
        if "spurious_signal" in query:
            return self.Job(pd.DataFrame({"source_id": [r["source_id"] for r in self.rows[:5]],
                                          "astro_circ_freq": 0.01, "astro_circ_freq_fap": -1.0}))
        return self.Job(pd.DataFrame({"source_id": [], "period": []}))  # DR4 nss: none

    def load_data(self, ids, data_release, retrieval_type, data_structure, format, output_file):
        assert retrieval_type == "EPOCH_ASTROMETRY" and len(ids) <= 5000
        frames = [epoch_table(sid, *self.epochs[sid]) for sid in ids]
        tmp = output_file + ".ecsv"
        write_ecsv(pd.concat(frames, ignore_index=True), tmp)
        with zipfile.ZipFile(output_file, "w") as zf:
            zf.write(tmp, "EPOCH_ASTROMETRY-Gaia DR4.ecsv")
        os.remove(tmp)


def test_record_copy_verifies_and_queries_fill():
    record = dr4.load_record(RECORD)
    assert record["provenance"]["git_commit"].startswith("b30a7e9")
    q = dr4.control_query(record, 8.0, 10.0, {"matched_transits_p25": 41.7, "k1_p75": 0.25, "k2_p75": 0.3})
    assert "TOP 600" in q and ">= 41" in q and "<= 0.25" in q and "{" not in q
    bad = dict(record, title="changed")
    path = os.path.join(ROOT, "tests", "_bad_record.json")
    try:
        with open(path, "w") as f:
            json.dump(bad, f)
        with pytest.raises(ValueError):
            dr4.load_record(path)
    finally:
        os.remove(path)


def test_dr3_orbit_moves_to_dr4_epoch():
    orbit = Orbit.from_campbell(300.0, 0.3, 20.0, 1.0, 1.0, 0.5, 0.2)
    row = {"period": 300.0, "eccentricity": 0.3, "t_periastron": 20.0 - dr4.DR3_TO_DR4_DAYS,
           "a_thiele_innes": orbit.a, "b_thiele_innes": orbit.b, "f_thiele_innes": orbit.f, "g_thiele_innes": orbit.g}
    back = dr4.dr3_orbit(row)
    t = np.linspace(-2, 2, 50)
    assert np.allclose(back.offsets(t), orbit.offsets(t))
    table = pd.DataFrame({"dr3_source_id": [1, 1, 2], "source_id": [10, 11, 20], "angular_distance": [0.5, 0.1, 0.2],
                          "magnitude_difference": [0.0, 0.1, 2.0]})
    assert dr4.best_match(table)["source_id"].tolist() == [11]


def test_fetch_and_analysis_on_a_simulated_archive(tmp_path):
    record = dr4.load_record(RECORD)
    archive = FakeArchive()
    out = str(tmp_path / "data")
    fetch = dr4.Fetch(archive, record, out, batch=7, per_bin=3, log=lambda *a: None)
    manifest = fetch.run()
    sample = pd.read_csv(os.path.join(out, "sample.csv"))
    assert (sample["sample"] == "control").sum() == 15 and (sample["sample"] == "dr3_companion").sum() == 4
    assert set(sample.loc[sample["sample"] == "named", "name"]) == set(dr4.KNOWN_BLACK_HOLES)
    assert all(len(h) == 64 for h in manifest["files"].values())
    n_queries = len(archive.queries)
    dr4.Fetch(archive, record, out, batch=7, per_bin=3, log=lambda *a: None).run()  # resumes: no new queries
    assert len(archive.queries) == n_queries

    report = dr4_analysis.analyze(record, out, jobs=1, replicates=2, log=lambda *a: None)
    assert report["counts"]["controls_with_epochs"] == 15
    assert report["H2"]["passed"] and set(report["H2"]["bins"]) == {"8-10", "10-12", "12-14", "14-16", "16-17"}
    assert report["H1"]["n"] == 30 and report["H1"]["passed"]
    assert len(report["_injections"]) == 15 and report["H3"]["tested"] == 0 and report["H3"]["passed"] is None
    assert report["H5"]["n"] == 4 and report["H5"]["rate"] == 1.0 and report["H5"]["passed"]
    assert "E2" in report["exploratory"] and "E1" in report["exploratory"]
    text = dr4_analysis.markdown(report)
    assert "H5" in text and "Choices not fixed by the record" in text


def test_h2_fallback_and_h4(tmp_path):
    record = dr4.load_record(RECORD)
    archive = FakeArchive(per_bin=2, n_companions=1, seed=3)
    for sid, (ep, w) in list(archive.epochs.items()):  # reported errors too small by 1.5
        archive.epochs[sid] = (type(ep)(ep.t, ep.theta, ep.f_al, ep.sigma / 1.5), w)
    out = str(tmp_path / "data")
    dr4.Fetch(archive, record, out, batch=50, per_bin=2, log=lambda *a: None).run()
    report = dr4_analysis.analyze(record, out, jobs=1, replicates=1, log=lambda *a: None)
    assert not report["H2"]["passed"]
    fb = report["fallback_rescaled"]
    assert all(1.3 < f < 1.7 for f in fb["factors"].values())
    rows = [{"noncentrality": 30.0, "detected": True, "detected_agis": False}] * 8 + \
        [{"noncentrality": 30.0, "detected": False, "detected_agis": True}]
    h4 = dr4_analysis.h4_agis(rows)
    assert h4["measurement_only"] == 8 and h4["agis_only"] == 1 and h4["passed"]
