"""Astrometric binaries: the measurement model, the fits, DR4 epoch tables and the experiment."""
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from scidatasim.engine import SimulatorEngine, simulate
from scidatasim.experiments.astrometry.epochs import (NS_PER_YEAR, epoch_table, read_epoch_astrometry,
                                                      source_epochs, summary, tcb_ns_to_years, write_ecsv)
from scidatasim.experiments.astrometry.fit import (_keplerian_solution, candidate_periods, detect, excess_noise,
                                                   false_alarm_probability, fit_single_star, orbit_noncentrality,
                                                   orbit_periodogram)
from scidatasim.experiments.astrometry.logic import AstrometricBinaryExperiment
from scidatasim.experiments.astrometry.model import (Orbit, along_scan, campbell, gaia_position, inject_orbit,
                                                     kepler_eccentric_anomaly, local_triad, parallax_factor_al,
                                                     photocentre_a0, simulated_epochs, thiele_innes)
from scidatasim.export.tasks import collect_bundle_tasks

EXP = "scidatasim.experiments.astrometry.logic.AstrometricBinaryExperiment"
ASTRO = dict(ra_offset=0.3, dec_offset=-0.2, parallax=8.0, pmra=5.0, pmdec=-3.0)


# ------------------------------------------------------------------ model
def test_thiele_innes_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a0, inc = rng.uniform(0.1, 5), np.arccos(rng.uniform(-1, 1))
        omega, node = rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)
        back = campbell(*thiele_innes(a0, inc, omega, node))
        assert back["a0"] == pytest.approx(a0, rel=1e-10)
        assert back["inclination"] == pytest.approx(inc, abs=1e-9)
        assert back["node"] == pytest.approx(node, abs=1e-9)
        assert abs(np.angle(np.exp(1j * (back["omega"] - omega)))) < 1e-7


@pytest.mark.parametrize("e", [0.0, 0.3, 0.7, 0.95])
def test_kepler_equation(e):
    m = np.linspace(-10, 10, 2001)
    ecc = kepler_eccentric_anomaly(m, e)
    assert np.max(np.abs(ecc - e * np.sin(ecc) - np.mod(m, 2 * np.pi))) < 1e-10
    with pytest.raises(ValueError):
        kepler_eccentric_anomaly(m, 1.0)


def test_orbit_at_periastron_and_period():
    orbit = Orbit.from_campbell(300.0, 0.4, 25.0, 2.0, 1.0, 0.5, 1.2)
    da, dd = orbit.offsets(np.array([25.0, 325.0]) / 365.25)
    assert da == pytest.approx([orbit.b * 0.6] * 2) and dd == pytest.approx([orbit.a * 0.6] * 2)
    assert orbit.a0 == pytest.approx(2.0)


def test_parallax_factor_is_the_derivative_of_w():
    """f_al from the formula equals the along-scan shift of the apparent direction of a star at
    distance 1/parallax seen from Gaia's position, per unit parallax."""
    ra, dec = 123.0, -41.0
    t = np.linspace(-2.5, 2.5, 13)
    theta = np.linspace(0.1, 6.0, 13)
    p, q = local_triad(ra, dec)
    u = np.cross(p, q)  # towards the star
    plx = 1e-6  # rad: au units, so the star is at 1 / plx au
    b = gaia_position(t)
    direction = u / plx - b
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    shift_w = (direction @ p) * np.sin(theta) + (direction @ q) * np.cos(theta)
    assert shift_w / plx == pytest.approx(parallax_factor_al(t, theta, ra, dec), rel=1e-5, abs=1e-6)


def test_photocentre_a0():
    # the Sun-Jupiter system seen from 10 pc: about 0.5 mas
    assert photocentre_a0(100.0, 4332.6, 1.0, 9.55e-4) == pytest.approx(0.497, rel=0.01)
    assert photocentre_a0(100.0, 365.25, 1.0, 1.0, flux_ratio=1.0) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ fits
def test_noise_free_single_star_is_recovered():
    rng = np.random.default_rng(1)
    ep = simulated_epochs(rng)
    fit = fit_single_star(ep, along_scan(ep.t, ep.theta, ep.f_al, ASTRO))
    for k, v in ASTRO.items():
        assert fit[k] == pytest.approx(v, abs=1e-9)
    assert fit["chi2"] < 1e-12


def test_orbit_detected_and_fitted():
    rng = np.random.default_rng(1)
    ep = simulated_epochs(rng)
    orbit = Orbit.from_campbell(300.0, 0.3, 50.0, 1.5, 1.0, 0.5, 1.2)
    w = along_scan(ep.t, ep.theta, ep.f_al, ASTRO, orbit) + rng.normal(0, ep.sigma)
    found = detect(ep, w)
    assert found["detected"] and found["period"] == pytest.approx(300.0, rel=0.02)
    kep = found["keplerian"]
    assert kep["a0"] == pytest.approx(1.5, rel=0.1) and kep["significance"] > 20
    assert kep["parallax"] == pytest.approx(8.0, abs=0.2)
    assert orbit_noncentrality(ep, orbit) > 100
    assert excess_noise(ep, w) > 0.5


def test_false_alarms_on_single_stars():
    """The FAP approximation is not anti-conservative: few single stars fall below 1e-2."""
    rng = np.random.default_rng(2)
    faps = []
    for _ in range(40):
        ep = simulated_epochs(rng, n_visits=20)
        faps.append(detect(ep, along_scan(ep.t, ep.theta, ep.f_al, ASTRO) + rng.normal(0, ep.sigma),
                           keplerian=False)["fap"])
    assert np.mean(np.array(faps) < 1e-2) <= 0.1
    assert false_alarm_probability(0.0, 100) == pytest.approx(1.0)


# ------------------------------------------------------------------ DR4 epoch tables
def test_epoch_table_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    ep = simulated_epochs(rng)
    w = along_scan(ep.t, ep.theta, ep.f_al, ASTRO) + rng.normal(0, ep.sigma)
    bad = w.copy()
    bad[5], bad[17] = 3e9, bad[17] + 5000.0  # the data model's grossly deviating centroids
    frame = pd.concat([epoch_table(42, ep, w, ra0=10.0, dec0=20.0, excess_noise=0.3), epoch_table(44, ep, bad)])
    write_ecsv(frame, str(tmp_path / "e.ecsv"))
    with zipfile.ZipFile(tmp_path / "EPOCH_ASTROMETRY.zip", "w") as zf:
        zf.write(tmp_path / "e.ecsv", "EPOCH_ASTROMETRY-Gaia DR4 42.ecsv")
    table = read_epoch_astrometry(str(tmp_path / "EPOCH_ASTROMETRY.zip"))
    assert summary(table) == {"sources": 2, "transits": 2 * len(ep), "ccd_used_al": 2 * len(ep)}
    got, w_got, info = source_epochs(table, 42)
    order = np.argsort(ep.t)
    assert np.max(np.abs(got.t - ep.t[order])) * 365.25 * 86400 < 1e-3  # ns resolution
    assert np.array_equal(w_got, w[order]) and info["n_gross"] == 0 and info["ra0"] == 10.0
    assert source_epochs(table, 42, weights="agis")[0].sigma == pytest.approx(np.hypot(0.2, 0.3))
    _, w_clean, info = source_epochs(table, 44)
    assert info["n_gross"] == 2 and np.max(np.abs(w_clean)) < 1000
    with pytest.raises(KeyError):
        source_epochs(table, 7)
    with pytest.raises(KeyError):
        read_epoch_astrometry(pd.DataFrame({"source_id": [1]}))


def test_tcb_time_origin():
    assert tcb_ns_to_years(7.5 * NS_PER_YEAR) == pytest.approx(0.0)
    assert tcb_ns_to_years(0) == pytest.approx(-7.5)


# ------------------------------------------------------------------ experiment
def test_experiment_is_reproducible_and_labelled():
    a = simulate(dict(experiment_type=EXP, seed=5, companion=True))
    b = simulate(dict(experiment_type=EXP, seed=5, companion=True))
    assert np.array_equal(a.results["w_observed"]["data"], b.results["w_observed"]["data"])
    assert a.results["has_companion"]["data"] == "yes"
    assert simulate(dict(experiment_type=EXP, seed=5, companion=False)).results["has_companion"]["data"] == "no"
    with pytest.raises(ValueError):
        AstrometricBinaryExperiment({"companion": "maybe"})
    with pytest.raises(ValueError):
        simulate(dict(experiment_type=EXP, seed=1, companion=True, orbit={"semi_major_axis": 1.0}))


def test_undetectable_label():
    rng = np.random.default_rng(0)
    weak = AstrometricBinaryExperiment({"seed": 1, "companion": True, "orbit": {"a0": 0.01}})
    weak.initialize({"seed": 1})
    strong = AstrometricBinaryExperiment({"seed": 1, "companion": True, "orbit": {"a0": 2.0}})
    strong.initialize({"seed": 1})
    single = AstrometricBinaryExperiment({"seed": 1, "companion": False})
    single.initialize({"seed": 1})
    assert weak.signal_undetectable() and not strong.signal_undetectable() and not single.signal_undetectable()
    items = {q["template"]: q for q in single.questions(single.get_results(), rng)}
    assert items["astrometric_period"]["answer"] == 0.0 and items["astrometric_companion"]["answer"] == "no"
    items = {q["template"]: q for q in strong.questions(strong.get_results(), rng)}
    assert items["astrometric_period"]["tolerance_rule"] == \
        {q["template"]: q for q in single.questions(single.get_results(), rng)}["astrometric_period"]["tolerance_rule"]


def test_bundle_hides_answers_and_labels(tmp_path):
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(dict(
        experiment_type=EXP, seed=3, companion=True, orbit={"a0": 0.01}, generate_plots=False,
        bundle={"qa": {"value_queries": 0, "exclude": ["argmax", "max", "min", "final", "peak_count", "compare_max"]}}))
    tasks = {t["template"]: t for t in collect_bundle_tasks(run.experiment_dir)}
    assert set(tasks) == {"astrometric_companion", "astrometric_period", "astrometric_parallax"}
    assert tasks["astrometric_companion"]["signal_undetectable"] is True
    with open(os.path.join(run.experiment_dir, "bundle", "questions.json")) as f:
        public = json.load(f)
    assert not any(k in q for q in public for k in ("answer", "tolerance", "signal_undetectable", "period_ambiguous"))
    assert tasks["astrometric_period"]["period_ambiguous"] is False


def test_injection_and_eccentric_period():
    """An eccentric orbit injected into single-star data: the circular search's peak may sit at a
    harmonic, the Keplerian candidates include twice each peak, and the final period is right; the
    pipeline's parallax comes from the orbit fit."""
    rng = np.random.default_rng(4)
    ep = simulated_epochs(rng)
    base = along_scan(ep.t, ep.theta, ep.f_al, ASTRO) + rng.normal(0, ep.sigma)
    orbit = Orbit.from_campbell(400.0, 0.8, 30.0, 2.0, 0.6, 2.0, 0.4)
    w = inject_orbit(ep, base, orbit)
    assert np.allclose(w - base, along_scan(ep.t, ep.theta, np.zeros(len(ep)), {}, orbit))
    gram = orbit_periodogram(ep, w)
    cands = candidate_periods(gram)
    peak = gram["period"][np.argmax(gram["dchi2"])]
    assert any(abs(c / (2 * peak) - 1) < 1e-9 for c in cands)
    found = detect(ep, w)
    assert found["period_best"] == pytest.approx(400.0, rel=0.02)
    assert found["parallax"] == pytest.approx(found["keplerian"]["parallax"])
    assert abs(found["parallax"] - 8.0) < abs(found["single_star"]["parallax"] - 8.0) + 0.05
    truth = _keplerian_solution(ep, w, 400.0, 0.8, 30.0)
    assert found["keplerian"]["chi2"] <= truth["chi2"] + 1e-6


def test_period_ambiguous_label():
    strong = AstrometricBinaryExperiment({"seed": 1, "companion": True, "orbit": {"a0": 2.0, "period": 150.0}})
    strong.initialize({"seed": 1})
    assert not strong.period_ambiguous()
    assert strong.detection_probability() > 0.99 and strong.noncentrality() > 100
    single = AstrometricBinaryExperiment({"seed": 1, "companion": False})
    single.initialize({"seed": 1})
    assert not single.period_ambiguous() and single.detection_probability() == 0.0


def test_preregistration_hashes_ignore_line_endings(tmp_path):
    """A Windows checkout (CRLF) and a Linux one (LF) give the same file hashes."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("prereg", os.path.join(root, "tools", "preregister_gaia_dr4.py"))
    prereg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prereg)
    (tmp_path / "lf.py").write_bytes(b"a = 1\nb = 2\n")
    (tmp_path / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    assert prereg.sha256_file(str(tmp_path / "lf.py")) == prereg.sha256_file(str(tmp_path / "crlf.py"))
    record = {"schema": prereg.SCHEMA, "x": 1.5}
    record["record_sha256"] = __import__("hashlib").sha256(prereg.canonical(record)).hexdigest()
    assert json.loads(json.dumps(record)) == record and prereg.canonical(json.loads(json.dumps(record))) == \
        prereg.canonical(record)
