"""Cosmology: distances, the supernova experiment, its labels, fits and the validation report."""
import os

import numpy as np
import pytest

from scidatasim import validation
from scidatasim.engine import simulate
from scidatasim.experiments.cosmology.logic import ABSOLUTE_M, SupernovaHubbleExperiment, compare_models
from scidatasim.experiments.cosmology.model import (C_KM_S, comoving_distance, distance_modulus, e_of_z, fit,
                                                    transverse_distance)

EXP = "scidatasim.experiments.cosmology.logic.SupernovaHubbleExperiment"
OBS = {"targets": ["m_b"], "operators": [{"type": "gaussian", "sigma": 0.15}]}


def test_distances_against_closed_forms_and_astropy():
    z = np.array([0.01, 0.1, 0.5, 1.0, 2.3])
    eds = 2 * C_KM_S / 70.0 * (1 - 1 / np.sqrt(1 + z))  # Einstein-de Sitter
    assert comoving_distance(z, 1.0) == pytest.approx(eds, rel=1e-10)
    assert e_of_z(0.0, 0.3, -0.8, 0.3) == pytest.approx(1.0)
    open_ = transverse_distance(z, 0.3, ok=0.1)
    assert np.all(open_ > comoving_distance(z, 0.3, ok=0.1))
    cosmo = pytest.importorskip("astropy.cosmology")
    ref = cosmo.Flatw0waCDM(H0=70, Om0=0.3, w0=-0.8, wa=-0.5, Tcmb0=0)
    assert distance_modulus(z, 0.3, -0.8, -0.5) == pytest.approx(ref.distmod(z).value, abs=1e-6)
    ref = cosmo.LambdaCDM(H0=70, Om0=0.3, Ode0=0.6, Tcmb0=0)
    assert transverse_distance(z, 0.3, ok=0.1) == pytest.approx(ref.comoving_transverse_distance(z).value, rel=1e-8)


def test_fit_recovers_and_marginalises_m():
    rng = np.random.default_rng(0)
    z = np.sort(np.exp(rng.uniform(np.log(0.01), np.log(1.5), 800)))
    m = distance_modulus(z, 0.31) - 19.0 + rng.normal(0, 0.12, z.size)
    r = fit(z, m, 0.12, "lcdm")
    assert abs(r["params"]["om"] - 0.31) < 3 * r["errors"]["om"] and 0.005 < r["errors"]["om"] < 0.05
    shifted = fit(z, m + 0.7, 0.12, "lcdm")  # M is marginalised: an overall offset changes nothing
    assert shifted["params"]["om"] == pytest.approx(r["params"]["om"], abs=1e-4)  # optimiser tolerance
    assert shifted["chi2"] == pytest.approx(r["chi2"], abs=1e-3)  # the optimiser reaches ~4e-6; BLAS kernels differ


def test_experiment_labels_and_trap():
    lcdm = simulate(dict(experiment_type=EXP, seed=3, dark_energy="lcdm", om=0.3, observation=OBS))
    q = {i["template"]: i for i in lcdm.logic.questions(lcdm.results, None)}
    assert q["cosmological_constant"]["answer"] == "cosmological constant" and q["w0"]["answer"] == -1.0
    assert not q["cosmological_constant"]["instrument_artifact"] and lcdm.logic.lcdm_dchi2() < 1e-6
    assert 0.0 < q["omega_m"]["tolerance"] < q["w0"]["tolerance"]
    strong = SupernovaHubbleExperiment({"seed": 1, "dark_energy": "w0wa", "w0": -0.6, "wa": -1.4, "n_sne": 3000,
                                        "sigma_m": 0.1})
    strong.initialize({"seed": 1})
    weak = SupernovaHubbleExperiment({"seed": 1, "dark_energy": "w0wa", "w0": -0.97, "wa": -0.05})
    weak.initialize({"seed": 1})
    assert not strong.questions({}, None)[0]["signal_undetectable"] and weak.questions({}, None)[0]["signal_undetectable"]
    trap = simulate(dict(experiment_type=EXP, seed=3, dark_energy="lcdm", om=0.3, calibration={"offset": 0.05},
                         observation=OBS))
    items = trap.logic.questions(trap.results, None)
    assert all(i["instrument_artifact"] for i in items) and trap.logic.lcdm_dchi2() > 1.0
    base = compare_models(trap.results["z"]["data"], trap.results["m_b"]["data"], 0.15)  # noise-free: the bias only
    assert base["w0"] > -0.95  # the offset pulls w0 away from -1
    with pytest.raises(ValueError):
        SupernovaHubbleExperiment({"dark_energy": "phantom"})


def test_redshifts_from_a_file(tmp_path):
    np.savez(tmp_path / "z.npz", zHD=np.linspace(0.005, 1.2, 300))
    run = simulate(dict(experiment_type=EXP, seed=1, redshifts={"source": str(tmp_path / "z.npz"), "column": "zHD"}))
    assert run.results["z"]["data"].min() >= 0.01 and len(run.results["z"]["data"]) == 298
    assert len(run.ground_truth["parameters"]["redshifts.sha256"]) == 64
    assert run.results["m_b"]["data"] == pytest.approx(run.results["mu"]["data"] + ABSOLUTE_M)


def test_validation_report():
    rng = np.random.default_rng(1)
    est = 0.3 + rng.normal(0, 0.02, 200)
    rep = validation.validation_report("toy", "mean", "om", {"estimate": 0.331, "error": 0.018},
                                       {"value": 0.334, "error": 0.018, "reference": "toy"}, 0.3, est,
                                       np.full(200, 0.02))
    assert rep["passed"] and rep["checks"] == {"reproduces_published": True, "unbiased": True, "coverage": True}
    bad = validation.validation_report("toy", "mean", "om", {"estimate": 0.4, "error": 0.018},
                                       {"value": 0.334, "error": 0.018}, 0.3, est + 0.01, np.full(200, 0.005))
    assert bad["checks"] == {"reproduces_published": False, "unbiased": False, "coverage": False}
    assert "reproduces the published value | NO" in validation.markdown(bad)


def test_validate_pantheon_tool_offline(tmp_path):
    """The tool on a small fake release (same file layout), without network."""
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("vp", os.path.join(root, "tools", "validate_pantheon.py"))
    vp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vp)
    rng = np.random.default_rng(2)
    z = np.sort(np.exp(rng.uniform(np.log(0.012), np.log(1.5), 150)))
    m = distance_modulus(z, 0.334) - 19.25 + rng.normal(0, 0.15, z.size)
    data = tmp_path / "data"
    data.mkdir()
    with open(data / "Pantheon+SH0ES.dat", "w") as f:
        f.write("CID zHD zHEL m_b_corr\n" + "".join(f"sn{i} {a} {a} {b}\n" for i, (a, b) in enumerate(zip(z, m))))
    with open(data / "Pantheon+SH0ES_STAT+SYS.cov", "w") as f:
        f.write(f"{z.size}\n" + "\n".join(str(v) for v in (np.eye(z.size) * 0.15 ** 2).ravel()) + "\n")
    vp.main(["--data", str(data), "--out", str(tmp_path / "out"), "--n-sim", "4"])
    assert (tmp_path / "out" / "validation_pantheon_om.md").exists()
    assert (tmp_path / "out" / "validation_pantheon_w0.json").exists()
