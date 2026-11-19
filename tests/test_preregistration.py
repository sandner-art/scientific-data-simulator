"""Generic pre-registration records: seal, verify, check, and the sds commands."""
import json
import os

import pytest

from scidatasim import cli
from scidatasim import preregistration as pr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _spec():
    return {"title": "Toy registration", "method": {"threshold": 0.01},
            "hypotheses": [
                {"id": "H1", "statement": "rate <= 0.02", "metric": "rate", "test": {"max": 0.02}},
                {"id": "H2", "statement": "bias in band", "metric": "bias", "test": {"between": [-0.1, 0.1]}},
                {"id": "H3", "statement": "needs data we lack", "metric": "power", "test": {"min": 0.8}},
                {"id": "E1", "kind": "exploratory", "statement": "look around"}]}


def test_seal_verify_check(tmp_path):
    (tmp_path / "method.py").write_bytes(b"x = 1\r\ny = 2\r\n")  # CRLF on disk
    record = pr.seal(_spec(), files=["method.py"], root=str(tmp_path))
    assert record["schema"] == pr.SCHEMA and len(record["record_sha256"]) == 64
    assert record["provenance"]["files_sha256"]["method.py"] == pr.file_hash(str(tmp_path / "method.py"))
    (tmp_path / "method.py").write_bytes(b"x = 1\ny = 2\n")  # the same file with LF: unchanged
    assert pr.verify(record, root=str(tmp_path))["ok"]
    (tmp_path / "method.py").write_bytes(b"x = 2\n")
    v = pr.verify(record, root=str(tmp_path))
    assert v["hash_ok"] and v["changed_files"] == ["method.py"] and not v["ok"]
    tampered = dict(record, method={"threshold": 0.05})
    assert not pr.verify(tampered, root=str(tmp_path))["hash_ok"]
    rows = {r["id"]: r for r in pr.check(record, {"rate": 0.015, "bias": 0.3})}
    assert rows["H1"]["passed"] is True and rows["H2"]["passed"] is False
    assert rows["H3"]["status"] == "not tested" and rows["E1"]["status"] == "exploratory"
    with pytest.raises(FileNotFoundError):
        pr.seal(_spec(), files=["nope.py"], root=str(tmp_path))


def test_gaia_record_is_compatible():
    """The deposited Gaia record verifies with the generic hash."""
    record = pr.load(os.path.join(ROOT, "docs", "preregistration", "preregistration_gaia_dr4.json"))
    assert pr.record_hash(record) == record["record_sha256"]


def test_cli(tmp_path, capsys):
    (tmp_path / "m.py").write_text("a = 1\n")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(dict(_spec(), files=["m.py"])))
    out = tmp_path / "rec.json"
    assert cli.main(["preregister", str(spec), "--root", str(tmp_path), "--out", str(out)]) == 0
    assert "record_sha256" in capsys.readouterr().out
    outcomes = tmp_path / "out.json"
    outcomes.write_text(json.dumps({"rate": 0.01, "bias": 0.0, "power": 0.9}))
    assert cli.main(["check-preregistration", str(out), "--root", str(tmp_path), "--outcomes", str(outcomes)]) == 0
    text = capsys.readouterr().out
    assert "| H1 | confirmatory | passed" in text and "unchanged" in text
    outcomes.write_text(json.dumps({"rate": 0.5}))
    assert cli.main(["check-preregistration", str(out), "--root", str(tmp_path), "--outcomes", str(outcomes),
                     "--json"]) == 1


def test_numerical_environment_is_recorded(tmp_path):
    from scidatasim.engine import SimulatorEngine
    from scidatasim.provenance import numerical_environment
    info = numerical_environment()
    assert "machine" in info and info.get("blas", {}).get("name")
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(dict(
        experiment_type="scidatasim.experiments.linear_function.logic.LinearFunctionExperiment", n_points=5,
        m=1.0, c=0.0, x_min=0.0, x_max=1.0, seed=1, generate_plots=False))
    record = json.load(open(os.path.join(run.experiment_dir, "experiment_record.json")))
    assert "numerics" in json.dumps(record)
    assert "numerics" in pr.seal({"title": "t"}, files=[], root=str(tmp_path))["provenance"]


def test_blas_reproducibility_tool(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("blas", os.path.join(ROOT, "tools", "check_blas_reproducibility.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    code = tool.main(["--kernels", "Haswell", "Nehalem", "--models", "linear", "astrometry", "--out", str(tmp_path)])
    rows = {r["output"]: r["status"] for r in json.load(open(tmp_path / "blas_reproducibility.json"))["rows"]}
    assert code == 0 and rows["linear: data"] == "bit-identical"
    assert rows["astrometry: estimate"] in ("bit-identical", "equal to rtol")
