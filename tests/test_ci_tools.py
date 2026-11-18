# tests/test_ci_tools.py - the CI helper that writes failed tests to the run summary
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tools", "ci_failures.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ci_failures  # noqa: E402

SAMPLE = '''
import pytest

def test_ok():
    pass

def test_plain_failure():
    assert 1 + 1 == 3, "arithmetic"

class TestGroup:
    def test_in_class(self):
        raise ValueError("```not a fence```")

@pytest.fixture
def broken():
    raise RuntimeError("fixture broke")

def test_error(broken):
    pass

@pytest.mark.skip
def test_skipped():
    pass
'''


def _report(tmp_path):
    (tmp_path / "test_sample.py").write_text(SAMPLE, encoding="utf-8")
    xml = tmp_path / "report.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
                    f"--junitxml={xml}", str(tmp_path / "test_sample.py")], cwd=tmp_path, capture_output=True)
    return str(xml)


def test_summary_lists_only_the_failures(tmp_path):
    text = ci_failures.main(_report(tmp_path), "pytest (windows-latest, 3.12)")
    assert text.startswith("### ❌ pytest (windows-latest, 3.12) - 3 failed, 1 passed, 1 skipped")
    assert "**test_sample.py::test_plain_failure** (failed)" in text
    assert "**test_sample.py::TestGroup::test_in_class** (failed)" in text
    assert "**test_sample.py::test_error** (error)" in text and "fixture broke" in text
    assert "test_ok" not in text and "test_skipped" not in text
    assert "````text" in text  # the traceback's ``` cannot close the block early


def test_long_traceback_keeps_its_end(tmp_path):
    xml = tmp_path / "r.xml"
    trace = "\n".join(f"line {i}" for i in range(200))
    xml.write_text(f'<testsuite tests="1" failures="1" errors="0" skipped="0"><testcase classname="tests.test_a" '
                   f'name="test_b"><failure message="boom">{trace}</failure></testcase></testsuite>', encoding="utf-8")
    text = ci_failures.main(str(xml), "job")
    assert "line 199" in text and "line 100" not in text and "…" in text
    assert "**tests/test_a.py::test_b** (failed)" in text


def test_missing_report_says_the_tests_did_not_run(tmp_path):
    text = ci_failures.main(str(tmp_path / "nope.xml"), "job")
    assert "No test report" in text and "Install step" in text


def test_writes_utf8_even_on_a_cp1252_console(tmp_path):
    """Windows runners redirect stdout as cp1252, which cannot encode the summary's symbols."""
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    result = subprocess.run([sys.executable, SCRIPT, _report(tmp_path), "job"], capture_output=True, env=env)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert "❌" in result.stdout.decode("utf-8")


def test_readme_figures_still_build(tmp_path):
    """tools/make_readme_figures.py uses the public API; the cheap figures must keep building."""
    env = dict(os.environ, SDS_FIGURES_OUT=str(tmp_path))
    result = subprocess.run([sys.executable, "-W", "ignore", os.path.join(ROOT, "tools", "make_readme_figures.py"),
                             "--only", "loophole", "series", "fields"], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr[-2000:]
    assert {"scorer_loophole.png", "series_operators.png", "field_operators.png"} <= set(os.listdir(tmp_path))


def test_check_scripts_run_on_a_small_scale(tmp_path):
    """The check scripts behind the verification note (tools/check_*.py) keep working."""
    import check_env_tolerance
    import check_formula_scoring
    rows = check_formula_scoring.family_rows(("power_law", 1, 1))
    assert len(rows) == 6 and {"poly_old", "poly_current"} <= set(rows[0])
    text = check_formula_scoring.markdown(check_formula_scoring.summarise(rows), len(rows), 1, 1, "test")
    assert "| 40 dB |" in text and "power_law" in text
    episodes = check_env_tolerance.play(40, 1)
    assert episodes[0]["tolerance"] >= check_env_tolerance.FIXED
    assert "| 40 dB |" in check_env_tolerance.markdown({"40": check_env_tolerance.summarise(episodes)}, "test")


def test_audit_calibration_plants_and_catches(tmp_path):
    """tools/check_audit_calibration.py plants each leak kind; the audit catches the ones it is meant to."""
    import numpy as np
    import pandas as pd
    import check_audit_calibration as cal
    from scidatasim.audit import audit_export
    from scidatasim.doe import run_design
    base = {"experiment_type": "scidatasim.experiments.oscillator.logic.OscillatorExperiment", "n_steps": 60,
            "dt": 0.05, "mass": 1.0, "damping": 0.3, "initial_position": 1.0, "initial_velocity": 0.0,
            "generate_plots": False, "bundle": True}
    design = run_design(base, pd.DataFrame({"stiffness": [2.0, 6.0]}), output_dir=str(tmp_path / "d"),
                        base_seed=3).output_dir
    values = cal.answers(design, 2, np.random.default_rng(0))
    assert len(values) == 2
    for kind, spec in cal.LEAKS.items():
        folder = tmp_path / kind.replace(" ", "_").replace(",", "")
        os.makedirs(folder)
        cal.plant(str(folder), kind, values)
        report = audit_export(str(folder), design, check_arrays=False)
        flagged = any(f.attack in cal.FLAG_ATTACKS for f in report.findings)
        assert flagged == spec["meant"], kind
