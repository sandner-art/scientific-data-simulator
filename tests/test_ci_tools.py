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
