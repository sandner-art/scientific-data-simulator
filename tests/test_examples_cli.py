# tests/test_examples_cli.py
import json
import os
import re
import subprocess
import sys

import pytest

import simulator
from simulator import cli, doctor
from simulator.examples import (Example, ManifestError, Param, build_command, check_all, default_examples_dir,
                                discover, load_manifest, run_example)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")


def test_every_example_has_a_valid_manifest():
    assert check_all(EXAMPLES) == []
    found = discover(EXAMPLES)
    folders = [d for d in os.listdir(EXAMPLES) if os.path.isfile(os.path.join(EXAMPLES, d, "run_experiment.py"))]
    assert sorted(found) == sorted(folders)
    names = list(found)
    assert names.index("example_2") < names.index("example_10_law_discovery")  # natural order
    assert default_examples_dir() == EXAMPLES


def test_manifest_validation(tmp_path):
    folder = tmp_path / "examples" / "ex"
    folder.mkdir(parents=True)
    (folder / "example.yaml").write_text("title: T\nsummary: S\nentry: examples.ex.run\nlevel: expert\n"
                                         "colour: red\nparams:\n  - {name: n, arg: --n, type: int, default: x}\n")
    with pytest.raises(ManifestError) as err:
        load_manifest(str(folder))
    text = str(err.value)
    for part in ("level must be", "unknown key 'colour'", "params[0] default", "entry module not found"):
        assert part in text
    assert "ex: no example.yaml" not in check_all(str(tmp_path / "examples"))


def test_param_coercion_and_args():
    n = Param("n", "--n", "int", 3, min=1, max=10)
    assert n.coerce("4") == 4 and n.to_args(4) == ["--n", "4"]
    with pytest.raises(ValueError):
        n.coerce("11")
    with pytest.raises(ValueError):
        n.coerce(2.5)
    flag = Param("f", "--f", "flag", False)
    assert flag.coerce("yes") is True and flag.to_args(True) == ["--f"] and flag.to_args(False) == []
    with pytest.raises(ValueError):
        flag.coerce("maybe")
    choice = Param("c", "--c", "choice", "a", choices=["a", 2])
    assert choice.coerce("2") == 2
    with pytest.raises(ValueError):
        choice.coerce("z")
    assert Param("s", "--s", "str", None).to_args(None) == []


def test_build_command_and_unknown_parameter():
    ex = discover(EXAMPLES)["example_10_law_discovery"]
    cmd = build_command(ex, {"replicates": "2", "export": True}, quick=False, output_dir="OUT", python="py")
    assert cmd[:5] == ["py", "-m", "examples.example_10_law_discovery.run_experiment", "--output_dir", "OUT"]
    assert "--export" in cmd and cmd[cmd.index("--replicates") + 1] == "2"
    with pytest.raises(KeyError):
        run_example(ex, {"nonexistent": 1})


def test_run_example_records_the_run(tmp_path):
    ex = discover(EXAMPLES)["example_4"]
    lines = []
    run = run_example(ex, quick=True, output_dir=str(tmp_path / "run"), on_line=lines.append)
    assert run.ok and run.seconds > 0 and any("Experiment completed" in l for l in lines)
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["returncode"] == 0 and record["simulator_version"] == simulator.__version__
    assert record["command"][2] == "examples.example_4.run_experiment"
    assert any(o.endswith(".png") for o in record["outputs"])
    assert (tmp_path / "run" / "run.log").read_text().count("Experiment completed. ID") == 1
    assert len({o["path"] for o in run.outputs}) == len(run.outputs)


def test_run_example_timeout(tmp_path):
    folder = tmp_path / "examples" / "slow"
    folder.mkdir(parents=True)
    (folder / "run.py").write_text("import time\ntime.sleep(30)\n")
    (folder / "example.yaml").write_text("title: Slow\nsummary: sleeps\nentry: examples.slow.run\noutput_arg: null\n")
    run = run_example(load_manifest(str(folder)), output_dir=str(tmp_path / "out"), timeout=1)
    assert run.returncode is None and not run.ok and run.seconds < 15
    assert "stopped after" in (tmp_path / "out" / "run.log").read_text()


def test_cli_list_show_run_check(tmp_path, capsys):
    assert cli.main(["examples", "list", "--level", "student"]) == 0
    out = capsys.readouterr().out
    assert "example_1 " in out and "example_11_testbed" not in out
    assert cli.main(["examples", "list", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == len(discover(EXAMPLES))
    assert cli.main(["examples", "show", "example_8_geometry_benchmark"]) == 0
    assert "--no-export" in capsys.readouterr().out
    assert cli.main(["examples", "show", "nope"]) == 2
    assert cli.main(["examples", "run", "example_4", "--quick", "--silent", "--output", str(tmp_path / "r")]) == 0
    assert "repeat with" in capsys.readouterr().out
    assert cli.main(["examples", "run", "example_8_geometry_benchmark", "--set", "n=zero"]) == 2
    assert cli.main(["examples", "check", "--run-quick", "--only", "example_4"]) == 0
    assert "[ ok ] example_4" in capsys.readouterr().out


def test_doctor(tmp_path, capsys):
    checks = doctor.run_checks(str(tmp_path / "out"))
    by_name = {c.name: c for c in checks}
    assert by_name["python"].status == doctor.OK and by_name["test simulation"].status == doctor.OK
    assert by_name["examples"].status == doctor.OK and "examples" in by_name["examples"].detail
    assert all(c.fix for c in checks if c.status != doctor.OK)
    assert doctor.main(["--json", "--output_dir", str(tmp_path / "o")]) == 0
    assert {c["name"] for c in json.loads(capsys.readouterr().out)} >= {"python", "core packages"}
    assert cli.main(["doctor", "--output_dir", str(tmp_path / "o")]) == 0
    assert "Ready." in capsys.readouterr().out


def test_python_dash_m_simulator(tmp_path):
    out = subprocess.run([sys.executable, "-m", "simulator", "examples", "list"], cwd=ROOT, capture_output=True,
                         text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert out.returncode == 0 and "example_1" in out.stdout


def test_version_follows_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        version = re.search(r'^version = "([^"]+)"', f.read(), flags=re.M).group(1)
    assert simulator.__version__ == version
    with open(os.path.join(ROOT, "CITATION.cff")) as f:
        assert f"version: {version}" in f.read()


def test_install_scripts_are_consistent():
    import shutil
    scripts = os.path.join(ROOT, "scripts")
    bash = shutil.which("bash")
    for name in ("install.sh", "update.sh"):
        if os.name != "nt" and bash:  # on Windows, "bash" may be WSL's, not Git's
            assert subprocess.run([bash, "-n", os.path.join(scripts, name)]).returncode == 0
            assert os.access(os.path.join(scripts, name), os.X_OK)
    for name in ("install", "update"):
        ps1 = open(os.path.join(scripts, f"{name}.ps1"), encoding="utf-8").read()
        cmd = open(os.path.join(scripts, f"{name}.cmd"), encoding="utf-8", newline="").read()
        assert "$Profile" not in ps1  # collides with PowerShell's automatic $PROFILE
        assert "-Setup" in cmd and f"{name}.ps1" in cmd and "-ExecutionPolicy Bypass" in cmd
    # line endings come from .gitattributes on checkout (a working tree written by git am may lag)
    with open(os.path.join(ROOT, ".gitattributes")) as f:
        attributes = f.read()
    assert "*.sh   text eol=lf" in attributes and "*.cmd  text eol=crlf" in attributes
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        text = f.read()
    for profile in ("standard", "all", "dev"):
        assert re.search(rf"^{profile} = \[", text, flags=re.M)
    assert 'sds = "simulator.cli:main"' in text


# ------------------------------------------------------------------ review regressions
def _write_example(tmp_path, name, script, manifest):
    folder = tmp_path / "examples" / name
    folder.mkdir(parents=True)
    (folder / "run.py").write_text(script, encoding="utf-8")
    (folder / "example.yaml").write_text(manifest, encoding="utf-8")
    return load_manifest(str(folder))


def test_timeout_stops_grandchildren(tmp_path):
    marker = tmp_path / "grandchild_alive"
    child = f"import time; time.sleep(4); open({str(marker)!r}, 'w')"
    script = f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {child!r}])\ntime.sleep(60)\n"
    ex = _write_example(tmp_path, "tree", script, "title: T\nsummary: S\nentry: examples.tree.run\noutput_arg: null\n")
    run = run_example(ex, output_dir=str(tmp_path / "out"), timeout=1)
    assert run.returncode is None and run.seconds < 5
    import time
    time.sleep(5)
    assert not marker.exists()


def test_failing_callback_and_non_ascii_output_do_not_stall(tmp_path):
    script = "print('sigma: \\u03c3 ' * 2000)\nprint('done')\n"
    ex = _write_example(tmp_path, "uni", script, "title: T\nsummary: S\nentry: examples.uni.run\noutput_arg: null\n")

    def broken(line):
        raise UnicodeEncodeError("cp1252", "x", 0, 1, "cannot encode")
    run = run_example(ex, output_dir=str(tmp_path / "out"), timeout=30, on_line=broken)
    assert run.ok and "done" in (tmp_path / "out" / "run.log").read_text(encoding="utf-8")


def test_user_values_win_over_quick_mode(tmp_path):
    ex = discover(EXAMPLES)["example_12_model_roles"]
    cmd = build_command(ex, {"episodes": 50}, quick=True, output_dir="OUT", python="py")
    assert cmd.count("--episodes") == 1 and cmd[cmd.index("--episodes") + 1] == "50"
    quick = build_command(ex, {}, quick=True, output_dir="OUT", python="py")
    assert quick[quick.index("--episodes") + 1] == "2"
    from simulator.examples import effective_values
    assert effective_values(ex, {}, quick=True)["episodes"] == 2
    testbed = discover(EXAMPLES)["example_11_testbed"]
    assert "--quick" in build_command(testbed, {}, quick=True, output_dir="OUT")


@pytest.mark.parametrize("text,problem", [
    ("- just\n- a list\n", "must be a mapping"),
    ("title: T\nsummary: S\nentry: examples.x.run\nruntime: 5 s\n", "'runtime' must be a mapping"),
    ("title: T\nsummary: S\nentry: examples.x.run\nquick_args: --quick\n", "'quick_args' must be a list"),
    ("title: T\nsummary: S\nentry: examples.x.run\noutput_arg: 5\n", "'output_arg' must be an option"),
    ("title: T\nsummary: S\nentry: examples.x.run\nparams:\n  - {name: n, arg: --n, type: int, default: true}\n",
     "expected a number"),
    ("title: [unclosed\n", "not valid YAML")])
def test_manifest_type_errors(tmp_path, text, problem):
    folder = tmp_path / "examples" / "x"
    folder.mkdir(parents=True)
    (folder / "run.py").write_text("")
    (folder / "example.yaml").write_text(text)
    with pytest.raises(ManifestError, match=problem):
        load_manifest(str(folder))
    assert discover(str(tmp_path / "examples")) == {}  # skipped, not crashing


def test_cli_errors_are_messages(tmp_path, capsys):
    assert cli.main(["examples", "check", "--run-quick", "--only", "nope"]) == 2
    assert "unknown example" in capsys.readouterr().err
    assert cli.main(["--examples-dir", str(tmp_path / "missing"), "examples", "list"]) == 2
    assert "examples folder not found" in capsys.readouterr().err


def test_sds_run_config(tmp_path, capsys):
    config = tmp_path / "c.yaml"
    config.write_text("experiment_type: experiments.example_random_walk.logic.RandomWalkExperiment\n"
                      "n_steps: 20\nstep_size: 1\nseed: 3\ngenerate_plots: false\n")
    assert cli.main(["run", str(config), "--output", str(tmp_path / "out")]) == 0
    assert "experiment" in capsys.readouterr().out and any((tmp_path / "out").iterdir())


def test_installed_sds_works_from_any_folder(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    probe = subprocess.run([sys.executable, "-c", "import simulator, experiments"], cwd=str(tmp_path), env=env,
                           capture_output=True)
    if probe.returncode != 0:
        pytest.skip("the package is not installed in this environment")
    out = subprocess.run([sys.executable, "-m", "simulator", "doctor", "--json"], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True)
    checks = {c["name"]: c for c in json.loads(out.stdout)}
    assert checks["test simulation"]["status"] == "ok", checks["test simulation"]
    assert list(tmp_path.iterdir()) == []  # the doctor left nothing behind


def test_python_and_security_floors_agree():
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        core = f.read()
    with open(os.path.join(ROOT, "ui", "pyproject.toml")) as f:
        ui = f.read()
    assert 'requires-python = ">=3.11"' in core and 'requires-python = ">=3.11"' in ui
    assert doctor.MIN_PYTHON == (3, 11)
    assert '"pillow>=13.0"' in core and '"pillow>=13.0"' in ui and '"streamlit>=1.67"' in ui
    for script in ("install.sh", "install.ps1"):
        with open(os.path.join(ROOT, "scripts", script), encoding="utf-8") as f:
            assert "(3, 11)" in f.read()
    by_name = {c.name: c for c in doctor.run_checks(os.path.join(ROOT, ".doctor_probe_test"))}
    assert by_name["security: pillow"].status in (doctor.OK, doctor.WARN)


def test_optional_extras_are_info_not_warnings(tmp_path):
    checks = doctor.run_checks(str(tmp_path / "o"))
    extras = [c for c in checks if c.name.startswith("extra: ")]
    assert extras and all(c.status in (doctor.OK, doctor.INFO) for c in extras)
    text = doctor.report([doctor.Check("extra: x", doctor.INFO, "not installed (optional)", "pip install x")])
    assert "[info]" in text and "Ready." in text and "optional part(s) not installed" in text


def test_prepare_output_dir_keeps_the_runner_files(tmp_path):
    from simulator.examples import prepare_output_dir
    out = tmp_path / "run"
    (out / "old" / "deep").mkdir(parents=True)
    (out / "old_report.md").write_text("stale")
    (out / "run.log").write_text("log so far")
    (out / "run.json").write_text("{}")
    assert prepare_output_dir(str(out)) == str(out)
    assert sorted(os.listdir(out)) == ["run.json", "run.log"]
    assert (out / "run.log").read_text() == "log so far"
    fresh = prepare_output_dir(str(tmp_path / "new" / "dir"))
    assert os.path.isdir(fresh) and os.listdir(fresh) == []


def test_examples_never_delete_their_output_folder():
    """The runner keeps run.log open inside the output folder: deleting the folder fails on
    Windows ('file in use') and silently loses the log elsewhere. Use prepare_output_dir."""
    offenders = []
    for root, _, files in os.walk(EXAMPLES):
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(root, name), encoding="utf-8") as f:
                    if "rmtree(" in f.read():
                        offenders.append(os.path.join(root, name))
    assert offenders == []


def test_runner_log_survives_an_example_that_cleans_its_folder(tmp_path):
    ex = discover(EXAMPLES)["example_12_model_roles"]
    run = run_example(ex, quick=True, output_dir=str(tmp_path / "out"), timeout=300)
    assert run.ok, (tmp_path / "out" / "run.log").read_text(encoding="utf-8")[-2000:]
    log = (tmp_path / "out" / "run.log").read_text(encoding="utf-8")
    assert log.strip(), "run.log was deleted while the example ran"
    assert (tmp_path / "out" / "run.json").exists()
