# ui/tests/test_ui.py
"""The UI, run headless with Streamlit's AppTest: every page, and a real example run."""
import json
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sds_ui import APP, core  # noqa: E402
from simulator.examples import discover  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES = os.path.join(ROOT, "examples")


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("SDS_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("SDS_EXAMPLES", EXAMPLES)
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _page(at, name):
    at.sidebar.radio(key="page").set_value(name).run()
    assert not at.exception, at.exception
    return at


# ------------------------------------------------------------------ pure helpers
def test_widget_specs_cover_every_manifest_parameter():
    for example in discover(EXAMPLES).values():
        for p in example.params:
            spec = core.widget_spec(p)
            assert spec.kind in ("number", "text", "checkbox", "select")
            if spec.kind == "number":
                assert p.default is not None and (p.min is None or p.min <= spec.default)


def test_form_values_keep_only_changes_and_validate():
    ex = discover(EXAMPLES)["example_10_law_discovery"]
    assert core.form_values(ex, {"replicates": 1, "suite_seed": "", "export": False}) == {}
    assert core.form_values(ex, {"replicates": 3, "suite_seed": "7", "export": True}) == \
        {"replicates": 3, "suite_seed": 7, "export": True}
    with pytest.raises(ValueError, match="replicates"):
        core.form_values(ex, {"replicates": 99})


def test_filter_and_missing_extras():
    found = discover(EXAMPLES)
    assert {e.name for e in core.filter_examples(found, level="student")} >= {"example_1", "example_doe_1"}
    assert all("geometry" in e.fields for e in core.filter_examples(found, fields=["geometry"]))
    assert [e.name for e in core.filter_examples(found, text="law discovery")][0] == "example_10_law_discovery"
    assert isinstance(core.missing_extras(found["example_8_geometry_benchmark"]), list)


def test_live_run_and_listing(tmp_path, monkeypatch):
    monkeypatch.setenv("SDS_RUNS", str(tmp_path))
    ex = discover(EXAMPLES)["example_4"]
    live = core.start_run(ex, {}, quick=True)
    live.thread.join(60)
    assert live.done and live.result.ok and any("Experiment completed" in l for l in live.lines)
    records = core.list_runs()
    assert len(records) == 1 and records[0].example == "example_4" and records[0].status == "ok"
    outputs = core.outputs_of(records[0], discover(EXAMPLES))
    assert outputs and all(o["kind"] in ("image", "html", "json") for o in outputs)
    assert core.list_runs(str(tmp_path / "missing")) == []


# ------------------------------------------------------------------ pages
def test_examples_page_lists_and_filters(app):
    titles = [s.value for s in app.subheader]
    assert len(titles) == len(discover(EXAMPLES))
    app.text_input(key="search").set_value("geometry").run()
    assert [s.value for s in app.subheader] == ["Geometry benchmark"]


def test_open_button_goes_to_the_run_form(app):
    app.button(key="open_example_8_geometry_benchmark").click().run()
    assert not app.exception and not app.warning, [w.value for w in app.warning]
    assert app.session_state["page"] == "Run"
    assert app.selectbox(key="run_example_select").value == "example_8_geometry_benchmark"
    assert app.number_input(key="p_example_8_geometry_benchmark_n").value == 12


def test_run_page_runs_an_example_and_shows_outputs(app, tmp_path):
    _page(app, "Run")
    app.selectbox(key="run_example_select").set_value("example_4").run()
    app.button[0].click().run(timeout=120)  # the form's Run button
    assert not app.exception, app.exception
    assert any("Done in" in s.value for s in app.success)
    assert any("examples.example_4.run_experiment" in c.value and "PYTHONPATH" in c.value for c in app.code)
    assert not app.warning, [w.value for w in app.warning]
    labels = [e.label for e in app.expander]
    assert "json: ground_truth.json" in labels and any(l.startswith("html:") for l in labels)
    assert len(app.get("imgs")) + len(app.get("image")) >= 1  # the figures (element name varies by version)
    runs = core.list_runs(str(tmp_path / "runs"))
    assert len(runs) == 1 and runs[0].status == "ok"


def test_invalid_form_value_is_reported(app, tmp_path):
    _page(app, "Run")
    app.selectbox(key="run_example_select").set_value("example_10_law_discovery").run()
    app.text_input(key="p_example_10_law_discovery_suite_seed").set_value("not a number")
    app.button[0].click().run()
    assert not app.exception
    assert any("suite_seed" in e.value for e in app.error)
    assert core.list_runs(str(tmp_path / "runs")) == []  # nothing was started


def test_results_page_shows_past_runs(app, tmp_path):
    folder = tmp_path / "runs" / "example_4" / "x"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"example": "example_4", "title": "Linear function",
                                                 "finished": "2026-01-01T00:00:00", "returncode": 1,
                                                 "seconds": 1.0, "parameters": {}, "command_line": "python -m x",
                                                 "outputs": []}))
    (folder / "run.log").write_text("Traceback: boom\n")
    _page(app, "Results")
    assert len(app.dataframe) >= 1
    assert any("boom" in c.value for c in app.code)


def test_corrupt_run_records_are_skipped(app, tmp_path):
    runs = tmp_path / "runs"
    for i, text in enumerate(['[1, 2]', '"str"', '{"seconds": "abc"}', '{"finished": null, "example": "x"}', "{bad"]):
        folder = runs / f"r{i}"
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(text)
    records = core.list_runs(str(runs))
    assert [r.example for r in records] == ["x"]
    _page(app, "Results")
    assert not app.exception


def test_two_runs_in_the_same_second_get_their_own_folders(tmp_path, monkeypatch):
    monkeypatch.setenv("SDS_RUNS", str(tmp_path))
    ex = discover(EXAMPLES)["example_4"]
    a, b = core.start_run(ex, {}, True), core.start_run(ex, {}, True)
    a.thread.join(60)
    b.thread.join(60)
    assert a.output_dir != b.output_dir and len(core.list_runs(str(tmp_path))) == 2


def test_repeat_command():
    posix = core.repeat_command("/my runs/x", "/repo", "python -m examples.e.run", windows=False)
    assert posix == "cd '/my runs/x' && PYTHONPATH=/repo python -m examples.e.run"
    win = core.repeat_command(r"C:\my runs", r"C:\repo", "python -m e", windows=True)
    assert win.splitlines() == ['cd /d "C:\\my runs"', 'set "PYTHONPATH=C:\\repo"', "python -m e"]


def test_results_page_empty(app):
    _page(app, "Results")
    assert any("No runs here yet" in i.value for i in app.info)


def test_installation_page(app):
    _page(app, "Installation")
    text = " ".join(m.value for m in app.markdown)
    assert "test simulation" in text and "python" in text
    assert app.success or app.error


def test_figures_fallback_and_citation(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "fig.png").write_bytes(b"\x89PNG")
    (tmp_path / "notes.txt").write_text("x")
    assert [os.path.basename(f["path"]) for f in core.figures_in(str(tmp_path))] == ["fig.png"]
    cite = core.citation()
    assert "Sandner" in cite["authors"] and cite["title"] and cite["version"]


def test_sidebar_credits(app):
    text = " ".join(c.value for c in app.sidebar.caption)
    assert "Daniel Sandner" in text and "Apache-2.0" in text
