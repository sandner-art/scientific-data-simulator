# sds_ui/app.py
"""Streamlit app: examples gallery, run form with live log, results browser, installation check.

Start it with ``sds ui`` (or ``python -m sds_ui``). Every page reads the example manifests, so new
examples appear without changes here.
"""
import json
import os
import time

import pandas as pd
import streamlit as st

from sds_ui import AUTHOR, LICENSE, REPOSITORY
from sds_ui import __version__ as UI_VERSION
from sds_ui import core

PAGES = ("Examples", "Run", "Results", "Installation")
STATUS_ICON = {"ok": "✅", "info": "ℹ️", "warn": "⚠️", "fail": "❌"}
MAX_IMAGES = 6
MAX_HTML_BYTES = 20 * 1024 * 1024

st.set_page_config(page_title="Scientific Data Simulator", layout="wide")


@st.cache_data(ttl=30, show_spinner=False)
def _examples_cached(folder):
    return core.load_examples(folder)


def examples():
    try:
        return _examples_cached(os.environ.get("SDS_EXAMPLES") or None)
    except FileNotFoundError as e:
        st.error(f"{e}")
        st.stop()


def go(page, **state):
    """Button callback: switch page (callbacks run before the widgets are drawn again)."""
    st.session_state.update(state)
    if "example" in state:
        st.session_state["run_example_select"] = state["example"]
    st.session_state["page"] = page


# ---------------------------------------------------------------------------- output rendering
def show_outputs(outputs, key, folder=None):
    images = [o for o in outputs if o["kind"] == "image"]
    if not images and folder:  # the manifest lists no figures: show the ones the run made
        images = core.figures_in(folder)
    if not outputs and not images:
        st.caption("No outputs listed by the example's manifest.")
        return
    others = [o for o in outputs if o["kind"] != "image"]
    if images:
        cols = st.columns(min(3, len(images)))
        for i, o in enumerate(images[:MAX_IMAGES]):
            with cols[i % len(cols)]:
                try:
                    st.image(o["path"], caption=os.path.basename(o["path"]))
                except Exception:
                    st.caption(f"(cannot display) {o['path']}")
        if len(images) > MAX_IMAGES:
            st.caption(f"{len(images) - MAX_IMAGES} more image(s) in the output folder.")
    for i, o in enumerate(others[:10]):
        with st.expander(f"{o['kind']}: {os.path.basename(o['path'])}", expanded=(o["kind"] == "markdown" and i == 0)):
            st.caption(o["path"])
            try:
                if o["kind"] == "markdown":
                    with open(o["path"], encoding="utf-8") as f:
                        st.markdown(f.read())
                elif o["kind"] == "table":
                    st.dataframe(pd.read_csv(o["path"], nrows=500, encoding_errors="replace"))
                elif o["kind"] == "json":
                    with open(o["path"], encoding="utf-8") as f:
                        st.json(json.load(f), expanded=False)
                elif o["kind"] == "html":
                    # interactive plots are megabytes each: load one only when asked
                    size = os.path.getsize(o["path"])
                    if size > MAX_HTML_BYTES:
                        st.caption(f"{size / 1e6:.0f} MB: open it from the folder in a browser.")
                    elif st.toggle("Show interactive plot", key=f"html_{key}_{i}"):
                        # a local file the simulator wrote (st.iframe reads HTML files directly)
                        from pathlib import Path
                        st.iframe(Path(o["path"]), height=480)
                elif o["kind"] == "folder":
                    st.write(sorted(os.listdir(o["path"]))[:50])
                else:
                    st.code(core.tail(o["path"], 100), language=None)
            except Exception as e:  # one unreadable file must not break the page
                st.warning(f"Could not show this file: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------- pages
def page_examples():
    st.header("Examples")
    found = examples()
    all_fields = sorted({f for e in found.values() for f in e.fields})
    c1, c2, c3 = st.columns([2, 2, 1])
    text = c1.text_input("Search", key="search")
    fields = c2.multiselect("Fields", all_fields, key="fields_filter")
    level = c3.selectbox("Level", ["any", "student", "research"], key="level_filter")
    shown = core.filter_examples(found, text, fields, level)
    st.caption(f"{len(shown)} of {len(found)} examples")
    for e in shown:
        with st.container():
            left, right = st.columns([5, 1])
            left.subheader(e.title)
            runtime = e.runtime.get("full") or ""
            left.caption(f"{e.name} · {e.level} · {', '.join(e.fields)}" + (f" · about {runtime}" if runtime else ""))
            left.write(e.summary)
            missing = core.missing_extras(e)
            if missing:
                left.warning("Needs: " + ", ".join(missing) + f'. Install with: pip install -e ".[{",".join(missing)}]"')
            right.button("Open", key=f"open_{e.name}", on_click=go, args=("Run",), kwargs={"example": e.name})
            st.divider()


def page_run():
    st.header("Run an example")
    found = examples()
    names = list(found)
    if st.session_state.get("run_example_select") not in found:  # first visit, or a removed example
        st.session_state["run_example_select"] = names[0]
    name = st.selectbox("Example", names, format_func=lambda n: f"{found[n].title} ({n})",
                        key="run_example_select")
    example = found[name]
    st.write(example.summary)
    missing = core.missing_extras(example)
    if missing:
        st.warning("This example needs: " + ", ".join(missing) + ". It may fail without them.")

    live = st.session_state.get("live")
    running = live is not None and not live.done

    with st.form(key=f"form_{name}"):
        raw = {}
        for p in example.params:
            spec = core.widget_spec(p)
            key = f"p_{name}_{p.name}"
            if spec.kind == "checkbox":
                raw[p.name] = st.checkbox(spec.label, value=spec.default, help=spec.help or None, key=key)
            elif spec.kind == "select":
                raw[p.name] = st.selectbox(spec.label, spec.options, index=spec.options.index(spec.default)
                                           if spec.default in spec.options else 0, help=spec.help or None, key=key)
            elif spec.kind == "number":
                kwargs = {"value": spec.default, "help": spec.help or None, "key": key}
                if spec.min is not None:
                    kwargs["min_value"] = type(spec.default)(spec.min)
                if spec.max is not None:
                    kwargs["max_value"] = type(spec.default)(spec.max)
                if spec.step is not None:
                    kwargs["step"] = type(spec.default)(spec.step)
                raw[p.name] = st.number_input(spec.label, **kwargs)
            else:
                placeholder = "(example's default)" if spec.optional else None
                raw[p.name] = st.text_input(spec.label, value=spec.default, help=spec.help or None,
                                            placeholder=placeholder, key=key)
        quick = st.checkbox("Quick mode (the example's fast settings)", value=bool(example.quick_args),
                            disabled=not example.quick_args, key=f"quick_{name}")
        timeout = st.number_input("Stop after (seconds, 0 = no limit)", min_value=0, value=600, step=60,
                                  key=f"timeout_{name}")
        submitted = st.form_submit_button("Run", disabled=running)

    if submitted and not running:
        try:
            values = core.form_values(example, raw)
        except ValueError as e:
            st.error(str(e))
        else:
            st.session_state["live"] = core.start_run(example, values, quick, timeout=timeout or None)
            live = st.session_state["live"]

    if live is None:
        st.caption(f"Runs are saved under {core.runs_dir()}.")
        return
    st.subheader(f"{live.example.title}")
    st.caption(live.output_dir)
    if not live.done:  # draw one frame, then rerun: the page stays responsive and nothing goes stale
        st.info("Running ...")
        st.code("\n".join(live.lines[-200:]) or "(starting)", language=None)
        time.sleep(0.5)
        st.rerun()
    st.code("\n".join(live.lines[-200:]) or "(no output)", language=None)
    if live.error:
        st.error(live.error)
        return
    result = live.result
    status_box = st.empty()
    if result.ok:
        status_box.success(f"Done in {result.seconds:.1f} s.")
    elif result.returncode is None:
        status_box.error(f"Stopped after {result.seconds:.0f} s (time limit).")
    else:
        status_box.error(f"Failed (exit code {result.returncode}). The log above shows why.")
    st.markdown("**Repeat from a terminal**")
    st.code(core.repeat_command(result.cwd, os.path.dirname(os.path.dirname(live.example.path)),
                                result.command_line), language=None)
    show_outputs(result.outputs, key="live", folder=result.output_dir)


def page_results():
    st.header("Results")
    folder = core.runs_dir()  # fixed (choose it with: sds ui --runs-dir DIR), never an arbitrary path
    st.caption(f"Runs folder: {folder}")
    records = core.list_runs(folder)
    if not records:
        st.info("No runs here yet. Start one on the Run page.")
        return
    def short(path):
        try:
            return os.path.relpath(path, folder)
        except ValueError:  # another drive on Windows
            return path
    table = pd.DataFrame([{"example": r.example, "finished": r.finished, "status": r.status,
                           "seconds": r.seconds, "quick": r.quick, "folder": short(r.folder)}
                          for r in records])
    st.caption("Click a row to open the run (or choose it below).")
    event = st.dataframe(table, on_select="rerun", selection_mode="single-row", hide_index=True,
                         key="results_table")
    rows = list(getattr(getattr(event, "selection", None), "rows", None) or [])
    if rows and rows[0] < len(records) and st.session_state.get("_results_row") != rows[0]:
        st.session_state["_results_row"] = rows[0]  # a new click: open that run
        st.session_state["result_pick"] = rows[0]
    if st.session_state.get("result_pick", 0) >= len(records):
        st.session_state["result_pick"] = 0
    labels = [f"{r.finished} · {r.example} · {r.status}" for r in records]
    choice = st.selectbox("Open a run", range(len(records)), format_func=lambda i: labels[i], key="result_pick")
    record = records[choice]
    st.subheader(f"{record.title or record.example} · {record.status}")
    c1, c2 = st.columns(2)
    c1.markdown("**Parameters**")
    c1.json(record.parameters, expanded=True)
    c2.markdown("**Repeat from a terminal**")
    c2.code(core.repeat_command(record.cwd, record.pythonpath, record.command_line), language=None)
    with st.expander("Log (last lines)"):
        st.code(core.tail(os.path.join(record.folder, "run.log")), language=None)
    show_outputs(core.outputs_of(record, examples()), key="result", folder=record.folder)


def page_installation():
    st.header("Installation check")
    from simulator.doctor import run_checks
    if st.button("Run the check", key="doctor_run") or "doctor" not in st.session_state:
        import tempfile
        with st.spinner("Checking ..."):
            st.session_state["doctor"] = run_checks(os.path.join(tempfile.gettempdir(), "sds_doctor_probe"))
    checks = st.session_state["doctor"]
    for c in checks:
        st.markdown(f"{STATUS_ICON.get(c.status, '')} **{c.name}** — {c.detail}")
        if c.fix:
            st.caption(f"{'Add it if you need it' if c.status == 'info' else 'Fix'}: `{c.fix}`")
    failed = sum(c.status == "fail" for c in checks)
    (st.error if failed else st.success)(f"{failed} problem(s)." if failed else
                                         "The core works. Optional parts (ℹ️) are needed only for what they name.")


# ---------------------------------------------------------------------------- layout
with st.sidebar:
    st.title("Scientific Data Simulator")
    st.caption(f"simulator {core.SIMULATOR_VERSION} · UI {UI_VERSION}")
    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]
    st.radio("Page", PAGES, key="page")
    st.caption("Everything here is also available from the command line: `sds examples ...`, `sds doctor`.")
    st.divider()
    cite = core.citation()
    st.caption(f"© {AUTHOR} · {LICENSE} · [source code]({REPOSITORY})")
    if cite.get("title"):
        with st.expander("How to cite"):
            text = (f"{cite['authors']}. *{cite['title']}*. Version {cite['version']}. "
                    + (f"https://doi.org/{cite['doi']}" if cite.get("doi") else cite.get("url", "")))
            st.markdown(text)
            st.caption("Details and BibTeX: CITATION.cff in the repository.")

{"Examples": page_examples, "Run": page_run, "Results": page_results,
 "Installation": page_installation}[st.session_state["page"]]()
