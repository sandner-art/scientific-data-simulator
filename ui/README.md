# Scientific Data Simulator UI

A browser UI for [Scientific Data Simulator](../README.md), for convenience and an overview of
what the simulator can do. It is a separate package that uses only the simulator's public
functions. Everything it does is also available from the command line, and every run it starts
is recorded exactly like `sds examples run`, including the command that repeats it.

## Install and start

From the repository folder:

```bash
bash scripts/install.sh ui        # Linux, macOS     (Windows: scripts\install.cmd ui)
sds ui                            # or: bash scripts/start-ui.sh   (Windows: scripts\start-ui.cmd)
```

Or into an existing environment: `pip install -e ./ui`, then `sds ui`. Your browser opens at
http://127.0.0.1:8501. Options: `sds ui --port 8600 --runs-dir my_runs`.

By default the UI listens on this computer only. Anyone who can open it can run the examples
with any parameters, fill the disk with results, and (in example 12) make Python import a
module by name. `--address 0.0.0.0` exposes it to the network: do that only on a network you
trust. The Results page shows only the runs folder given at start.

## Pages

- **Examples**: every example, from its `example.yaml`, searchable by text, field and level, with
  the optional packages each one needs.
- **Run**: a form built from the example's parameters (types, ranges and choices come from the
  manifest), a quick mode, a time limit, a live log, then the outputs (figures, tables, reports,
  JSON) and the command that repeats the run.
- **Results**: past runs in the runs folder (default `sds_runs` in the folder where the UI was
  started). Click a row to open a run: its figures, parameters, the command that repeats it, the
  log and the other outputs. If an example's manifest lists no figures, the figures found in the
  run folder are shown.
  Interactive plots load only when you ask, since each is several megabytes.
- **Installation**: the `sds doctor` check, with the fix for anything missing.

New examples appear automatically once they have a manifest; the UI needs no change.

## Development

The UI is tested headless with Streamlit's AppTest (`pytest ui/tests`). The tests open every page
and run a real example through the form. CI runs them on Linux, Windows and macOS. Its version
always equals the simulator's, and the simulator never imports it (both checked by
`tests/test_ui_boundary.py`). It needs Python 3.11 or newer and Streamlit 1.67 or newer (the first that accepts Pillow 13).
