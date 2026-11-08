# Changelog

## 0.3.0 (unreleased)

### Added
- **Exporters** (`simulator/export/`):
  - `tasks`: bundle questions as scored tasks (`collect_bundle_tasks`, `write_tasks_jsonl`, `score` with free-text parsing).
  - `huggingface`: dataset folder with a dataset card and Croissant 1.0 metadata (`export_huggingface`, `write_croissant`).
    - Validated with mlcroissant and loaded with `datasets`.
    - Figures are embedded as images; the export is byte-reproducible.
    - Ground truth, answers and tolerances are withheld by default.
  - `sbi`: `BatchSimulator` (theta -> x with recorded seeds), `generate_training_data` and `training_data_from_runs` for simulation-based inference.
  - `rl`: `export_openenv` writes an OpenEnv 0.12 package that passes `openenv validate`; `export_verifiers_qa` writes a verifiers v1 taskset; `verifiers_openenv_taskset_config` covers multi-turn training.
- **`simulator.envs.ExperimentEnv`**: an interactive environment. An agent runs noisy experiments on a system with hidden parameters within a budget, then reports a numeric or categorical answer that is scored automatically.
- **`simulator.engine.simulate()`**: in-memory runs, with the same seeding, law shift and observation as the engine but no files written.
- **Physical units** (`simulator/units.py`, optional pint):
  - `parameter_units` on models; unit-bearing config values (`"2 kN/m"`) are converted to the model's declared units.
  - Output unit checks, with known traps flagged (`units` parses as micronit, `au` as astronomical unit).
  - `unit_check: warn | error | off`.
- **Model checker** (`simulator/checker.py`, also `python -m simulator.checker`):
  - a static scan of the source;
  - runtime checks: determinism, global random-state isolation, config isolation, finite outputs, roles, fidelity, units, convergence order under halving of `dt`, invariant drift, dead parameters;
  - `check_source` checks code in a separate, time-limited process.
- **LLM drafting** (`simulator/llm.py`): `draft_model` asks any LLM through `CallableLLMClient`, gates the answer with the checker and retries with its findings. Rejected drafts cannot be saved.
- **Models:**
  - `OscillatorExperiment`: damped / Duffing oscillator in SI units; closed form or DOP853 as `reference`, explicit Euler as a documented `audit` model.
  - `TriangleExperiment`: exact triangle constructions with diagram bundles and geometry questions.
- **Bundles:**
  - diagram figures from `figure_elements()`, drawn by a shipped script from `elements.csv`;
  - model-specific exact questions from `questions()`;
  - a `tolerance_rule` on every question; a `bundle` random stream.
- **Model hooks:** `invariants(results)`, `parameter_units`, `figure_elements()`, `figure_caption()`, `questions()`.
- **Examples:** `example_8_geometry_benchmark` (geometry benchmark with Hugging Face export) and `example_9_discovery_env` (interactive environment with a scripted baseline).
- **Extras:** `units`, `huggingface`, `validate`, `openenv`, `verifiers`. `static-plots` and `dev` require plotly >= 6.1.1, which silences the kaleido 'engine' deprecation warnings.

### Changed
- Public question files no longer contain the numeric `tolerance`. It is derived from the answer (5 % of a parameter) or from the data (3 % of a series' range), so it leaked them; it is now private like the answer.
- Unit labels of bundled models are now valid pint units: `units` -> `dimensionless`, `arbitrary` -> `arbitrary_unit`, Lotka-Volterra time -> `time_unit`, and the linear function's units are configurable via `x_units`/`y_units`.
- The engine's model execution is factored into `prepare_config`, `_execute_model`, `validate_results` and `resolve_experiment_type`, shared by `run_config` and `simulate`.

## 0.2.0 (unreleased)

The aim of this release is to make the generated data trustworthy as reference data. Every result can now be reproduced from a recorded seed, the measurement process is explicit and recorded, ground truth is stored separately from observations, and the bundled models are numerically verified.

### Added
- **Packaging and licence:** `pyproject.toml` (`pip install -e ".[dev]"`, Python ≥ 3.9) and the Apache-2.0 `LICENSE` and `NOTICE` files.
- **Continuous integration:** a GitHub Actions workflow runs the tests on Python 3.9–3.13 on Linux and 3.12 on Windows.
- **Seeds:** one recorded seed per run, set in the config or drawn and recorded. Independent `model` and `observation` random streams are derived from it, and models use `ExperimentLogic.rng`.
- **Provenance:** each record stores the config hash, the model's source hash, its git commit and dirty flag, the simulator version, the versions of all installed dependencies, and a `synthetic` flag.
- **Config validation** of the engine-level keys (pydantic).
- **Output roles:** `DataDescriptor.role` can be parameter, latent, label, observation, coordinate or auxiliary.
- **Observation operators** (`observation:`):
  - sampling: every k-th step, n points, or explicit indices;
  - noise: Gaussian (`sigma` or `snr_db`), multiplicative, Poisson;
  - quantisation, outliers, missing values and clipping;
  - custom operators via `register_operator`.
- **Ground truth** is written to `ground_truth.json`, and data files are grouped by role (`data/ground_truth`, `data/observation`, …).
- **Counterfactual law shifts** (`law_shift:`): the original and shifted values are both recorded.
- **Storage:** checksummed data files (`.npy`, HDF5, Parquet or CSV) replace data inlined in JSON. Checksums are verified on load, and arrays can be memory-mapped.
- **Design of experiments:**
  - `lhs`, `sobol`, `halton` and `random` designs, with log-scale and integer parameters;
  - replicates and per-row or common-random-number seeds;
  - dotted keys for nested settings;
  - `run_design`, which writes `design.csv`, `design_manifest.json` and `design_results.csv`;
  - parallel runs with `n_jobs`.
- **Predator–prey model options:** `dt`, `integrator: rk4`, `clamp_negative`, Holling type II predation and logistic prey.
- **Numerical verification tests** (`tests/test_verification.py`).
- **Multimodal run bundles** (`bundle:`, `simulator/bundle.py`):
  - The figure is rendered by executing the shipped `plot_figure.py` on `table.csv`, so code, data and image always agree.
  - Each bundle also holds a caption that reveals no answers, and questions with answers and tolerances computed from the exact arrays. The questions cover extremes, argmax, values, peak counts, comparisons and parameter estimates; each is marked as answerable from the figure or by inference.
  - It also holds the mechanism graph (JSON and Graphviz), a private `description.md`, and a manifest with checksums and public/private file labels.
  - Bundles are byte-for-byte reproducible from the seed.
  - `run_design` collects all runs' questions into `questions.jsonl` (public) and `qa.jsonl` (private). `check_answer` scores responses.
- **Model fidelity labels** (`reference`, `audit`, `unverified`, `observed`) with a reason, recorded in provenance and ground truth, with a warning for `audit` and `unverified` runs. The Lotka–Volterra 0.1.0 default is labelled `audit` and documented as a deliberately weak model.
- **Model hooks:** `ExperimentLogic.inferable_parameters` and `mechanism_graph()`, implemented for the bundled models.
- **Example 7:** a predator–prey reference dataset with an SNR × replicate design and an optional law shift.

### Fixed
- `DataDescriptor.to_dict` stored `"DataType.X"` instead of the type's value. Old records still load.
- Loaded records now rebuild NumPy arrays and DataFrames.
- `pkg_resources` was removed from recent setuptools, so the engine could not be imported; it now uses `importlib.metadata`.
- Importing the library no longer configures logging.
- Parameter sweeps bypassed the engine (no validation, plots or provenance). They now use `SimulatorEngine.run_config`.
- Validation rejected NumPy scalars and never checked shapes.
- `save_csv` crashed on mixed scalar and array results, and dropped DataFrames.
- The calibration model left stale descriptor shapes.
- The random-walk test could not pass, and the model used the global RNG.
- The parameter-sweep example read a nonexistent column.
- Plot export failures (for example, kaleido without Chrome) aborted runs. They are now logged in the record instead.

- Issues found in an independent review of this release, each with a regression test (`tests/test_review_fixes.py`):
  - **Design re-runs:** re-running a saved `design.csv` rounded 63-bit seeds and turned integer parameters into floats (pandas `iterrows` upcasting).
  - **Law shifts:** shifts of engine settings (seed, observation, storage) were recorded but had no effect. They are now rejected; `n_steps` is still allowed.
  - **Storage:**
    - output names that sanitise to the same file name, or differ only in case, overwrote each other;
    - string arrays broke HDF5 storage and are now kept inline;
    - a non-default DataFrame index was dropped.
  - **Failed saves:** a failure while writing data files left no record, and a failed save could hide the model's original error.
  - **Sampling:** an out-of-range sampling `offset` silently wrapped around or produced an empty observation.
  - **`results.csv`:** sampled observations sat on the wrong rows; they now sit on the rows of their sampling times.
  - **Config handling:**
    - string booleans (`save_csv: "no"`) were treated as true;
    - with an explicitly passed class, the stored config could name a different model;
    - a model could modify the stored config.
  - **Validation:**
    - it modified shared descriptors;
    - integer design parameters with non-integer bounds could fall outside their bounds.
- Plot tests that need static export are skipped when kaleido/Chrome is unavailable. kaleido is now part of the `dev` extra.

### Changed
- The Lotka–Volterra defaults are unchanged: forward Euler, dt = 1, clamping. This configuration is now documented and tested as a deliberately simple teaching/audit model; use `integrator: rk4` for reference data.
- The `run_parameter_sweep` 'list' output now also includes each run's `seed`.

### Removed
- Empty files: `simulator/llm_client.py` (the abstract `LLMClient` interface stays in `simulator/base.py`) and `examples/example_notebook.ipynb`. Also removed: the stray `test_pyyaml.py` script.
- Features advertised in the README but not implemented (experiment chaining, LLM-assisted design, notebook integration) are now listed under *Planned*.

## 0.1.0 (2025-03-16)
- Initial concept: the `ExperimentLogic` plug-in interface, YAML configs, the engine, experiment records, full-factorial sweeps and plots.
