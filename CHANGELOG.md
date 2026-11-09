# Changelog

Versions 0.2–0.6 were development steps; they were first released together in 0.7.0.

## 0.7.0rc1 (2026-11-09)

Release candidate for 0.7.0, the first release since the 0.1.0 prototype.

### Added
- **Install and update scripts.** `scripts/install.cmd` / `install.ps1` (Windows) and `scripts/install.sh` (Linux, macOS; bash 3.2 compatible) create `.venv`, install a profile (`core`, `standard`, `all`, `dev`) and run the doctor. `update.*` pulls a git checkout (fast-forward only, never over local changes) and reinstalls the remembered profile. `create_venv.bat` and `run_example.bat` now delegate to them.
- **`sds doctor`** (`python -m simulator.doctor`) checks:
  - Python version and virtual environment;
  - core packages and each optional extra, with the command that adds it;
  - Chrome for static plot export;
  - a writable output folder;
  - the example manifests;
  - a short reproducibility test.

  It exits with status 1 when the core is broken.
- **Example manifests** (`examples/*/example.yaml`) and `simulator.examples`. Each manifest gives the example's title, summary, fields, level, runtime, needed extras, entry module, output argument, quick-mode arguments, typed parameters (with ranges and choices) and the outputs to show. `run_example` runs any example in a subprocess, with UTF-8 output, a log and a timeout that also stops the example's own child processes. It writes `run.json`: the exact command, the effective parameters (your values first, then quick-mode values, then defaults), versions and the return code.
- **The `sds` command** (also `python -m simulator`): `doctor`; `examples list | show | run | check [--run-quick]`; `run CONFIG.yaml`. After a run it prints the command and folder to repeat it, quoted for the current system's shell.
- `sds doctor` lists optional extras that are not installed as `info` (add them only if you need them), not as warnings; warnings are kept for things that affect what you installed (security floors, Chrome for static plots).
- The install scripts rebuild a `.venv` made with a Python older than 3.11.
- `standard` install profile (extra); `all` now also includes `validate`. `simulator.__version__` reads `pyproject.toml` in a checkout, and run records use it.
- The bundled models (`experiments`) are now part of the installed package, so `sds` and the engine find them from any folder.
- `.gitattributes` fixes line endings: LF for `.sh`, CRLF for `.cmd`, `.bat` and `.ps1`.
- **Browser UI** (`ui/`, the separate package `scientific-data-simulator-ui`, started with `sds ui`):
  - pages: an examples gallery, a run form generated from each manifest with a live log and outputs, a results browser (click a row to open a run with its figures, parameters, log and the command that repeats it) and the installation check;
  - the sidebar shows the author, licence, source link and how to cite (from `CITATION.cff`);
  - it listens on 127.0.0.1 by default (`--address` changes that, with a warning) and needs Streamlit 1.67 or newer;
  - runs started at the same time get separate folders, and runs in progress refresh their log without blocking the page;
  - it uses only the simulator's public functions, and a test checks the simulator never imports it;
  - its version moves with the simulator's;
  - it is installed with the `ui` profile and started with `scripts/start-ui.*`;
  - headless tests (Streamlit AppTest) open every page and run an example through the form.
- CI runs the tests on macOS too, runs the install scripts on Linux, macOS and Windows as a new user would, runs every example in quick mode, and tests the UI on all three systems.

### Changed
- **Python 3.11 or newer** is required. Python 3.9 and 3.10 are end-of-life, and Pillow's current security release does not support them. CI tests 3.11–3.14.
- **Security floor for Pillow:** `pillow>=13.0` (core and UI). Pillow 13.0.0 (16 October 2026) fixes CVE-2026-112900 (EPS filenames reaching Ghostscript) and memory-corruption bugs in its TIFF and JPEG decoders. The UI needs Streamlit 1.67 or newer, the first release that accepts Pillow 13; the floor also makes pip upgrade an older Streamlit that is already installed.
- Examples 1-6 accept `--output_dir`.
- Static matplotlib figures are drawn with `matplotlib.figure.Figure`, not pyplot, so the simulator never picks or starts a GUI backend (the images are unchanged).
- Relative paths written into records and model-facing text use forward slashes on every system: the attacker's release view, audit findings, the design manifest's figure path, task files and `run.json`.

### Fixed
- Examples 11 and 12 deleted their whole output folder, including the runner's open `run.log`: on Windows the run failed (file in use), elsewhere the log was lost. `simulator.examples.prepare_output_dir` clears earlier results and keeps the runner's files.
- Task files fall back to absolute paths when no relative path exists (Windows, another drive) instead of failing.
- The UI shows interactive plots with `st.iframe` (`st.components.v1.html` is deprecated), and `sds ui` turns on colour support in the Windows console, which showed raw escape codes.
- CI: failed tests are listed, short, on the run's Summary page (`tools/ci_failures.py`).

## 0.6.0 (unreleased)

### Added
- **`simulator.ai`**: language models in defined roles through one logged, replayable client. Answers always come from the simulator.
  - `LoggedClient` writes one JSON line per call (role, model, settings, prompt, output, hashes, latency). `ReplayClient` regenerates results from a log without the model. `llm_provenance` / `write_llm_provenance` summarise model use for a release, without prompt text. `roles_of` shows which models played which roles.
  - `LLMAgent`: a model playing an `ExperimentEnv`. The prompt is rebuilt from the episode each turn, so episodes replay exactly. The remaining budget, which counts failed experiments, is stated each turn, and the agent is told to answer when none is left. The last valid JSON action in a reply is used. Replies that cannot be parsed get a correction round, then an answer that scores as invalid.
  - `llm_attack`: the model answers a release's questions from its evidence-free view (`evidence_free_view` / `build_view`). The view has no figures, figure scripts, measurement columns, data tables, answer keys or files a bundle marks private. Accuracy per template is tested against the blind baseline, chance and a 0.5/n floor (binomial, Bonferroni). The verdict is PASS, FLAGGED (answerable without the evidence: inspect) or INCONCLUSIVE (too few answers, or a truncated view). Replies are accepted as an id-to-answer mapping or a list of records.
  - `paraphrase_tasks`: model-rephrased questions, accepted only if they keep the numbers (in order, no number words added), keep the answer options (in order, without emphasis), add no answer from the dataset vocabulary and no negation, and fit in one line. The question must occur exactly once in the prompt. Rejections are reported by reason (`check_rewrite`, `answer_vocabulary`). Synonyms and subtle changes of meaning are not caught, so review a sample.
  - `solve_tasks`: a model answering bundle questions as text; replies are scored by `simulator.export.tasks.score`.
- Example 12: attacker, writer and agent on the law-discovery suite, with provenance and replay; runs offline with a scripted stand-in or with `--model module:function --model-name ...`.

## 0.5.0 (unreleased)

### Added
- **`simulator.diagnostics`**, the simulator as a testbed for benchmarks and agents:
  - `shortcuts`: `blind_baseline` (answers from the other answers only: majority, correct position, median), `feature_shortcuts` (leave-one-out nearest neighbours on public per-run cues, with a permutation test), `public_features` (figure size, caption length, public table rows).
  - `contamination`: `make_twin` (same design, new seeds), `pair_tasks`, `contamination_gap` (paired permutation test and bootstrap interval, per template), canaries (`make_canary`, `add_canary`, `canary_probe` with the chance level of a match).
  - `episodes`: `run_episodes` records agent episodes; `diagnose_episode` flags answering without evidence, premature commitment, repeated experiments, narrow sampling, answers that contradict the agent's own data, and answers matching an alternative hypothesis (e.g. the textbook form of a counterfactual law); `summarize`.
  - `quality`: `quality_card` (balance, entropy, correct-position chi-square, spread, duplicates, fidelity, optional reference-solver accuracy) and `card_markdown`.
  - `stats`: `gap_test` (paired / unpaired permutation test, bootstrap interval; paired tests can be clustered, e.g. by run, and `contamination_gap` clusters by run by default), exact binomial test.
- `experiments.law_discovery.logic.textbook_counterpart`: the textbook form of a law fitted to it.
- `bundle.qa.exclude`: leave question templates out (ids stay consecutive; other questions unchanged).
- `ExperimentEnv.domain`: the public scoring range of expression answers.
- Example 11: the testbed on the law-discovery suite (shortcuts, leak audit with canary, twin contamination gaps, canary probe, agent-fallacy flags).

### Changed
- Example 10 excludes the `argmax` and `peak_count` templates, which example 11 shows are answerable without the data (56 % and 78 %).

## 0.4.0 (unreleased)

### Added
- **Law-discovery pack** (`experiments/law_discovery/`, example 10):
  - `HiddenLawExperiment`: nine law families (power law, inverse power, exponential decay, logistic, Michaelis–Menten, Hill, damped oscillation, Gaussian peak, Arrhenius) with seeded priors; `counterfactual: true` moves the structural constants (exponents, Hill coefficient) off their textbook values.
  - Questions: law family (choice), formula (`answer_type: expression`), extrapolation; parameter questions only on request (their wording names the family).
  - All families share one x range by default (a family-specific range would identify the family).
  - `law_suite_design()` crosses families, textbook/counterfactual and noise levels, in a shuffled, unrecorded row order unless `seed` is given.
  - Example 10: a classical baseline (curve fit over all families, AIC selection), the interactive formula-discovery environment, and a Hugging Face export that is leak-audited.
- **`simulator.expressions`**: a safe formula parser (AST whitelist, no `eval`) that normalises LaTeX, `^`, `y =` and implicit products; `matches()` scores formulas by relative RMS error on a domain, so equivalent forms are accepted.
- **Expression answers** in bundles, the tasks exporter (`score` extracts formulas from free text) and `ExperimentEnv` (`target_type: expression`, `domain`).
- **`simulator.audit`**: leak audit of public exports (`audit_export`, `python -m simulator.audit`) and of interactive environments (`audit_environment`). Attacks: published private fields, seed replay, config-hash lookup, tolerance inversion, answer lookup (exact, rounded or re-spaced), exposure of hidden arrays, design-point indices. Reads JSON/JSONL, CSV, Parquet, NumPy and HDF5; unreadable files fail the audit.
- **Models:**
  - `HeatExperiment` (1-D/2-D periodic heat equation): spectral exact solution (`reference`) and FTCS (`audit`, with an explicit instability warning above the stability limit).
  - `PendulumExperiment`: nonlinear pendulum (`reference`, period checked against the elliptic integral) and the small-angle approximation (`audit`, a model-form error).
  - `PolygonExperiment` (regular, convex, simple polygons) and `SimplexExperiment` (n-simplices in 2–6 dimensions, Cayley–Menger verified, projected diagrams).
  - Random walk: `generator: randu` (`audit`), the RANDU generator whose triples lie on 15 planes.
- **`simulator.export.the_well`**: export PDE runs as HDF5 in The Well's schema (checked with `the_well` 1.2.0's `WellDataset`); fractional splits work when every run has its own parameters; `the-well` extra.

### Changed
- Observation operators without explicit `targets` now apply only to 1-D numeric latent series (fields such as a 2-D temperature map are left alone).
- `ExperimentEnv`: an experiment the model rejects returns a generic error (the model's message could name hidden values) and uses budget (whether a run fails can depend on hidden values, so failures must not be a free oracle); the episode truth includes label targets.
- Hugging Face export: when design columns are withheld, `design_point` is dropped and runs are ordered by their random ids (row order and design-point index would map runs back to a public or regenerable design table).

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
