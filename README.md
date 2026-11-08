# Scientific Data Simulator

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![tests](https://github.com/sandner-art/scientific-data-simulator/actions/workflows/tests.yml/badge.svg)](https://github.com/sandner-art/scientific-data-simulator/actions/workflows/tests.yml)

| Author: Daniel Sandner | Scientific Data Simulator: A Modular Concept for Reproducible Experiment Design and Synthetic Data Generation
- part of the **100 Scientific Visions** project (2025-2027)

## Overview

Scientific Data Simulator turns an executable scientific model into **reference data**: datasets whose generating mechanism, parameters and measurement process are known exactly and recorded with the data.

It is meant for situations where real data are unavailable, not yet available, or have no known ground truth:

- validating analysis methods, fitting code and inference pipelines;
- testing and training AI systems, including law discovery, parameter inference and noisy-data robustness;
- teaching, and early-stage research before experiments exist.

The simulator does not try to be a general solver. It is an **observation-and-provenance layer** between a mechanism (yours, or a wrapped library) and whatever consumes the data.

![simulator](https://github.com/user-attachments/assets/1c627a6a-b575-4474-8bd8-1e748db05c15)

## Features

* **Plug-in models.** A model is a small `ExperimentLogic` class (`initialize`, `run_step`, `get_results`). A YAML config names it by dotted path.
* **Seeded, recorded randomness.** Every run has one integer seed, either set in the config or drawn and then recorded. Independent random streams for the model and for observation noise are derived from it, so a stored config is a complete recipe for re-running.
* **Role-tagged outputs.** Each output is labelled `parameter`, `latent`, `label`, `observation`, `coordinate` or `auxiliary`.
* **Observation operators** turn exact (latent) outputs into realistic measurements without changing the mechanism:
  * sampling: every k-th step, n points, or explicit indices;
  * Gaussian noise (absolute `sigma` or `snr_db`), multiplicative and Poisson noise;
  * quantisation, outliers, missing values and clipping.

  Every setting is recorded, including noise levels derived from an SNR and the positions of outliers.
* **Ground truth exported separately.** `ground_truth.json` and `data/ground_truth/` hold the answers, while `data/observation/` holds what a method under test gets to see.
* **Counterfactual law shifts.** `law_shift:` changes a constant or swaps a functional form, recording both the original and the shifted law. Benchmarks built this way cannot be answered from memorised textbook equations.
* **Provenance in every record:**
  * config SHA-256, and the model's source-file SHA-256, git commit and dirty flag;
  * simulator and dependency versions, system info and a `synthetic` flag;
  * a SHA-256 for every data file. Checksums are verified on load.
* **Data files instead of inline JSON.** Arrays are written as `.npy` (or one HDF5 file) and DataFrames as Parquet or CSV, grouped by role. Arrays can be memory-mapped when loaded.
* **Design of experiments.** Full-factorial, Latin hypercube, Sobol, Halton and random designs (linear or log scale, integer parameters). Designs support replicates, per-row or common-random-number seeds, dotted keys for nested settings (e.g. sweeping `observation.operators.0.snr_db`), parallel workers, and a tidy results table with one row per run.
* **Verified reference models.** The bundled models are checked against closed forms, high-accuracy solvers, convergence orders and conserved quantities (`tests/test_verification.py`).
* **Multimodal run bundles.** With `bundle: true`, one run also writes, all derived from the same arrays and seed:
  * the figure, rendered by executing the shipped `plot_figure.py`, so code and image cannot disagree;
  * the plotted data as `table.csv`;
  * a caption that reveals no answers;
  * questions with answers and tolerances computed from the exact curves (peaks, extremes, values, comparisons, parameter estimates);
  * the mechanism graph (JSON and Graphviz).

  Models that describe a diagram (`figure_elements()`) get a geometric drawing instead of a chart, and can add their own exact questions (`questions()`). Files are marked public (what a system under test sees) or private (the answers and the numeric tolerances, which would reveal them). Hiding the exact curves (`show_latent: false`) turns figure-reading questions into inference questions. A design collects all of its runs' questions into `questions.jsonl` (public) and `qa.jsonl` (private).
* **Model fidelity labels.** Every model declares a fidelity level with a short reason, recorded in each run (see [Model fidelity](#model-fidelity-reference-and-deliberately-weak-models)).
* **Exporters** (`simulator.export`):
  * questions as scored tasks (`tasks.jsonl`);
  * Hugging Face dataset folders with a dataset card and Croissant 1.0 metadata (validated with `mlcroissant`, loadable with `datasets`);
  * (θ, x) training sets and a batch simulator for simulation-based inference (`sbi`);
  * OpenEnv environment packages and verifiers tasksets for reinforcement learning.

  Ground truth is withheld unless you ask for it.
* **Interactive discovery environments** (`simulator.envs`). An agent runs a limited number of noisy experiments on a system with hidden parameters, then reports an estimate, which is scored automatically. It has a Gymnasium-style `reset`/`step`/`state` API and no dependencies.
* **Physical units** (optional `pint`). Parameters can be given with units (`stiffness: "2 kN/m"`) and are converted to the model's declared units. Output units are checked, including traps such as `units` (pint reads it as *micronit*) and `au` (astronomical unit).
* **Model checker** (`simulator.checker`). A gate for new and LLM-written models:
  * a static scan of the source;
  * determinism and untouched global random state;
  * finite outputs;
  * convergence order when `dt` is halved;
  * invariant drift;
  * dead parameters;
  * units.

  `simulator.llm.draft_model` asks any LLM for a model, runs the checker in a separate time-limited process, and retries with the findings.
* **Law discovery** (`experiments.law_discovery`). Hidden laws y = f(x) from nine families (power law, inverse power, exponential decay, logistic, Michaelis–Menten, Hill, damped oscillation, Gaussian peak, Arrhenius), each with a textbook or a *counterfactual* setting of its structural constants (e.g. an inverse-*2.3* law). Formula answers are scored by evaluating them on the data domain (`simulator.expressions`: a safe parser with no `eval`), so any equivalent form counts. A suite design, a classical curve-fitting baseline and an interactive formula-discovery environment come with it (example 10).
* **Leak audit** (`simulator.audit`). Before publishing, attack your own export: it tries seed replay, config-hash lookup, tolerance inversion, answer lookup in public files and exposure of hidden arrays, using the private runs as the answer key, and reports what would leak. `audit_environment` does the same for interactive environments.
* **Automatic plots.** matplotlib PNG and interactive Plotly HTML, driven by output descriptors; observed points are drawn over the latent curves. Static SVG/PDF export needs the optional `kaleido` package.

### Planned

* Multi-variable and dynamical (ODE) law discovery; symbolic-regression baselines (e.g. PySR) as optional comparisons.
* 3-D solids beyond the tetrahedron; dihedral angles.
* Adapters: Mesa (agent-based), Exponax (PDE), FMPy (FMI models), MLflow/DVC tracking.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + pytest, pyarrow, h5py, pint, kaleido
```

Python 3.9 or newer is required. Optional extras:

| Extra | Adds |
|---|---|
| `static-plots` | kaleido and plotly ≥ 6.1.1. Static SVG/PDF export also needs Chrome. |
| `parquet`, `huggingface` | pyarrow |
| `hdf5` | h5py |
| `the-well` | h5py and PyYAML, to write The Well HDF5 (loading it needs `the_well` and PyTorch) |
| `units` | pint |
| `validate` | mlcroissant and datasets, to check exported datasets |
| `openenv` | the OpenEnv library (Python ≥ 3.10), to use exported RL packages |
| `verifiers` | the verifiers library (Python ≥ 3.11), to use exported RL packages |
| `all` | static-plots, parquet, hdf5 and units |

Install `openenv` and `verifiers` in separate environments: their current releases pin incompatible versions of `mcp`.

## Quick start

```bash
python -m examples.example_1.run_experiment                       # sine wave
python -m examples.example_7_reference_dataset.run_experiment     # reference dataset with noise sweep
pytest                                                            # full test suite
```

A minimal config:

```yaml
experiment_type: experiments.predator_prey.logic.PredatorPreyExperiment
seed: 42
n_steps: 3000
dt: 0.01
integrator: rk4
clamp_negative: false
initial_prey: 10.0
initial_predators: 5.0
prey_growth_rate: 1.1
prey_death_rate: 0.4
predator_growth_rate: 0.1
predator_death_rate: 0.4

observation:                    # optional measurement model
  sampling: {every: 25}
  operators:
    - {type: gaussian, snr_db: 20}
    - {type: missing, rate: 0.05}

law_shift:                      # optional counterfactual law
  changes: {functional_response: holling2, handling_time: 0.3}
  description: saturating predation

storage: {format: npy}          # npy (default) | hdf5 | inline
bundle: true                    # figure + code + table + caption + Q&A + graph
```

From Python:

```python
from simulator.engine import SimulatorEngine
from simulator.doe import create_doe_table, run_design

engine = SimulatorEngine(output_dir="experiments_output")
run = engine.run_config(config)                 # or engine.run_experiment("config.yaml")
run.record.ground_truth["parameters"]           # the answers
run.results["prey_population_observed"]["data"] # what a method gets to see

design = create_doe_table({"prey_growth_rate": [0.8, 1.4],
                           "observation.operators.0.snr_db": [10, 40]},
                          design_type="lhs", n_samples=32, seed=1)
out = run_design(config, design, output_dir="lhs_study", replicates=5, n_jobs=4)
out.table      # design + seed + experiment_id + scalar outputs, one row per run
```

With `n_jobs > 1`, scripts must put this code under `if __name__ == "__main__":` because worker processes re-import the main module.

## Model fidelity: reference and deliberately weak models

Not every bundled model is meant to be correct. Some are kept, or will be added, as **deliberately weak models**: they are numerically or structurally flawed in a known, documented way, so that you can test whether a fitting method, an AI system or a validation pipeline *notices* model error. To keep this transparent, every model declares a fidelity level and a reason. The engine records it in `provenance.fidelity` and `ground_truth.json`, writes it into each bundle's `description.md`, and logs a warning for `audit` and `unverified` runs.

| Level | Meaning |
|---|---|
| `reference` | Verified against closed forms, high-accuracy solvers, convergence orders or invariants (see `tests/test_verification.py`). |
| `audit` | Deliberately simplified or flawed. Use it to test whether a method notices model error; do not use it as ground truth for the real system. |
| `unverified` | No verification tests, or outside the verified range. Not reference data. |
| `observed` | Outputs come from loaded real data, not from a simulation. |

**Lotka–Volterra.** The predator–prey model's default configuration (`integrator: euler`, `dt: 1`, clamping of negative populations) reproduces version 0.1.0. It is labelled `audit`: forward Euler at this step size carries first-order integrator error and violates the conserved Lotka–Volterra quantity by more than 1 %. Use `integrator: rk4` with `dt ≤ 0.04` for `reference` data; larger steps are labelled `unverified`.

**Oscillator.** `integrator: euler` is the second deliberately weak model. Explicit Euler adds energy to an undamped oscillator at every step, so its amplitude grows; it is labelled `audit`. The closed-form (`exact`) and DOP853 options are `reference`.

**Pendulum.** `approximation: small_angle` replaces sin θ by θ. The solution is exact, but of the wrong equation: a *model-form* error, not a numerical one. Its period does not depend on amplitude and is 7 % short at 60° and 15 % short at 90°; the fidelity note states the error for the configured amplitude. The nonlinear DOP853 solution is `reference` (period checked against the elliptic-integral formula).

**Random walk.** `generator: randu` draws the coin flips from RANDU, IBM's notorious linear congruential generator, whose consecutive triples lie on 15 planes. Means and variances look normal; the raw uniforms are stored (`uniforms`, auxiliary) so that randomness diagnostics can find the defect.

**Heat equation.** `integrator: ftcs` is explicit finite differences. It is labelled `audit`: first order in time, second in space, and *unstable* above the stability limit (the fidelity note then says so and gives the growth factor). The spectral `exact` solution is `reference`.

All audit models have pinning tests (`tests/test_audit_models.py`, `tests/test_heat_and_the_well.py`) that check the flaw is still there and still the size documented.

A model sets its level with the class attributes `fidelity_level` and `fidelity_note`, or overrides `fidelity(config)` when the level depends on the configuration. The model checker never promotes a model to `reference`: that needs verification tests against independent knowledge.

## Exporting and using the data

```python
from simulator.export.tasks import collect_bundle_tasks, write_tasks_jsonl, score
from simulator.export.huggingface import export_huggingface
from simulator.export.sbi import generate_training_data
from simulator.export.rl import export_openenv, export_verifiers_qa
from simulator.export.the_well import export_the_well
from simulator.audit import audit_export

tasks = collect_bundle_tasks("experiments_output/my_design")          # questions from bundles
write_tasks_jsonl(tasks, "tasks.jsonl")                              # public (no answers)
score(tasks[0], "The maximum is about 13.7")                         # 1.0 or 0.0

export_huggingface("experiments_output/my_design", "hf_dataset", name="my-benchmark")
data = generate_training_data(config, {"prey_growth_rate": [0.6, 1.6]}, n=1000, seed=1)  # theta, x
export_openenv("examples/example_9_discovery_env/env.yaml", "rl/", name="lv_discovery")
export_verifiers_qa(tasks, "rl/", name="lv_figure_qa")
export_the_well("experiments_output/heat_design", "well/", dataset_name="sds_heat")  # PDE fields

print(audit_export("hf_dataset", "experiments_output/my_design").summary())  # PASS / FAIL
```

Ground truth stays private by default:
- The Hugging Face export omits latent and label arrays, `ground_truth.json`, run configurations, answers and numeric tolerances.
- `include_seeds=False` also withholds seeds, since a seed plus the simulator regenerates the data.

Run the leak audit (`audit_export`, or `python -m simulator.audit PUBLIC --private RUNS`) on anything you publish. It reports, per attack, how many hidden answers it recovered: published seeds (seed replay), config hashes, numeric tolerances that are a fixed fraction of the answer, answers that appear in public tables or text (exactly, rounded to 4 or more digits, or as a formula with other spacing), hidden arrays (noise-free curves) found among public numbers, and public design-point indices. It reads JSON/JSONL, CSV, Parquet, NumPy and HDF5 files and text; a file it cannot read is itself a failure, because it cannot be cleared. It exits with status 1 on a high-severity finding, so it can gate a release. It finds the known kinds of shortcut only: passing it is necessary, not sufficient.

The Well export writes HDF5 files in the schema of [The Well](https://github.com/PolymathicAI/the_well) (fields `t0_fields/u`, dimensions, periodic boundary conditions, scalars, `stats.yaml`), loadable with `the_well.data.WellDataset(path=...)`.

The OpenEnv package serves the interactive environment over HTTP/WebSocket and passes `openenv validate`. Use a WebSocket session for multi-step episodes: plain HTTP `/step` calls are stateless. The verifiers taskset scores answers with `score`, and verifiers' `openenv` taskset can drive the OpenEnv package for multi-turn training.

## Checking a model (and drafting one with an LLM)

```bash
python -m simulator.checker experiments.oscillator.logic.OscillatorExperiment config.yaml --markdown check.md
```

```python
from simulator.llm import CallableLLMClient, draft_model
draft = draft_model(CallableLLMClient(my_llm_function), "A pendulum with linear drag ...", config)
if draft.accepted:
    draft.save("experiments/drag_pendulum/logic.py")      # writes logic_check.md next to it
```

`my_llm_function(prompt) -> text` wraps any provider SDK; the simulator itself depends on none. Generated code is scanned before it runs, then checked in a separate process with a time limit. Rejected drafts cannot be saved.

## Writing a model

```python
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np

class Decay(ExperimentLogic):
    def __init__(self, config):
        self.k, self.n = config["k"], config["n_steps"]

    def initialize(self, config):
        self.t, self.x = [0.0], [1.0]
        return {"x": 1.0}

    def run_step(self, state, step):
        x = state["x"] * np.exp(-self.k) + 0.01 * self.rng.normal()  # always use self.rng
        self.t.append(step + 1.0); self.x.append(x)
        return {"x": x}

    def get_results(self):
        return {
            "t": {"data": np.array(self.t), "descriptor": DataDescriptor("t", DataType.NDARRAY, role="coordinate")},
            "x": {"data": np.array(self.x), "descriptor": DataDescriptor(
                "x", DataType.NDARRAY, x_axis="t", group="time_series", plot_type="line", role="latent")},
        }
```

Stochastic models must draw random numbers from `self.rng`, never from the global `np.random`. Override `ground_truth_parameters(config)` to export derived parameters.

## Output of a run

```
experiments_output/<timestamp>_<experiment_id>/
├── experiment_record.json   config, seed, provenance, descriptors, data-file manifest (sha256)
├── ground_truth.json        parameters, law shift, realised observation model, field roles
├── data/ground_truth/       latent / parameter / label arrays
├── data/observation/        observed arrays
├── data/coordinate/         time / x axes
├── data/auxiliary/          everything else
├── bundle/                  (with bundle:) figure, plot_figure.py, table.csv, caption.txt,
│                            questions.json | qa.json, graph.json/.dot, description.md, bundle.json
└── *_matplotlib.png, *_plotly.html
```

## Bundled models

| Model | Path | Notes |
|---|---|---|
| Sine wave | `experiments.example_experiment.logic.ExampleExperiment` | `reference`: closed form |
| Random walk | `experiments.example_random_walk.logic.RandomWalkExperiment` | `reference`: seeded; increment statistics tested. `audit` with `generator: randu`. |
| Linear function | `experiments.linear_function.logic.LinearFunctionExperiment` | `reference`: closed form |
| Lotka–Volterra | `experiments.predator_prey.logic.PredatorPreyExperiment` | `reference` with `integrator: rk4`, `dt ≤ 0.04`; `audit` with the 0.1.0 default (forward Euler, dt = 1). Options: `holling2`, logistic prey. |
| Oscillator | `experiments.oscillator.logic.OscillatorExperiment` | SI units. `reference`: closed form (linear) or DOP853 (Duffing cubic term); `audit` with `integrator: euler`. |
| Pendulum | `experiments.pendulum.logic.PendulumExperiment` | SI units. `reference`: nonlinear, DOP853; `audit` with `approximation: small_angle`. |
| Heat equation | `experiments.heat.logic.HeatExperiment` | 1-D/2-D periodic. `reference`: spectral exact solution; `audit` with `integrator: ftcs`. Export to The Well HDF5. |
| Hidden laws | `experiments.law_discovery.logic.HiddenLawExperiment` | `reference`: closed forms; nine families, textbook or counterfactual constants; formula questions. |
| Triangle geometry | `experiments.geometry.logic.TriangleExperiment` | `reference`: exact constructions (circumcircle, incircle, centroid, orthocentre, medians, altitudes, Euler line); diagrams and questions in bundles. |
| Polygons | `experiments.geometry.polygons.PolygonExperiment` | `reference`: regular, convex or simple polygons; area, centroid, angles, convexity, reflex corners. |
| n-simplex | `experiments.geometry.simplex.SimplexExperiment` | `reference`: simplices in 2–6 dimensions; volume (Cayley–Menger checked), facets, circum- and in-sphere; projected diagrams. |
| Calibration overlay | `experiments.predator_prey_calibration.logic.PredatorPreyCalibrationExperiment` | overlays observed CSV data (manual calibration); fidelity as for Lotka–Volterra |
| CSV summary | `experiments.data_analysis.logic.DataAnalysisExperiment` | `observed`: real data, recorded as `synthetic: false` |

## Project structure

```
simulator/        engine.py, base.py, config.py, schema.py, rng.py, observation.py,
                  bundle.py, storage.py, persistence.py, provenance.py, experiment_record.py,
                  doe.py, data_handler.py, visualization.py, utils.py,
                  envs.py, units.py, checker.py, llm.py, expressions.py, audit.py,
                  export/ (tasks, huggingface, sbi, rl, the_well)
experiments/      bundled reference models (one package per model)
examples/         runnable examples (python -m examples.<name>.run_experiment)
tests/            pytest suite, incl. test_verification.py (numerical verification)
docs/             Sphinx sources
```

## Citation

If you use Scientific Data Simulator in your research, please cite it:

> Daniel Sandner. (2025). Scientific Data Simulator (Version 0.3.0.dev0) [Computer software]. https://github.com/sandner-art/scientific-data-simulator

Full citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
