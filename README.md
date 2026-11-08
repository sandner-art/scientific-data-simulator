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

  Files are marked public (what a system under test sees) or private (the answers). Hiding the exact curves (`show_latent: false`) turns figure-reading questions into inference questions. A design collects all of its runs' questions into `questions.jsonl` (public) and `qa.jsonl` (private).
* **Model fidelity labels.** Every model declares a fidelity level with a short reason, recorded in each run (see [Model fidelity](#model-fidelity-reference-and-deliberately-weak-models)).
* **Automatic plots.** matplotlib PNG and interactive Plotly HTML, driven by output descriptors; observed points are drawn over the latent curves. Static SVG/PDF export needs the optional `kaleido` package.

### Planned

* Law-discovery and symbolic-regression packs; geometry diagrams with exact coordinates.
* Exporters: Hugging Face datasets with Croissant metadata, verifiers/OpenEnv tasks, `sbi` (θ, x) pairs, The Well HDF5 layout.
* Physical units (pint) and LLM-assisted experiment authoring gated by automatic numerical checks. The abstract `LLMClient` interface in `simulator/base.py` is the placeholder for this.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core + pytest, pyarrow, h5py
```

Optional extras are `static-plots` (kaleido; static Plotly export needs Chrome), `parquet`, `hdf5` and `all`. Python 3.9 or newer is required.

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

A model sets its level with the class attributes `fidelity_level` and `fidelity_note`, or overrides `fidelity(config)` when the level depends on the configuration.

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
| Random walk | `experiments.example_random_walk.logic.RandomWalkExperiment` | `reference`: seeded; increment statistics tested |
| Linear function | `experiments.linear_function.logic.LinearFunctionExperiment` | `reference`: closed form |
| Lotka–Volterra | `experiments.predator_prey.logic.PredatorPreyExperiment` | `reference` with `integrator: rk4`, `dt ≤ 0.04`; `audit` with the 0.1.0 default (forward Euler, dt = 1). Options: `holling2`, logistic prey. |
| Calibration overlay | `experiments.predator_prey_calibration.logic.PredatorPreyCalibrationExperiment` | overlays observed CSV data (manual calibration); fidelity as for Lotka–Volterra |
| CSV summary | `experiments.data_analysis.logic.DataAnalysisExperiment` | `observed`: real data, recorded as `synthetic: false` |

## Project structure

```
simulator/        engine.py, base.py, config.py, schema.py, rng.py, observation.py,
                  bundle.py, storage.py, persistence.py, provenance.py, experiment_record.py,
                  doe.py, data_handler.py, visualization.py, utils.py
experiments/      bundled reference models (one package per model)
examples/         runnable examples (python -m examples.<name>.run_experiment)
tests/            pytest suite, incl. test_verification.py (numerical verification)
docs/             Sphinx sources
```

## Citation

If you use Scientific Data Simulator in your research, please cite it:

> Daniel Sandner. (2025). Scientific Data Simulator (Version 0.2.0.dev0) [Computer software]. https://github.com/sandner-art/scientific-data-simulator

Full citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
