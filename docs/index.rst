.. Scientific Data Simulator documentation master file

Scientific Data Simulator - User Guide
=======================================

.. contents:: Contents
   :local:
   :depth: 1

.. _introduction:

Introduction
============

Scientific Data Simulator turns executable scientific models into **reference
data**: datasets whose generating mechanism, parameters and measurement process
are known exactly and are recorded with the data. Typical uses:

* validating analysis methods, fitting code and inference pipelines;
* testing and training AI systems (law discovery, parameter inference,
  robustness to noise);
* teaching, and early-stage research where real data are not yet available.

The framework is an *observation-and-provenance layer*: a model (your own
``ExperimentLogic`` or a wrapped library) produces exact outputs; the engine
seeds it, applies a recorded measurement model, separates ground truth from
observations, and writes checksummed data files with a full provenance record.

.. _installation:

Installation
============

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"

Python 3.9+ is required. Optional extras: ``static-plots`` (kaleido; static
plotly export needs Chrome), ``parquet`` (pyarrow), ``hdf5`` (h5py), ``all``.

.. _getting_started:

Getting Started
===============

From the repository root:

.. code-block:: bash

   python -m examples.example_1.run_experiment                    # sine wave
   python -m examples.example_7_reference_dataset.run_experiment  # reference dataset
   pytest

Output is written to ``experiments_output/``. Each example directory contains
``run_experiment.py``, a config file and a README (see ``examples/README.md``).

.. _experiment_logic:

Creating New Experiments (ExperimentLogic)
==========================================

A model subclasses :class:`simulator.base.ExperimentLogic` and implements:

* ``initialize(config)`` - return the initial state (a dict);
* ``run_step(state, step)`` - advance one step and return the new state;
* ``get_results()`` - return ``{name: {"data": ..., "descriptor": DataDescriptor(...)}}``.

.. code-block:: python

    from simulator.base import ExperimentLogic
    from simulator.utils import DataDescriptor, DataType
    import numpy as np

    class NoisyDecay(ExperimentLogic):
        def __init__(self, config):
            self.k = config["k"]

        def initialize(self, config):
            self.t, self.x = [0.0], [1.0]
            return {"x": 1.0}

        def run_step(self, state, step):
            x = state["x"] * np.exp(-self.k) + 0.01 * self.rng.normal()
            self.t.append(step + 1.0)
            self.x.append(x)
            return {"x": x}

        def get_results(self):
            return {
                "t": {"data": np.array(self.t),
                      "descriptor": DataDescriptor("t", DataType.NDARRAY, role="coordinate")},
                "x": {"data": np.array(self.x),
                      "descriptor": DataDescriptor("x", DataType.NDARRAY, x_axis="t", group="time_series",
                                                   plot_type="line", role="latent")},
            }

Rules for reference-quality models (see also ``experiments/README.md``):

* draw random numbers only from ``self.rng`` (set by the engine from the run seed);
* give every output a ``role``;
* override ``ground_truth_parameters(config)`` if the exported parameters
  should differ from the config values;
* set the class attribute ``synthetic = False`` if outputs come from real data;
* add verification tests (closed form, convergence order, invariants).

.. _configuration:

Configuration
=============

Experiments are configured in YAML. Model parameters are free-form; the engine
validates its own keys:

.. code-block:: yaml

    experiment_type: experiments.predator_prey.logic.PredatorPreyExperiment  # dotted path (required)
    experiment_description: "..."
    seed: 42                    # optional; drawn and recorded if absent
    n_steps: 3000               # number of run_step calls (default 1)
    generate_plots: true
    static_plot_format: null    # null | svg | pdf | png | jpeg | webp
    save_csv: false
    observation: {...}          # measurement model (see below)
    law_shift: {...}            # counterfactual law (see below)
    storage: {format: npy}      # npy | hdf5 | inline

Values of the form ``$NAME`` are read from environment variables (or ``.env``).

.. _reproducibility:

Reproducibility and Seeds
=========================

Each run has one integer seed. If the config has none, a seed is drawn and
written back into the stored config, so the stored config always re-runs
identically. Independent streams are derived with
``numpy.random.SeedSequence``: ``model`` (available as ``self.rng``) and
``observation`` (noise). For a fixed seed, changing the noise model does not
change the latent trajectory.

The record's ``provenance`` holds the config SHA-256, the model's source-file
SHA-256 and git commit/dirty flag, the simulator version and a ``synthetic``
flag; ``software_versions`` lists every installed dependency.

.. _observation:

Observation Operators
=====================

Observation operators turn exact outputs into measurements without touching the
mechanism. For each target ``x`` the engine keeps ``x`` (role ``latent``) and
adds ``x_observed`` (role ``observation``):

.. code-block:: yaml

    observation:
      targets: [prey_population]          # default: all outputs with role "latent"
      sampling: {every: 25}               # or {n: 40, method: random|uniform} or {indices: [...]}
      operators:                          # applied in order
        - {type: gaussian, snr_db: 20}    # or sigma: 0.5
        - {type: multiplicative, sigma: 0.05}
        - {type: poisson, scale: 1.0}
        - {type: quantize, step: 1.0}
        - {type: outliers, rate: 0.02, scale: 5.0}
        - {type: missing, rate: 0.05}
        - {type: clip, low: 0.0}

Realised settings (the sigma implied by an SNR, outlier positions, number of
missing values) are recorded. Custom operators can be added with
:func:`simulator.observation.register_operator`.

.. _ground_truth:

Ground Truth and Law Shifts
===========================

Every run writes ``ground_truth.json`` next to its record: model parameters,
the realised observation model, the role of every field and the lists of
ground-truth and observation fields. Data files are grouped by role
(``data/ground_truth``, ``data/observation``, ``data/coordinate``,
``data/auxiliary``), so observations can be distributed without the answers.

A *law shift* runs the model under a counterfactual law while recording both
versions:

.. code-block:: yaml

    law_shift:
      changes: {functional_response: holling2, handling_time: 0.3}
      description: saturating predation

.. _fidelity:

Model Fidelity
==============

Every model declares how far its output can be trusted as reference data, with a
short reason. The engine records the label in ``provenance.fidelity`` and
``ground_truth.json``, and logs a warning for ``audit`` and ``unverified`` runs.

* ``reference`` - verified against closed forms, high-accuracy solvers,
  convergence orders or invariants (``tests/test_verification.py``).
* ``audit`` - deliberately simplified or flawed. Such models are kept on purpose
  to test whether a method notices model error, and are clearly marked as such.
* ``unverified`` - no verification tests, or outside the verified range.
* ``observed`` - outputs come from loaded real data.

The predator-prey default (forward Euler, ``dt = 1``, clamping; the 0.1.0
behaviour) and the oscillator with ``integrator: euler`` (which gains energy)
are labelled ``audit``. ``integrator: rk4`` with ``dt <= 0.04`` is
``reference``. Models set ``fidelity_level`` / ``fidelity_note`` or override
``fidelity(config)``.

.. _bundles:

Multimodal Run Bundles
======================

With ``bundle: true`` (or a ``bundle:`` mapping) each run writes ``bundle/``.
Models that describe a diagram (``figure_elements()``) get a geometric drawing
from ``elements.csv`` instead of a chart, and may add exact questions
(``questions()``). Public question files carry a ``tolerance_rule``; the numeric
tolerance is private because it is derived from the answer or the data.

* ``figure.png`` (or svg/pdf), rendered by executing ``plot_figure.py``, which
  reads ``table.csv``. Code, data and image therefore always agree, and the
  bundle is byte-for-byte reproducible from the seed.
* ``caption.txt`` - describes what is drawn without revealing answers.
* ``qa.json`` / ``questions.json`` - questions with answers and tolerances
  computed from the exact arrays: argmax, extremes, final and sampled values,
  peak counts, comparisons, and parameter estimates (``inferable_parameters``).
  Each question says whether it is answerable from the ``figure`` or needs
  ``inference``. Score responses with :func:`simulator.bundle.check_answer`.
* ``graph.json`` / ``graph.dot`` - the mechanism graph (``mechanism_graph()``).
* ``description.md`` - mechanism, parameters, law shift, noise model, fidelity.
* ``bundle.json`` - manifest with SHA-256 and a ``public``/``private`` label
  per file.

.. code-block:: yaml

    bundle:
      figure: {format: png, dpi: 120, show_latent: true, show_observed: true}
      qa: {value_queries: 2, parameters: null}
      graph: true

``show_latent: false`` draws observations only. Questions about the exact curves
then become inference tasks. :func:`simulator.doe.run_design` collects the
questions of all runs into ``questions.jsonl`` (public) and ``qa.jsonl``
(private).

.. _units:

Physical Units
==============

With ``pip install "scientific-data-simulator[units]"`` (pint), a model can
declare the units it computes in::

    class OscillatorExperiment(ExperimentLogic):
        parameter_units = {"mass": "kg", "stiffness": "N/m", "x0": "m", "dt": "s"}

Config values may then carry any compatible units (``stiffness: "2 kN/m"`` or
``x0: {value: 5, units: cm}``). They are converted before the model runs; the
stored config keeps what was written, and ``ground_truth.parameter_units``
records the units of the exported parameters. Output units are parsed and
reported in ``provenance.units``. Unknown units and traps are logged, or raise
with ``unit_check: error``: pint reads ``units`` as *micronit* and ``au`` as
*astronomical unit*. Simulator units: ``individual``, ``step``, ``index``,
``arbitrary_unit`` and ``time_unit`` (a model time without physical scale).

.. _exports:

Exporting Data and Tasks
========================

* :mod:`simulator.export.tasks` - bundle questions as tasks with prompts, figure
  paths and answer formats; ``score(task, response)`` accepts free text.
* :mod:`simulator.export.huggingface` - ``export_huggingface(source, out, name)``
  writes a dataset folder: dataset card, ``croissant.json`` (Croissant 1.0),
  ``data/runs.parquet``, ``data/questions.parquet`` with embedded figures. Ground
  truth, answers and tolerances are withheld unless requested.
* :mod:`simulator.export.sbi` - ``generate_training_data(config, prior, n, seed)``
  gives ``theta``/``x`` arrays (and files); ``BatchSimulator`` is a seeded
  ``theta -> x`` simulator; ``to_sbi_prior`` builds an ``sbi`` ``BoxUniform``.
* :mod:`simulator.export.rl` - ``export_openenv(spec, out)`` writes an OpenEnv
  package serving an interactive environment; ``export_verifiers_qa(tasks, out)``
  writes a verifiers taskset. Install openenv and verifiers in separate
  environments (their current releases pin incompatible ``mcp`` versions).

.. _environments:

Interactive Environments
========================

:class:`simulator.envs.ExperimentEnv` turns any model into a discovery task.
Hidden parameters are drawn per episode from priors. The agent sets the
controlled inputs of up to ``budget`` experiments, receives the noisy
observations, and answers. A numeric target is correct within ``tolerance``
(relative); a categorical target such as a functional form must match. The
interface follows Gymnasium (``reset(seed)``, ``step(action)``, ``state``), and
episodes are reproducible from their seed. See
``examples/example_9_discovery_env``.

.. _checker:

Model Checker and LLM-Drafted Models
====================================

:func:`simulator.checker.check_model` (or ``python -m simulator.checker
module.Class config.yaml``) is the gate for new and generated models:

* a static scan of the source: global random-number use, eval/exec, subprocess,
  network modules, file deletion;
* runtime checks: the model runs and its outputs validate; outputs are finite;
  roles are declared; the same seed reproduces the outputs; the global random
  state is untouched; the config is not modified; fidelity and units are valid;
* the observed convergence order when ``dt`` is halved;
* the drift of quantities from ``invariants(results)``;
* that every parameter in ``inferable_parameters`` changes the output.

:func:`simulator.checker.check_source` runs the same checks on source code in a
separate, time-limited process, after the static scan.
:func:`simulator.llm.draft_model` asks any LLM (wrapped with
``CallableLLMClient``) for a model, gates it, and retries with the checker's
findings. Accepted drafts are still ``unverified``: passing the gate never makes
a model ``reference``.

.. _data_handling:

Data Handling and Storage
=========================

Arrays are stored as ``.npy`` (or one ``data.h5`` with ``storage: {format: hdf5}``),
DataFrames as Parquet (CSV without pyarrow). The record lists every file with
its SHA-256; :func:`simulator.persistence.load_experiment_record` verifies the
checksums (``verify=False`` to skip) and can memory-map arrays (``mmap=True``).
``storage: {format: inline}`` keeps the original JSON-only layout.

Inputs can be loaded with ``simulator.data_handler`` (``load_csv``,
``load_json``, ``load_numpy``); ``save_csv`` / ``results_to_dataframe`` flatten
results into one table.

.. _doe:

Design of Experiments
=====================

.. code-block:: python

    from simulator.doe import create_doe_table, run_design

    design = create_doe_table({"prey_growth_rate": [0.8, 1.4],
                               "observation.operators.0.snr_db": [10, 40]},
                              design_type="lhs", n_samples=32, seed=1)
    out = run_design(base_config, design, output_dir="lhs_study",
                     replicates=5, base_seed=7, n_jobs=4)

* ``create_doe_table``: ``full_factorial`` (lists of levels) or ``lhs``,
  ``sobol``, ``halton``, ``random`` (``[low, high]`` or
  ``{low, high, scale: log, integer: true}``).
* Dotted keys set nested config entries.
* ``run_design`` adds ``design_point``, ``replicate`` and ``seed`` columns
  (``seed_mode="per_row"`` or ``"per_replicate"`` for common random numbers),
  runs every row through the engine and writes ``design.csv``,
  ``design_manifest.json`` and ``design_results.csv``.
* ``n_jobs > 1`` uses worker processes; scripts must then guard their entry
  point with ``if __name__ == "__main__":``.
* ``run_parameter_sweep`` is a full-factorial convenience wrapper;
  ``append_results_to_doe_table`` flattens array outputs into columns.

.. _visualization:

Visualization
=============

Plots are generated from descriptors: ``group`` selects the plot family
(``time_series``, ``histogram``), ``plot_type`` the mark (``line``, ``scatter``)
and ``x_axis`` the coordinate. Observed series are drawn as markers over the
latent curves. matplotlib PNG and plotly HTML are always written; static plotly
export (``static_plot_format``) needs kaleido. Plot failures are logged in the
record and never abort a run.

.. _experiment_records:

Experiment Records
==================

``experiment_record.json`` contains the experiment ID, timestamps, the config
(including the realised seed), model class and module, descriptors, a manifest
of data files with checksums, log messages, system information, software
versions, ``randomness``, ``provenance`` and ``ground_truth``. Records are also
written when a run fails.

.. code-block:: python

    from simulator.engine import SimulatorEngine

    engine = SimulatorEngine()
    record = engine.load_experiment_record(experiment_id)
    record.ground_truth["parameters"]
    record.output_data["prey_population_observed"]["data"]

.. _testing:

Testing and Verification
========================

.. code-block:: bash

    pytest

``tests/test_verification.py`` checks the bundled models numerically: closed
forms, agreement with a high-accuracy SciPy solver, observed convergence orders
(Euler 1, RK4 4) and conservation of the Lotka-Volterra invariant. Statistical
tests check that observation operators realise the configured noise.

.. _extending:

Extending the Framework
=======================

* **New experiments:** add a package under ``experiments/`` (see
  :ref:`experiment_logic` and ``experiments/README.md``).
* **New observation operators:** :func:`simulator.observation.register_operator`.
* **Variations by inheritance:** subclass an existing model, as
  ``PredatorPreyCalibrationExperiment`` extends ``PredatorPreyExperiment`` to
  overlay observed data.
* **Plot types:** ``simulator/visualization.py``; **inputs:** ``simulator/data_handler.py``.

.. _api_reference:

API Reference
=============

.. automodule:: simulator.engine
   :members: SimulatorEngine, RunOutput

.. automodule:: simulator.base
   :members:

.. automodule:: simulator.utils
   :members:

.. automodule:: simulator.observation
   :members: apply_observation, register_operator, validate_observation_config

.. automodule:: simulator.bundle
   :members: resolve_bundle_config, build_bundle, generate_questions, check_answer

.. automodule:: simulator.envs
   :members: ExperimentEnv, ExperimentTaskSpec, render_observation

.. automodule:: simulator.checker
   :members: check_model, check_source, lint_source, CheckReport

.. automodule:: simulator.llm
   :members: CallableLLMClient, draft_model, Draft

.. automodule:: simulator.units
   :members: check_units, convert_parameters, parse_units

.. automodule:: simulator.export.tasks
   :members: collect_bundle_tasks, write_tasks_jsonl, score

.. automodule:: simulator.export.huggingface
   :members: export_huggingface, write_croissant

.. automodule:: simulator.export.sbi
   :members: BatchSimulator, generate_training_data, training_data_from_runs, to_sbi_prior

.. automodule:: simulator.export.rl
   :members: export_openenv, export_verifiers_qa, verifiers_openenv_taskset_config

.. automodule:: simulator.doe
   :members: create_doe_table, add_replicates_and_seeds, run_design, DesignRun, run_parameter_sweep, append_results_to_doe_table

.. automodule:: simulator.storage
   :members: write_outputs, read_output, resolve_storage_config, ChecksumError

.. automodule:: simulator.persistence
   :members: save_experiment_record, load_experiment_record

.. automodule:: simulator.rng
   :members:

.. _contributing:

Contributing
============

Issues and pull requests are welcome at
https://github.com/sandner-art/scientific-data-simulator. Please add tests for
new features; new models should come with verification tests. The project is
licensed under the Apache License 2.0.
