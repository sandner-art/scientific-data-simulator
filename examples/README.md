# Examples

This directory contains example scripts that demonstrate how to use the Scientific Data Simulator.

## Running them

```bash
sds examples list                     # all examples: level, fields, runtime, title
sds examples show NAME                # what it does, its parameters, what it writes
sds examples run NAME [--quick] [--set name=value ...] [--output DIR]
```

The `python -m examples.NAME.run_experiment` commands in the table below also still work.

## Adding an example

1. Make a folder `examples/<name>/` with a `run_experiment.py` that has a `main()`. It should take
   `--output_dir` and write everything there.
2. Add an `example.yaml` next to it:

   ```yaml
   title: Short title
   summary: One sentence on what it shows.
   entry: examples.<name>.run_experiment
   fields: [physics]                 # physics, mathematics, geometry, social, economics, methods, ai
   level: student                    # student | research
   runtime: {full: 20 s, quick: 5 s}
   extras: []                        # pip extras it needs beyond the core
   output_arg: --output_dir          # null if the script cannot take one (it then runs in the output folder)
   quick_args: [--n, "3"]            # fast settings, used by CI and by --quick
   params:                           # what a user may change (types: int, float, str, flag, choice)
     - {name: n, arg: --n, type: int, default: 12, min: 1, max: 500, help: number of cases}
   outputs:                          # what to show after a run (glob patterns)
     - {path: report.md, kind: markdown}
   ```

3. Check it with `sds examples check --run-quick --only <name>`. The test suite and CI check every
   manifest, and CI runs every example in quick mode.

## Available Examples

| Example        | Description                                                                                      | Experiment Logic                                                                   | Command to Run                                         |
| :------------- | :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :----------------------------------------------------- |
| `example_1`    | Simulates a simple sine wave.                                                                    | `experiments.example_experiment.logic.ExampleExperiment`                             | `python -m examples.example_1.run_experiment`          |
| `example_2`    | Simulates a random walk in 1D.                                                                   | `experiments.example_random_walk.logic.RandomWalkExperiment`                       | `python -m examples.example_2.run_experiment`          |
| `example_3`    | Simulates a predator-prey population dynamics model (Lotka-Volterra equations).                | `experiments.predator_prey.logic.PredatorPreyExperiment`                             | `python -m examples.example_3.run_experiment`      |
| `example_4`    | Generates data from a linear function (y = mx + c).                                               | `experiments.linear_function.logic.LinearFunctionExperiment`                        | `python -m examples.example_4.run_experiment`          |
| `example_5`    | Loads data from a CSV file, performs basic analysis (mean, std, min, max), and generates plots. | `experiments.data_analysis.logic.DataAnalysisExperiment`                           | `python -m examples.example_5.run_experiment`    |
| `example_6`    | Overlays observed CSV data on the Predator-Prey model for manual calibration. | `experiments.predator_prey_calibration.logic.PredatorPreyCalibrationExperiment` | `python -m examples.example_6.run_experiment`    |
| `example_7_reference_dataset` | Reference dataset: RK4 ground truth, sampled noisy observations, SNR × replicate design, optional law shift. | `experiments.predator_prey.logic.PredatorPreyExperiment` | `python -m examples.example_7_reference_dataset.run_experiment` |
| `example_8_geometry_benchmark` | Random triangles with exact constructions: diagram bundles with verifiable questions, exported as a Hugging Face dataset with Croissant metadata. | `experiments.geometry.logic.TriangleExperiment` | `python -m examples.example_8_geometry_benchmark.run_experiment --n 20` |
| `example_9_discovery_env` | Interactive environment: run noisy experiments on a predator-prey system with a hidden growth rate, then estimate it; includes a scripted baseline agent. | `experiments.predator_prey.logic.PredatorPreyExperiment` | `python -m examples.example_9_discovery_env.run_experiment --show` |
| `example_10_law_discovery` | Law-discovery suite: 36 hidden laws (9 families × textbook/counterfactual × 2 noise levels) with formula questions, a classical curve-fitting baseline, an interactive formula-discovery environment, and a leak-audited Hugging Face export. | `experiments.law_discovery.logic.HiddenLawExperiment` | `python -m examples.example_10_law_discovery.run_experiment --export` |
| `example_11_testbed` | The simulator as a testbed: quality card and shortcut (blind / metadata) baselines, leak audit with a canary, fresh-twin contamination gap tests, canary probe, and agent-fallacy flags for scripted faulty agents. Writes `report.md`. | `experiments.law_discovery.logic.HiddenLawExperiment` | `python -m examples.example_11_testbed.run_experiment` |
| `example_12_model_roles` | A language model as attacker (answers a release with the evidence removed), question writer (leak-checked paraphrases) and agent (interactive law discovery), through one logged, replayable client; runs offline with a scripted stand-in or with `--model module:function`. | `experiments.law_discovery.logic.HiddenLawExperiment` | `python -m examples.example_12_model_roles.run_experiment` |
| `example_doe_1` | Full-factorial sweep of the linear function. | `experiments.linear_function.logic.LinearFunctionExperiment` | `python -m examples.example_doe_1.run_experiment --plot` |
| `parameter_sweep_example` | Full-factorial sweep of the predator-prey model. | `experiments.predator_prey.logic.PredatorPreyExperiment` | `python -m examples.parameter_sweep_example.run_experiment --plot` |

Each example has its own subdirectory containing a `run_experiment.py` script and a `config.yaml` (or `config_base.yaml`) file.

To run an example, navigate to the project root directory in your terminal and use the command listed in the table above.