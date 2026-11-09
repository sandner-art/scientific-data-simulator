# Parameter sweep example: predator–prey

This example sweeps `prey_growth_rate` and `prey_death_rate` (3 × 3 levels) of the `PredatorPreyExperiment`. It prints the final prey population for each combination and can plot every prey trajectory in one figure.

```bash
python -m examples.parameter_sweep_example.run_experiment --plot
python -m examples.parameter_sweep_example.run_experiment --output_dir my_results
```

## Files

- `config_base.yaml` – the fixed parameters (`n_steps`, initial populations, predator rates).
- `run_experiment.py`:
  - defines `param_ranges` and runs `run_parameter_sweep`;
  - joins the results to the design with `append_results_to_doe_table`;
  - reads the final value from column `prey_population_<n_steps>`.

## Output

`experiments_output/parameter_sweep_example/` (or `--output_dir`) contains:

- one run directory per combination, each with its record, ground truth, data files and plots;
- `design.csv`, `design_manifest.json` and `design_results.csv` for the whole sweep;
- `parameter_sweep_plot.png` when `--plot` is given.

The base config uses the model's default forward-Euler step (dt = 1), the 0.1.0 behaviour. For reference-quality trajectories, add `integrator: rk4` and a smaller `dt`; see `experiments/predator_prey/README.md`.
