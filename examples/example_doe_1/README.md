# DOE Example 1: linear function parameter sweep

This example runs a full-factorial sweep of the `LinearFunctionExperiment` over slope `m` and intercept `c` (3 × 3 = 9 runs) and processes the results by hand.

```bash
python -m examples.example_doe_1.run_experiment --plot
python -m examples.example_doe_1.run_experiment --append                 # print the joined results table
python -m examples.example_doe_1.run_experiment --output_format nested
python -m examples.example_doe_1.run_experiment --output_dir my_results
```

## Files

- `config_base.yaml` – the parameters that are *not* swept (`n_points`, `x_min`, `x_max`).
- `run_experiment.py` – defines `param_ranges`, calls `run_parameter_sweep`, then prints and plots the results.

## The DOE workflow

1. **Base configuration.** A YAML file with the fixed parameters.
2. **Design.** `create_doe_table(param_ranges)` builds a full-factorial table. For continuous ranges, use `design_type="lhs" | "sobol" | "halton" | "random"` with `n_samples` and a design `seed`.
3. **Run.**
   - `run_parameter_sweep(cls, base_config, param_ranges, ...)` is the convenience wrapper for full-factorial sweeps.
   - `run_design(base_config, table, replicates=..., base_seed=..., n_jobs=...)` runs any design table.
   - Every run goes through the engine, so each one gets a seed, a full provenance record and data files.
4. **Analyse.**
   - `append_results_to_doe_table(table, results)` flattens the outputs into one row per run.
   - `run_design` also writes `design.csv`, `design_manifest.json` and `design_results.csv` to the output directory.

Any row can be re-run on its own: pass its `seed` (from `design.csv`) together with its parameter values.
