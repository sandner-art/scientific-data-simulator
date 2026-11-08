#!/usr/bin/env python3
# examples/parameter_sweep_example/run_experiment.py
import os
import argparse
from simulator.config import yaml_load
import pandas as pd
from simulator.doe import run_parameter_sweep, create_doe_table, append_results_to_doe_table
from experiments.predator_prey.logic import PredatorPreyExperiment
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Run a parameter sweep for the Predator-Prey model.")
    parser.add_argument("--config", type=str, default="examples/parameter_sweep_example/config_base.yaml",
                        help="Path to the base configuration file.")
    parser.add_argument("--output_dir", type=str, default="experiments_output",
                        help="Base directory for output files.")
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--static_plot_format', type=str, default="svg", help="Format for static plots")
    args = parser.parse_args()

    # --- Load Base Config (same as before) ---
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        config_path = os.path.join(project_root, args.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Base configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        base_config = yaml_load(f)

    # --- Parameter Ranges (same as before) ---
    param_ranges = {
        "prey_growth_rate": [0.08, 0.1, 0.12],
        "prey_death_rate": [0.015, 0.02, 0.025],
    }

    # --- Create Sweep-Specific Output Directory ---
    example_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))  # "parameter_sweep_example"
    sweep_output_dir = os.path.join(args.output_dir, example_name)
    os.makedirs(sweep_output_dir, exist_ok=True)


    # --- Run Parameter Sweep ---
    results = run_parameter_sweep(
        PredatorPreyExperiment,
        base_config,
        param_ranges,
        output_dir=sweep_output_dir, # Use the sweep-specific directory
        output_transform='list'  # Always return list for consistency
    )

    # --- Create and Append to DOE Table ---
    doe_table = create_doe_table(param_ranges)
    results_table = append_results_to_doe_table(doe_table, results)
    print("Results Table:\n", results_table)

    # --- Example Analysis: final prey population per parameter combination ---
    final_col = f"prey_population_{base_config['n_steps']}"  # trajectories have n_steps + 1 points
    mean_final_prey = results_table.groupby(['prey_growth_rate', 'prey_death_rate'])[final_col].mean()
    print("\nFinal Prey Population:\n", mean_final_prey)

    # --- Plotting: one prey trajectory per run, against that run's own time axis ---
    if args.plot:
        plt.figure()
        for run in results:
            label = ", ".join(f"{k}={v}" for k, v in run['params'].items())
            plt.plot(run['results']['time']['data'], run['results']['prey_population']['data'], label=label)
        plt.xlabel("Step")
        plt.ylabel("Prey Population")
        plt.title("Predator-Prey Parameter Sweep")
        plt.legend(fontsize="small")
        plt.grid(True)
        plt.savefig(os.path.join(sweep_output_dir, 'parameter_sweep_plot.png'))
        plt.show()

if __name__ == "__main__":
    main()