#!/usr/bin/env python3
# examples/parameter_sweep_example/run_experiment.py
import os
import argparse
import yaml
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
        base_config = yaml.safe_load(f)

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

     # --- Example Analysis (Simplified) ---
    try:
        mean_final_prey = results_table.groupby(['prey_growth_rate', 'prey_death_rate'])['prey_population_199'].mean()
        print("\nMean Final Prey Population:\n", mean_final_prey)
    except KeyError:
        print("\nMean Final Prey Population: Could not calculate (n_steps too small?).")

    # --- Plotting (Simplified and Corrected) ---
    if args.plot:
        plt.figure()
        for label, row in results_table.iterrows():
            param_combination = ", ".join(f"{k}={v}" for k, v in row[:2].items())  # First 2 columns
            # Extract the relevant columns *by name*
            prey_columns = [col for col in results_table.columns if 'prey_population_' in col]
            time_columns = [col for col in results_table.columns if 'time_' in col]
            if len(prey_columns) > 0 and len(time_columns) > 0: # added check
                # Extract the data as NumPy arrays for plotting
                time_data = results_table[time_columns].values.flatten()
                prey_data = row[prey_columns].values.flatten()
                # Trim arrays to same length, use min_len for slicing
                min_len = min(len(time_data), len(prey_data))
                plt.plot(time_data[:min_len], prey_data[:min_len], label=param_combination)

        plt.xlabel("Step")
        plt.ylabel("Prey Population")
        plt.title("Predator-Prey Parameter Sweep")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(sweep_output_dir, 'parameter_sweep_plot.png'))  # Use sweep_output_dir
        plt.show()

if __name__ == "__main__":
    main()