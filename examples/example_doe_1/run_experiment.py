#!/usr/bin/env python3
# examples/example_doe_1/run_doe_example.py

import os
import argparse
import yaml
import numpy as np
import pandas as pd
from simulator.doe import run_parameter_sweep, create_doe_table, append_results_to_doe_table
from experiments.linear_function.logic import LinearFunctionExperiment  # Import
from simulator.visualization import generate_plots
import matplotlib.pyplot as plt  # Import matplotlib


def main():
    parser = argparse.ArgumentParser(description="Run a DOE example with the LinearFunctionExperiment.")
    parser.add_argument("--config", type=str, default="examples/example_doe_1/config_base.yaml",
                        help="Path to the base configuration file.")
    parser.add_argument("--output_dir", type=str, default="experiments_output",
                        help="Base directory for output files.")
    parser.add_argument("--output_format", type=str, default="list", choices=['list', 'nested'],
                        help="Format of the output from run_parameter_sweep ('list' or 'nested').")
    parser.add_argument('--append', action='store_true', help='Append results to DOE table')
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
        "m": [1.0, 2.0, 3.0],  # Vary the slope
        "c": [-1.0, 0.0, 1.0],  # Vary the intercept
    }

    # --- Create Sweep-Specific Output Directory ---
    example_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))  # "example_doe_1"
    sweep_output_dir = os.path.join(args.output_dir, example_name)
    os.makedirs(sweep_output_dir, exist_ok=True)

    # --- Create DOE Table (same as before) ---
    doe_table = create_doe_table(param_ranges)
    print("DOE Table:\n", doe_table)

    # --- Run Parameter Sweep (same as before) ---
    results = run_parameter_sweep(
        LinearFunctionExperiment,  # Pass the class
        base_config,
        param_ranges,
        output_dir=sweep_output_dir, # Use the sweep output dir
        output_transform=args.output_format
    )

    # --- Manual Result Processing (Illustrative) ---
    if args.output_format == 'list':
        print("\nProcessing results (list format):")
        for run_data in results:
            params = run_data['params']
            results_dict = run_data['results']
            record_id = run_data['record_id']

            # Example: Print the parameters and the mean of the 'y' values
            print(f"  Parameters: {params}, Record ID: {record_id}")
            y_mean = np.mean(results_dict['y']['data'])
            print(f"    Mean of 'y': {y_mean:.2f}")

    elif args.output_format == 'nested':
        print("\nProcessing results (nested format):")
        for param_combo_name, results_dict in results.items():
            print(f"  Parameter Combination: {param_combo_name}")
            # Example: Print the mean of the 'y' values
            y_mean = np.mean(results_dict['y']['data'])
            print(f"    Mean of 'y': {y_mean:.2f}")

    # --- Append Results (optional) ---
    if args.append:
        results_table = append_results_to_doe_table(doe_table, results)
        print("\nResults Table:")
        print(results_table)

    # --- Generate Plots (CORRECTED) ---
    if args.plot:
        if args.output_format == 'list':
            for run_data in results:
                record_id = run_data['record_id']
                # Construct the path to the experiment directory
                experiment_dir = None
                for item in os.listdir(sweep_output_dir):
                    item_path = os.path.join(sweep_output_dir, item)
                    if os.path.isdir(item_path) and record_id in item:
                        experiment_dir = item_path
                        break
                if experiment_dir is not None:
                    # Call generate_plots with the experiment directory and static format
                    generate_plots(run_data['results'], experiment_dir, static_format = args.static_plot_format)
                else:
                    print("No output directory")

        elif args.output_format == 'nested':
            # In this case you will need to load records.
            print("\nPlotting for nested format is not supported")

        # --- Combined Sweep Plot (NEW) ---
        plt.figure()
        for run_data in results:
            params = run_data['params']
            results_dict = run_data['results']
            # Create a label for the plot legend
            label = f"m={params['m']}, c={params['c']}"
            plt.plot(results_dict['x']['data'], results_dict['y']['data'], label=label)

        plt.xlabel("x (x_units)")  # Use units if you have them in your ExperimentLogic
        plt.ylabel("y (y_units)")
        plt.title("Linear Function Parameter Sweep")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(sweep_output_dir, "combined_sweep_plot.png"))  # Save in sweep dir
        plt.show()

if __name__ == "__main__":
    main()