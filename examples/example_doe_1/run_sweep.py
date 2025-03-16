#!/usr/bin/env python3
# examples/example_doe_1/run_doe_example.py

import os
import argparse
import yaml
import numpy as np  # Import numpy
import pandas as pd
from simulator.doe import run_parameter_sweep, create_doe_table, append_results_to_doe_table
from experiments.linear_function.logic import LinearFunctionExperiment  # Import the ExperimentLogic

def main():
    parser = argparse.ArgumentParser(description="Run a DOE example with the LinearFunctionExperiment.")
    parser.add_argument("--config", type=str, default="examples/example_doe_1/config_base.yaml",
                        help="Path to the base configuration file.")
    parser.add_argument("--output_dir", type=str, default="doe_example_results",
                        help="Base directory for output files.")
    parser.add_argument("--output_format", type=str, default="list", choices=['list', 'nested'],
                        help="Format of the output from run_parameter_sweep ('list' or 'nested').")
    parser.add_argument('--append', action='store_true', help='Append results to DOE table')

    args = parser.parse_args()

    # Load the base configuration
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        config_path = os.path.join(project_root, args.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Base configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Define the parameter ranges
    param_ranges = {
        "m": [1.0, 2.0, 3.0],  # Vary the slope
        "c": [-1.0, 0.0, 1.0],  # Vary the intercept
    }

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create the DOE table
    doe_table = create_doe_table(param_ranges)
    print("DOE Table:\n", doe_table)

    # Run the parameter sweep
    results = run_parameter_sweep(
        LinearFunctionExperiment,  # Pass the class, not a string
        base_config,
        param_ranges,
        output_dir=args.output_dir,
        output_transform=args.output_format  # Use the command-line argument
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
            y_mean = np.mean(results_dict['y']['data'])  # Calculate the mean
            print(f"    Mean of 'y': {y_mean:.2f}")

            # Example: Access the full experiment record if needed
            # from simulator.persistence import load_experiment_record
            # record = load_experiment_record(args.output_dir, record_id)
            # print(f"    Experiment Description: {record.experiment_description}")


    elif args.output_format == 'nested':
        print("\nProcessing results (nested format):")
        for param_combo_name, results_dict in results.items():
            print(f"  Parameter Combination: {param_combo_name}")
            # Example: Print the mean of the 'y' values
            y_mean = np.mean(results_dict['y']['data'])
            print(f"    Mean of 'y': {y_mean:.2f}")
    if args.append:
        results_table = append_results_to_doe_table(doe_table, results)
        print(results_table)

if __name__ == "__main__":
    main()