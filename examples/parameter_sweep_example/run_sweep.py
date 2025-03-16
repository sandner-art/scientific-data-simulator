# examples/parameter_sweep_example/run_sweep.py
import os
import argparse
import yaml  # For loading the base config
import pandas as pd
from simulator.doe import run_parameter_sweep, create_doe_table, append_results_to_doe_table
from experiments.predator_prey.logic import PredatorPreyExperiment  # Import directly
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Run a parameter sweep for the Predator-Prey model.")
    parser.add_argument("--config", type=str, default="examples/parameter_sweep_example/config_base.yaml",
                        help="Path to the base configuration file.")
    parser.add_argument("--output_dir", type=str, default="parameter_sweep_results",
                        help="Base directory for output files.")
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

    # Define the parameter ranges (you could also load these from a separate file)
    param_ranges = {
        "prey_growth_rate": [0.08, 0.1, 0.12],  # Example values
        "prey_death_rate": [0.015, 0.02, 0.025], # Example values
    }

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run the parameter sweep
    results = run_parameter_sweep(
        PredatorPreyExperiment,  # Pass the *class* itself
        base_config,
        param_ranges,
        output_dir=args.output_dir
    )

    # Create the DOE table
    doe_table = create_doe_table(param_ranges)

    # Append the results to the DOE table
    results_table = append_results_to_doe_table(doe_table, results)
    print(results_table)

    # Example Analysis and Plotting (Adapt as needed)
    # Calculate mean final prey population (example)
    try:
        mean_final_prey = results_table.groupby(['prey_growth_rate', 'prey_death_rate'])['prey_population_199'].mean() # Check if column exists.
        print("\nMean Final Prey Population:")
        print(mean_final_prey)
    except KeyError: # if column not created
         print("\nMean Final Prey Population: Could not create statistics, n_steps is probably too small.")


    # --- Example Visualization (using matplotlib) ---
    plt.figure()
    for label, row in results_table.iterrows():
        param_combination = ", ".join(f"{k}={v}" for k, v in row[:2].items())  # First 2 columns

        # --- Corrected Plotting ---
        # Extract the relevant columns *by name*
        prey_columns = [col for col in results_table.columns if 'prey_population_' in col]
        time_columns = [col for col in results_table.columns if 'time_' in col]
        if len(prey_columns) > 0 and len(time_columns) > 0: # Check if we have columns
            # Extract data.
            time_data = results_table[time_columns].values.flatten()
            prey_data = row[prey_columns].values.flatten()
            # Trim arrays
            min_len = min(len(time_data), len(prey_data))
            time_data = time_data[:min_len]
            prey_data = prey_data[:min_len]
            plt.plot(time_data, prey_data, label=param_combination)
        # --- End Correction ---

    plt.xlabel("Step")
    plt.ylabel("Prey Population")
    plt.title("Predator-Prey Parameter Sweep")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'parameter_sweep_plot.png'))
    plt.show()

if __name__ == "__main__":
    main()