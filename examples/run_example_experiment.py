#!/usr/bin/env python3
# examples/run_example_experiment.py
import argparse
# import os # Removed
# import sys # Removed

# sys.path insert is no longer needed when using -m

from simulator.engine import SimulatorEngine

def main():

    parser = argparse.ArgumentParser(description="Run an example experiment.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    # Make config path relative to the script, for easier use
    # script_dir = os.path.dirname(os.path.abspath(__file__)) # Removed
    # config_path = os.path.join(script_dir, args.config)      # Removed

    engine = SimulatorEngine()
    experiment_id = engine.run_experiment(args.config) # Pass args.config directly
    print(f"Experiment completed. ID: {experiment_id}")

if __name__ == "__main__":
    main()