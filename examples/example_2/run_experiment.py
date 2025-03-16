#!/usr/bin/env python3
# examples/example_1/run_experiment.py
import argparse
import os
from simulator.engine import SimulatorEngine

def main():
    parser = argparse.ArgumentParser(description="Run an example experiment.")
    parser.add_argument("--config", type=str, default="examples/example_1/config.yaml",  # UPDATED PATH
                        help="Path to the configuration file.")
    args = parser.parse_args()

    # --- Corrected Path Handling ---
    # If the provided config path is absolute, use it directly.
    if os.path.isabs(args.config):
        config_path = args.config
    else:
        # Otherwise, make it relative to the project root (where we expect to run from)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        config_path = os.path.join(project_root, args.config)

    # --- Check if the config file exists ---
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    engine = SimulatorEngine()
    experiment_id = engine.run_experiment(config_path)
    print(f"Experiment completed. ID: {experiment_id}")

if __name__ == "__main__":
    main()