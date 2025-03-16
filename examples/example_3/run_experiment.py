#!/usr/bin/env python3
# examples/example_3/run_experiment.py
import argparse
import os
from simulator.engine import SimulatorEngine

def main():
    parser = argparse.ArgumentParser(description="Run the predator-prey example.")
    parser.add_argument("--config", type=str, default="examples/example_3/config.yaml",
                        help="Path to the configuration file.")
    args = parser.parse_args()

    if os.path.isabs(args.config):
        config_path = args.config
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        config_path = os.path.join(project_root, args.config)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    engine = SimulatorEngine()
    experiment_id = engine.run_experiment(config_path)
    print(f"Experiment completed. ID: {experiment_id}")

if __name__ == "__main__":
    main()