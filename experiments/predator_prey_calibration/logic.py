# experiments/predator_prey_calibration/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
from simulator.data_handler import load_csv
from experiments.predator_prey.logic import PredatorPreyExperiment  # Import the base class
import numpy as np

class PredatorPreyCalibrationExperiment(PredatorPreyExperiment):  # Inherit
    def __init__(self, config):
        super().__init__(config)  # Call the base class constructor
        self.observed_data_info = None
        if 'observed_data_path' in config:
            try:
                self.observed_data_info = load_csv(config['observed_data_path'])
            except FileNotFoundError as e:
                print(f"Warning: Observed data file not found: {e}")
                # Don't raise the error; allow the simulation to run
                # even if the observed data is missing (for flexibility).
            except Exception as e:
                print(f"Warning: Error loading observed data: {e}")

    def get_results(self):
        results = super().get_results() # Get result from parent class
        if self.observed_data_info:
            results['observed_data'] = self.observed_data_info
            # Adjust length
            results['time']['data'] = results['time']['data'][:len(results['observed_data']['data'])]
            results["prey_population"]['data'] = results["prey_population"]['data'][:len(results['observed_data']['data'])]
            results["predator_population"]['data'] = results["predator_population"]['data'][:len(results['observed_data']['data'])]
        return results