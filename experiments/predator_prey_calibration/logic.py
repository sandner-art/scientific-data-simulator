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
        results = super().get_results()  # simulated trajectories from the parent model
        if self.observed_data_info:
            results['observed_data'] = {
                'data': self.observed_data_info['data'],
                'descriptor': self.observed_data_info['descriptor'].replace(role="observation"),
            }
            # Align the simulated series with the observation window and keep the
            # descriptors consistent with the truncated data.
            n_obs = len(results['observed_data']['data'])
            for key in ('time', 'prey_population', 'predator_population'):
                truncated = results[key]['data'][:n_obs]
                results[key]['data'] = truncated
                results[key]['descriptor'] = results[key]['descriptor'].replace(shape=truncated.shape)
        return results
