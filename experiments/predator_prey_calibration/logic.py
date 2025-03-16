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
            self.observed_data_info = load_csv(config['observed_data_path'])
    def get_results(self):
        results = super().get_results() # Get result from parent class
        if self.observed_data_info:
            results['observed_data'] = self.observed_data_info
        return results