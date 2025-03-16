# experiments/data_analysis/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
from simulator.data_handler import load_csv
import numpy as np
import pandas as pd
import os # For the file name

class DataAnalysisExperiment(ExperimentLogic): # Corrected Class name
    def __init__(self, config):
        self.config = config
        self.data_info = None  # Store loaded data and descriptor here
        self.data = None

    def initialize(self, config):
        # Load the data in the initialization step
        self.data_info = load_csv(config['input_data_path'])
        self.data = self.data_info['data']  # Access the DataFrame
        return {} # No persistent state

    def run_step(self, state, step):
        pass  # No simulation steps

    def get_results(self):
        # Perform some basic analysis (example)
        mean_value = self.data['value'].mean()
        std_value = self.data['value'].std()
        max_value = self.data['value'].max()
        min_value = self.data['value'].min()

        # Convert time and value for plotting
        time_data = self.data['time'].to_numpy()
        value_data = self.data['value'].to_numpy()

        return {
            "time": { # Return time
                "data": time_data,
                "descriptor": DataDescriptor("time", DataType.NDARRAY, shape=time_data.shape, units="seconds", group="time_series")
            },
            "value": { # Return values
                "data": value_data,
                "descriptor": DataDescriptor("value", DataType.NDARRAY, shape=value_data.shape, units="measurement", group="time_series", plot_type='line', x_axis='time')
            },
            "mean": {
                "data": mean_value,
                "descriptor": DataDescriptor("mean", DataType.FLOAT, units="measurement", group="summary")
            },
            "std": {
                "data": std_value,
                "descriptor": DataDescriptor("std", DataType.FLOAT, units="measurement", group="summary")
            },
            "max": {
                "data": max_value,
                "descriptor": DataDescriptor("max", DataType.FLOAT, units="measurement", group="summary")
            },
            "min": {
                "data": min_value,
                "descriptor": DataDescriptor("min", DataType.FLOAT, units="measurement", group="summary")
            },
            # Return also loaded data
            "loaded_data": {
              "data": self.data,
              "descriptor": self.data_info['descriptor']
            }

        }