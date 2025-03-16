# experiments/example_experiment/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np

class ExampleExperiment(ExperimentLogic):
    def __init__(self, config):
        self.n_steps = config['n_steps']
        self.amplitude = config['amplitude']

    def initialize(self, config):
        return {'time': 0, 'value': 0}

    def run_step(self, state, step):
        new_time = state['time'] + 1
        new_value = self.amplitude * np.sin(new_time)
        return {'time': new_time, 'value': new_value}

    def get_results(self):
        # In a real experiment, you'd store data during the simulation steps.
        # This is a simplified example.
        time_data = np.arange(self.n_steps)
        value_data = self.amplitude * np.sin(time_data)

        return {
            "time": {
                "data": time_data,
                "descriptor": DataDescriptor("time", DataType.NDARRAY, shape=time_data.shape, units="seconds", group="time_series")
            },
            "value": {
                "data": value_data,
                "descriptor": DataDescriptor("value", DataType.NDARRAY, shape=value_data.shape, units="arbitrary", group="time_series", plot_type="line")
            }
        }