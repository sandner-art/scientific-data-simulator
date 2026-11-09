# experiments/linear_function/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np

class LinearFunctionExperiment(ExperimentLogic):
    fidelity_level = "reference"
    fidelity_note = "Closed form y = m x + c; checked exactly in tests."
    inferable_parameters = ("m", "c")

    def mechanism_graph(self, config):
        return {"directed": True,
                "nodes": [{"id": "x", "kind": "input"}, {"id": "y", "kind": "output"}],
                "edges": [{"source": "x", "target": "y", "sign": "+" if self.m >= 0 else "-",
                           "parameter": "m", "value": self.m, "term": "slope"}]}

    def __init__(self, config):
        self.n_points = config['n_points']
        self.m = config['m']
        self.c = config['c']
        self.x_min = config['x_min']
        self.x_max = config['x_max']
        self.config_units = (config.get('x_units', 'dimensionless'), config.get('y_units', 'dimensionless'))


    def initialize(self, config):
        return {} # No state needed for a simple function

    def run_step(self, state, step):
        pass  # No steps needed

    def get_results(self):
        x_values = np.linspace(self.x_min, self.x_max, self.n_points)
        y_values = self.m * x_values + self.c

        return {
            "x": {
                "data": x_values,
                "descriptor": DataDescriptor("x", DataType.NDARRAY, shape=x_values.shape, units=self.config_units[0], group="time_series", x_axis='x', role="coordinate")
            },
            "y": {
                "data": y_values,
                "descriptor": DataDescriptor("y", DataType.NDARRAY, shape=y_values.shape, units=self.config_units[1], group="time_series", plot_type="line", x_axis='x', role="latent")
            }
        }