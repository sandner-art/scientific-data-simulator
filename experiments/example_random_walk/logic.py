# experiments/example_random_walk/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np

class RandomWalkExperiment(ExperimentLogic):
    def __init__(self, config):
        self.n_steps = config['n_steps']
        self.step_size = config['step_size']
        self.position = 0  # Initialize position
        # Create arrays to store values
        self.positions = [0]
        self.steps = [0]


    def initialize(self, config):
        return {'step': 0, 'position': 0}

    def run_step(self, state, step):
        # Generate a random step (-1 or 1)
        step_direction = np.random.choice([-1, 1])
        new_position = state['position'] + self.step_size * step_direction
        self.steps.append(step + 1)
        self.positions.append(new_position)
        return {'step': step + 1, 'position': new_position}

    def get_results(self):
        return {
            "step": {
                "data": np.array(self.steps),
                "descriptor": DataDescriptor("step", DataType.NDARRAY,  units="steps", group="time_series", x_axis='step')
            },
            "position": {
                "data": np.array(self.positions),
                "descriptor": DataDescriptor("position", DataType.NDARRAY,  units="units", group="time_series", plot_type="line", x_axis="step")
            }
        }