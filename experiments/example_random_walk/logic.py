# experiments/example_random_walk/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np

class RandomWalkExperiment(ExperimentLogic):
    def __init__(self, config):
        self.n_steps = config['n_steps']
        self.step_size = config['step_size']
        self.position = 0  # Initialize position

    def initialize(self, config):
        return {'step': 0, 'position': 0}

    def run_step(self, state, step):
        # Generate a random step (-1 or 1)
        step_direction = np.random.choice([-1, 1])
        new_position = state['position'] + self.step_size * step_direction
        return {'step': step + 1, 'position': new_position}

    def get_results(self):
        # In a real experiment, you might store the position at each step.
        # For simplicity, we'll just generate all positions here.
        positions = [0]  # Start at position 0
        current_position = 0
        for _ in range(self.n_steps):
            step_direction = np.random.choice([-1, 1])
            current_position += self.step_size * step_direction
            positions.append(current_position)

        steps = np.arange(self.n_steps + 1)  # Include initial position
        positions = np.array(positions)

        return {
            "step": {
                "data": steps,
                "descriptor": DataDescriptor("step", DataType.NDARRAY, shape=steps.shape, units="steps", group="time_series",  plot_type="line", x_axis="step") # Added x_axis
            },
            "position": {
                "data": positions,
                "descriptor": DataDescriptor("position", DataType.NDARRAY, shape=positions.shape, units="units", group="time_series", plot_type="line", x_axis="step")  # Specify x_axis
            }
        }