# experiments/example_random_walk/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np


class RandomWalkExperiment(ExperimentLogic):
    """Symmetric 1-D lattice random walk: x_{k+1} = x_k + step_size * s_k,
    s_k in {-1, +1} with equal probability. Draws come from ``self.rng``."""

    fidelity_level = "reference"
    fidelity_note = ("Exact simulation of the lattice walk; seeded reproducibility and increment "
                     "statistics (mean 0, variance step_size^2) are tested.")
    inferable_parameters = ("step_size",)

    def __init__(self, config):
        self.n_steps = config['n_steps']
        self.step_size = config['step_size']
        self.positions = [0]
        self.steps = [0]

    def initialize(self, config):
        self.get_rng(config)  # engine-provided generator, or one seeded from config['seed']
        self.positions = [0]
        self.steps = [0]
        return {'step': 0, 'position': 0}

    def run_step(self, state, step):
        step_direction = self.get_rng().choice([-1, 1])
        new_position = state['position'] + self.step_size * step_direction
        self.steps.append(step + 1)
        self.positions.append(new_position)
        return {'step': step + 1, 'position': new_position}

    def get_results(self):
        return {
            "step": {
                "data": np.array(self.steps),
                "descriptor": DataDescriptor("step", DataType.NDARRAY, units="steps", group="time_series", x_axis='step', role="coordinate")
            },
            "position": {
                "data": np.array(self.positions, dtype=float),
                "descriptor": DataDescriptor("position", DataType.NDARRAY, units="dimensionless", group="time_series", plot_type="line", x_axis="step", role="latent")
            }
        }
