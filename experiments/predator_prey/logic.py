# experiments/predator_prey/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np

class PredatorPreyExperiment(ExperimentLogic):
    def __init__(self, config):
        self.n_steps = config['n_steps']
        self.initial_prey = config['initial_prey']
        self.initial_predators = config['initial_predators']
        self.prey_growth_rate = config['prey_growth_rate']
        self.prey_death_rate = config['prey_death_rate']
        self.predator_growth_rate = config['predator_growth_rate']
        self.predator_death_rate = config['predator_death_rate']

        # Store data for plotting (can be protected, for use in subclasses)
        self._prey_populations = [self.initial_prey]
        self._predator_populations = [self.initial_predators]
        self._time_points = [0]


    def initialize(self, config):
        return {
            'prey': self.initial_prey,
            'predators': self.initial_predators,
        }

    def run_step(self, state, step):
        prey = state['prey']
        predators = state['predators']

        # Lotka-Volterra equations
        delta_prey = (self.prey_growth_rate * prey) - (self.prey_death_rate * prey * predators)
        delta_predators = (self.predator_growth_rate * prey * predators) - (self.predator_death_rate * predators)

        new_prey = max(0, prey + delta_prey)  # Prevent negative populations
        new_predators = max(0, predators + delta_predators)

        # Store data
        self._prey_populations.append(new_prey)
        self._predator_populations.append(new_predators)
        self._time_points.append(step + 1)

        return {
            'prey': new_prey,
            'predators': new_predators,
        }

    def get_results(self):
        return {
            "time": {
                "data": np.array(self._time_points),
                "descriptor": DataDescriptor("time", DataType.NDARRAY, shape=(self.n_steps + 1,), units="steps", group="time_series")
            },
            "prey_population": {
                "data": np.array(self._prey_populations),
                "descriptor": DataDescriptor("prey_population", DataType.NDARRAY, shape=(self.n_steps + 1,), units="individuals", group="time_series", plot_type="line", x_axis="time")
            },
            "predator_population": {
                "data": np.array(self._predator_populations),
                "descriptor": DataDescriptor("predator_population", DataType.NDARRAY, shape=(self.n_steps+1,), units="individuals", group="time_series", plot_type="line", x_axis="time")
            }
        }