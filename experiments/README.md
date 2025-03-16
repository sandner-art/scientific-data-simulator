# Experiment Logic Implementations

This directory contains the core implementations of different scientific models and algorithms, using the `ExperimentLogic` abstract base class.

## Available Experiments

| Experiment                       | Description                                                                                                                               |
| :------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `example_experiment`             | Simulates a simple sine wave. This is a basic example for demonstration.                                                               |
| `example_random_walk`            | Simulates a 1D random walk, demonstrating stochastic processes.                                                                           |
| `predator_prey`                 | Simulates predator-prey population dynamics using the Lotka-Volterra equations.                                                          |
| `linear_function`           | Generates data points from linear function.                                                       |
| `data_analysis`       | Loads data from file, performs basic analysis and visualization.                                                  |
| `predator_prey_calibration`      | Extends the `predator_prey` experiment to include loading and plotting of observed data for calibration purposes.                       |

Each experiment is defined in a separate subdirectory, containing a `logic.py` file (and potentially other supporting modules).