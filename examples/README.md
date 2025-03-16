# Examples

This directory contains example scripts and notebooks that demonstrate how to use the Scientific Data Simulator.

## Available Examples

| Example        | Description                                                                                      | Experiment Logic                                                                   | Command to Run                                         |
| :------------- | :----------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- | :----------------------------------------------------- |
| `example_1`    | Simulates a simple sine wave.                                                                    | `experiments.example_experiment.logic.ExampleExperiment`                             | `python -m examples.example_1.run_experiment`          |
| `example_2`    | Simulates a random walk in 1D.                                                                   | `experiments.example_random_walk.logic.RandomWalkExperiment`                       | `python -m examples.example_2.run_experiment`          |
| `example_3`    | Simulates a predator-prey population dynamics model (Lotka-Volterra equations).                | `experiments.predator_prey.logic.PredatorPreyExperiment`                             | `python -m examples.example_3.run_experiment`      |
| `example_4`    | Generates data from a linear function (y = mx + c).                                               | `experiments.linear_function.logic.LinearFunctionExperiment`                        | `python -m examples.example_4.run_experiment`          |
| `example_5`    | Loads data from a CSV file, performs basic analysis (mean, std, min, max), and generates plots. | `experiments.data_analysis.logic.DataAnalysisExperiment`                           | `python -m examples.example_5.run_experiment`    |
| `example_6`    | Calibrates the parameters of the Predator-Prey model using observed data loaded from a CSV file. | `experiments.predator_prey_calibration.logic.PredatorPreyCalibrationExperiment` | `python -m examples.example_6.run_experiment`    |
| `example_notebook` | Interactive example using a Jupyter Notebook.                                                     | (Various)                                                                       | Open `example_notebook.ipynb` in Jupyter Notebook.    |

Each example has its own subdirectory containing a `run_experiment.py` script and a `config.yaml` file.

To run an example, navigate to the project root directory in your terminal and use the command listed in the table above.