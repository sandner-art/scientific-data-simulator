# Examples

This directory contains example scripts and notebooks that demonstrate how to use the Scientific Data Simulator.

## Available Examples

| Example        | Description                                      | Experiment Logic                       | Command to Run                                      |
| :------------- | :----------------------------------------------- | :------------------------------------- | :-------------------------------------------------- |
| `example_1`    | Simulates a simple sine wave.                   | `experiments.example_experiment.logic.ExampleExperiment` | `python -m examples.example_1.run_experiment`     |
| `example_2`    | Simulates a random walk in 1D.                  | `experiments.example_random_walk.logic.RandomWalkExperiment` | `python -m examples.example_2.run_experiment`      |
| `example_notebook` | Interactive example using a Jupyter Notebook. |  (Various)                             | Open `example_notebook.ipynb` in Jupyter Notebook. |

Each example has its own subdirectory containing a `run_experiment.py` script and a `config.yaml` file.

To run an example, navigate to the project root directory in your terminal and use the command listed in the table above.