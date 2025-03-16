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


> A step-by-step guide on how to design a new experiment using the structure of the `Scientific Data Simulator`, incorporating all the best practices we are using. This will cover creating the `ExperimentLogic`, configuration, running the experiment, and handling the results.

# **I. Designing a New Experiment: Step-by-Step**

1.  **Define the Scientific Problem/Model:**

    *   **What are you simulating?** Clearly define the system, process, or phenomenon you want to model.
    *   **What are the key variables?** Identify the inputs (parameters) and outputs (results) of your simulation.
    *   **What are the governing equations/rules?** Write down the mathematical equations, algorithms, or logical rules that describe the system's behavior.
    *   **What are the assumptions and limitations?** Be explicit about any simplifying assumptions you're making.

2.  **Create the Experiment Directory:**

    *   Inside the `experiments/` directory, create a new directory for your experiment. Use a descriptive name (e.g., `experiments/my_new_experiment/`).
    *   Inside this new directory, create the following files:
        *   `__init__.py` (an empty file, to make the directory a Python package)
        *   `logic.py` (this will contain your `ExperimentLogic` implementation)
        *   `README.md` (this will describe your experiment)

    ```
    experiments/
        my_new_experiment/
            __init__.py
            logic.py
            README.md
    ```

3.  **Implement the `ExperimentLogic` Class (`logic.py`):**

    *   **Create a Class:** Define a new class that inherits from `simulator.base.ExperimentLogic`.
    *   **`__init__(self, config)`:**
        *   This is the constructor.  It takes the configuration dictionary (`config`) as input.
        *   Use this method to:
            *   Store any configuration parameters as instance variables (e.g., `self.param1 = config['param1']`).
            *   Initialize any internal data structures needed for the simulation.
            * Load any data (if the experiment will be using input data).
    *   **`initialize(self, config)`:**
        *   This method should return the *initial state* of your simulation as a dictionary.
        *   The keys of this dictionary should be descriptive names for the state variables.
        *   This method might not be needed for all experiments (e.g., if you don't have an iterative simulation).
    *   **`run_step(self, state, step)`:**
        *   This method implements a *single step* of your simulation.
        *   It takes the current `state` (a dictionary) and the current `step` number as input.
        *   It should update the state based on the simulation logic.
        *   It should return the *updated* state as a dictionary.
        *   This method is only needed for iterative simulations. If you have a closed-form solution, you can skip this and do everything in `get_results`.
    *   **`get_results(self)`:**
        *   This method returns the *results* of the simulation.
        *   The return value *must* be a dictionary with the following structure:

            ```python
            {
                "data_item_1": {
                    "data": ...,  # The actual data (NumPy array, list, DataFrame, etc.)
                    "descriptor": DataDescriptor(...)  # Metadata about the data
                },
                "data_item_2": {
                    "data": ...,
                    "descriptor": DataDescriptor(...)
                },
                # ... more data items ...
            }
            ```

        *   **`DataDescriptor`:** Use the `DataDescriptor` class (from `simulator.utils`) to provide metadata for *each* data item:
            *   `name`: A descriptive name (e.g., "population_size", "energy", "position").
            *   `data_type`:  Use the `DataType` enum (e.g., `DataType.NDARRAY`, `DataType.FLOAT`).
            *   `shape`: The shape of the data (if applicable).
            *   `units`: The physical units (if applicable).
            *   `group`:  A string indicating how the data should be grouped for plotting (e.g., "time_series", "histogram", "spatial_data").
            *   `plot_type`: A hint for the default plot type (e.g., "line", "scatter", "histogram").
            *   `x_axis`: The name of the data item to use for the x-axis (for time series, this is usually "time" or "step").
    *   **Example (`logic.py`):**  See the examples we've already created (e.g., `experiments/example_experiment/logic.py`, `experiments/predator_prey/logic.py`) for concrete implementations.

4.  **Create the Configuration File (`config.yaml`):**

    *   Create a `config.yaml` file *within the example directory* that will use your new experiment (e.g., `examples/my_new_example/config.yaml`).
    *   **`experiment_type`:**  Set this to the *module path* of your `ExperimentLogic` class.  For example:

        ```yaml
        experiment_type: experiments.my_new_experiment.logic.MyNewExperiment
        ```

    *   **`experiment_description`:**  Provide a short description.
    *   **Parameters:**  Define *all* the parameters needed by your `ExperimentLogic` class's `__init__` method.
    * **`save_csv`:** Set to `True` or `False`, depending on the needs.
    * **`static_plot_format`:** Set to desired format, or `null` if not needed.
    *   **Example (`config.yaml`):**

        ```yaml
        experiment_type: experiments.my_new_experiment.logic.MyNewExperiment
        experiment_description: "My new experiment description."
        param1: 10
        param2: "some_value"
        n_steps: 100
        # ... other parameters ...
        ```

5.  **Create an Example Script (`run_experiment.py`):**

    *   Create a `run_experiment.py` file *within the example directory* (e.g., `examples/my_new_example/run_experiment.py`).
    *   This script should:
        *   Use `argparse` to handle command-line arguments (at least the `--config` option).
        *   Create an instance of `SimulatorEngine`.
        *   Call `engine.run_experiment()` with the path to your `config.yaml` file.

    *   **Example (`run_experiment.py`):**

        ```python
        #!/usr/bin/env python3
        # examples/my_new_example/run_experiment.py
        import argparse
        import os
        from simulator.engine import SimulatorEngine

        def main():
            parser = argparse.ArgumentParser(description="Run my new experiment.")
            parser.add_argument("--config", type=str, default="examples/my_new_example/config.yaml",  # Update default path
                                help="Path to the configuration file.")
            args = parser.parse_args()

            if os.path.isabs(args.config):
                config_path = args.config
            else:
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
                config_path = os.path.join(project_root, args.config)

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            engine = SimulatorEngine()
            experiment_id = engine.run_experiment(config_path)
            print(f"Experiment completed. ID: {experiment_id}")

        if __name__ == "__main__":
            main()
        ```

6.  **Create Example Directory and `README.md`:**

    *   Create directory in `examples`.
    *   Create a `README.md` file *within your example directory* (e.g., `examples/my_new_example/README.md`) that explains:
        *   The purpose of the example.
        *   How to run the example.
        *   The meaning of the configuration parameters.
        *   The expected output.
        * Which experiment logic is targeted.

7.  **Create Experiment `README.md`:**
    * Create `experiments/your_experiment/README.md` to explain the experiment logic.

8.  **Update Top-Level `README.md` Files:**

    *   Update `examples/README.md` to include a brief entry for your new example.
    *   Update `experiments/README.md` to include a brief entry for your new `ExperimentLogic` implementation.

9. **Add tests:**

    *   Add a `tests/test_<your_experiment_name>.py` file with unit tests for your new `ExperimentLogic` class.

10. **Run the Experiment:**

    ```bash
    python -m examples.my_new_example.run_experiment
    ```

11. **Inspect the Output:** Check the `experiments_output` directory for the results (the `experiment_record.json` file and any plots).

12. **Iterate:** Refine your experiment logic, configuration, and documentation as needed.

**Example: Adding a Simple "Constant Value" Experiment**
Let's walk through a complete, minimal example. We'll create a new experiment that simply generates a constant value.
1. Create experiment:
```
mkdir experiments/constant_value
touch experiments/constant_value/{__init__.py,logic.py,README.md}
```
2. **`experiments/constant_value/logic.py`:**

```python
# experiments/constant_value/logic.py
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np

class ConstantValueExperiment(ExperimentLogic):
    def __init__(self, config):
        self.value = config['value']
        self.n_steps = config['n_steps']

    def initialize(self, config):
        return {}  # No state

    def run_step(self, state, step):
        return {}  # No state changes

    def get_results(self):
        data = np.full(self.n_steps, self.value) # creates array filled with value
        return {
            "value": {
                "data": data,
                "descriptor": DataDescriptor("value", DataType.NDARRAY, shape=data.shape, units="constant", group="time_series", plot_type='line', x_axis='step')
            },
            "step": {
                "data": np.arange(self.n_steps),
                "descriptor": DataDescriptor("step", DataType.NDARRAY, shape=(self.n_steps,), units="step", group="time_series", x_axis='step')
            }

        }
```

3.  **`experiments/constant_value/README.md`:**

    ```markdown
    # Constant Value Experiment

    This experiment generates a constant value for a specified number of steps.

    ## Files

    *   `logic.py`: Contains the `ConstantValueExperiment` class.

    ## Model Description

    Generates a constant value.
    ```

4.  **Create the Example Directory:**

    ```bash
    mkdir examples/example_7
    touch examples/example_7/{run_experiment.py, config.yaml, README.md}
    ```

5.  **`examples/example_7/config.yaml`:**

    ```yaml
    experiment_type: experiments.constant_value.logic.ConstantValueExperiment
    experiment_description: "Generates a constant value."
    n_steps: 50
    value: 25.4
    save_csv: True
    ```

6.  **`examples/example_7/run_experiment.py`:**

    ```python
    #!/usr/bin/env python3
    # examples/example_7/run_experiment.py
    import argparse
    import os
    from simulator.engine import SimulatorEngine

    def main():
        parser = argparse.ArgumentParser(description="Run the constant value example.")
        parser.add_argument("--config", type=str, default="examples/example_7/config.yaml",
                            help="Path to the configuration file.")
        args = parser.parse_args()

        if os.path.isabs(args.config):
            config_path = args.config
        else:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            config_path = os.path.join(project_root, args.config)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        engine = SimulatorEngine()
        experiment_id = engine.run_experiment(config_path)
        print(f"Experiment completed. ID: {experiment_id}")

    if __name__ == "__main__":
        main()
    ```

7.  **`examples/example_7/README.md`:**

    ```markdown
    # Example 7: Constant Value

    This example demonstrates generating a constant value over a number of steps.

    ## Configuration

    The `config.yaml` file allows you to configure the following:
    * `n_steps`: Number of steps.
    * `value`: The constant value.

    ## Running

    ```
    python -m examples.example_7.run_experiment
    ```
    ```

8.  **Update `examples/README.md` and `experiments/README.md`:** Add entries for the new example and experiment.

9. **Add tests:** Create `tests/test_constant_value.py`:
```python
import pytest
from experiments.constant_value.logic import ConstantValueExperiment
import numpy as np

@pytest.fixture
def constant_config():
    return {
        'n_steps': 5,
        'value': 25.4
    }

def test_constant_value_initialize(constant_config):
    experiment = ConstantValueExperiment(constant_config)
    assert experiment.initialize(constant_config) == {}

def test_constant_value_run_step(constant_config):
     experiment = ConstantValueExperiment(constant_config)
     # The experiment doesn't use run_step,
     # so we just check that it exists and doesn't raise an error.
     assert experiment.run_step({},0) is not None

def test_constant_value_get_results(constant_config):
    experiment = ConstantValueExperiment(constant_config)
    results = experiment.get_results()
    assert 'value' in results
    assert 'step' in results

    assert results['value']['descriptor'].name == 'value'
    assert results['step']['descriptor'].name == 'step'

    assert isinstance(results['value']['data'], np.ndarray)
    assert len(results['value']['data']) == 5
    assert np.allclose(results['value']['data'], 25.4)
```

10. **Run:**  `python -m examples.example_7.run_experiment`

This example, although simple, demonstrates all the key steps involved in creating a new experiment and integrating it into the framework. You can follow this same process for more complex simulations. Remember to focus on clear separation of concerns, good documentation, and thorough testing.
