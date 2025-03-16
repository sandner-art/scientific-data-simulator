.. Scientific Data Simulator documentation master file

Scientific Data Simulator - User Guide
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   getting_started
   examples
   experiment_logic
   configuration
   data_handling
   visualization
   experiment_records
   testing
   extending  
   api_reference
   contributing

.. _introduction:

Introduction
============

The Scientific Data Simulator is a Python framework designed for creating, managing,
and executing reproducible scientific simulations and generating synthetic data.
It addresses common challenges in computational science, including:

* **Reproducibility:** Ensuring that experiments can be easily replicated.
* **Extensibility:**  Adapting the framework to new scientific models and algorithms.
* **Data Management:**  Handling data input, output, and provenance effectively.
* **Experiment Design:** Supporting systematic exploration of parameter spaces.

This user guide provides comprehensive instructions on installing, using, and extending
the Scientific Data Simulator.

.. _installation:

Installation
============

It is highly recommended to use a virtual environment for managing project dependencies.

Using `venv` (Recommended)
----------------------------

1. **Create a virtual environment**

   .. code-block:: bash

      python3 -m venv .venv

2. **Activate the environment**

   * **Linux/macOS:**

     .. code-block:: bash

        source .venv/bin/activate

   * **Windows:**

     .. code-block:: powershell

        .venv\Scripts\Activate.ps1

3.  **Install the dependencies:**

    .. code-block:: bash

        pip install -r requirements.txt

4. **Install in editable mode:**

   .. code-block:: bash

        pip install -e .

Using `conda`
---------------

1.  **Create a conda environment:**

    .. code-block:: bash

       conda create -n scientific-data-simulator python=3.9  # Or another Python version

2.  **Activate the environment:**

    .. code-block:: bash

        conda activate scientific-data-simulator

3.  **Install the dependencies:**

    .. code-block:: bash

        pip install -r requirements.txt

4. **Install in editable mode:**

    .. code-block:: bash
   
      pip install -e .
   

.. _getting_started:

Getting Started
===============

The simplest way to get started is to run one of the included examples:

1.  **Clone the repository:**

    .. code-block:: bash

        git clone <your_repository_url>
        cd scientific-data-simulator

2.  **Activate your virtual environment** (see Installation).

3.  **Run an example:**

    .. code-block:: bash

        python -m examples.example_1.run_experiment

    This will execute the example simulation and generate output in the `experiments_output` directory.

.. _examples:

Examples
========

The `examples/` directory contains several example simulations:

*   **`example_1`:**  A simple sine wave simulation.
*   **`example_2`:**  A 1D random walk simulation.
*   **`example_3`:**  A predator-prey simulation (Lotka-Volterra equations).
*   **`example_4`:**  Linear function data generation.
*   **`example_notebook.ipynb`:** An interactive Jupyter Notebook example.

Each example has its own subdirectory containing a `run_experiment.py` script and a `config.yaml` file.  You can run them using:

.. code-block:: bash

    python -m examples.<example_name>.run_experiment

.. _experiment_logic:

Creating New Experiments (ExperimentLogic)
===========================================

The core of the Scientific Data Simulator is the `ExperimentLogic` abstract base class. To create a new experiment, you need to:

1.  **Create a new directory** within the `experiments/` directory (e.g., `experiments/my_new_experiment/`).
2.  **Create a `logic.py` file** within that directory.
3.  **Define a class** that inherits from `simulator.base.ExperimentLogic`.
4.  **Implement the required methods:**
    *   `initialize(self, config)`:  Initializes the simulation state.
    *   `run_step(self, state, step)`:  Executes a single simulation step (if applicable).
    *   `get_results(self)`:  Returns the simulation results as a dictionary, along with `DataDescriptor` instances.

Example (`experiments/my_new_experiment/logic.py`):

.. code-block:: python

    from simulator.base import ExperimentLogic
    from simulator.utils import DataDescriptor, DataType
    import numpy as np

    class MyNewExperiment(ExperimentLogic):
        def __init__(self, config):
            self.param1 = config['param1']

        def initialize(self, config):
            return {"my_data": 0}

        def run_step(self, state, step):
            new_data = state['my_data'] + self.param1 * step
            return {"my_data": new_data}

        def get_results(self):
            # Example - replace with your actual data generation
            data = np.arange(10) * self.param1
            return {
                "my_data": {
                    "data": data,
                    "descriptor": DataDescriptor("my_data", DataType.NDARRAY, shape=data.shape, units="my_units")
                }
            }

.. _configuration:

Configuration (config.yaml)
===========================

Experiments are configured using YAML files.  A typical `config.yaml` file looks like this:

.. code-block:: yaml

    experiment_type: experiments.my_new_experiment.logic.MyNewExperiment  # The Python module path
    experiment_description: "My new experiment description."
    param1: 10  # Experiment-specific parameters
    n_steps: 100

*   **`experiment_type`:**  Specifies the Python module path to your `ExperimentLogic` class.  *This is crucial.*
*   **`experiment_description`:**  A human-readable description of the experiment.
*   **Other parameters:**  Any other parameters required by your `ExperimentLogic` implementation.

.. _data_handling:

Data Handling
=============

The `Scientific Data Simulator` uses `DataDescriptor` to keep track of data and its metainformation.

*Creating `DataDescriptor` objects*

You can create `DataDescriptor` for any data (for example numpy array) using:

.. code-block:: python

        from simulator.utils import DataDescriptor, DataType
        import numpy as np
        data = np.array([1,2,3])
        descriptor = DataDescriptor("my_array", DataType.NDARRAY, shape=data.shape, units="my_units", group="time_series", plot_type='line')

.. _visualization:

Visualization
=============

The `Scientific Data Simulator` automatically generates plots based on the `DataDescriptor` information provided by your `ExperimentLogic`.

*   **`group`:**  The `group` field in the `DataDescriptor` specifies the general type of plot (e.g., "time_series", "histogram").
*   **`plot_type`:** The `plot_type` field provides a more specific hint (e.g., "line", "scatter").
* **`x_axis`**: The field allows to set custom x axis for plotting.

The framework currently supports generating plots using both `matplotlib` (for static PNG images) and `plotly` (for interactive HTML plots). These are saved in experiment output directory.

.. _experiment_records:

Experiment Records
==================

Every experiment run generates an `experiment_record.json` file in the `experiments_output` directory. This file contains a complete, JSON-formatted record of the experiment, including:

*   Experiment ID (UUID)
*   Start and end timestamps
*   Configuration parameters
*   Experiment logic class name and module
*   Input and output data descriptors
*   Output data (converted to JSON-serializable formats)
*   Log messages
*   System information (OS, CPU, RAM, Python version)
*   Software versions (of packages listed in `requirements.txt`)

This record is crucial for reproducibility and for analyzing and comparing experiments.  You can load and inspect experiment records using the `SimulatorEngine.load_experiment_record()` method.

.. code-block:: python

    from simulator.engine import SimulatorEngine

    engine = SimulatorEngine()
    experiment_id = "your_experiment_id"  # Replace with the actual ID
    try:
        record = engine.load_experiment_record(experiment_id)
        print(f"Experiment Description: {record.experiment_description}")
        # Access other data from the record...
    except FileNotFoundError:
        print(f"Experiment record not found for ID: {experiment_id}")

.. _testing:

Testing
=======

The `Scientific Data Simulator` includes a comprehensive suite of unit tests. To run the tests, you'll need to have `pytest` installed (it should be installed if you followed the installation instructions).

From the project root directory, run:

.. code-block:: bash

    pytest

This will execute all the tests in the `tests/` directory.

**Writing Tests:**

When you add new features or experiments, you should also add corresponding tests.

* **Unit tests** should be located at `test_<module_name>.py`
* **Experiment tests** should be named as `test_<experiment_name>.py`

**Test structure:**

Each test should follow structure:

*   Set up any necessary preconditions (e.g., create a configuration dictionary).
*   Call the function or method you're testing.
*   Assert that the results are as expected (using `assert` statements).


.. _extending:

Extending the Framework
=======================

The Scientific Data Simulator is designed to be easily extensible. Here's how you can extend the framework:

* **Adding New Experiments:** Create a new directory in `experiments/`, add a `logic.py` file, and implement the `ExperimentLogic` interface (see :ref:`experiment_logic`).

* **Adding Visualization:** You can add new plot types to `visualization.py` module.

* **Adding Data Input:** Implement new input methods in `data_handler.py` module.

* **Inheriting from Existing Experiments (Advanced):**  You can create variations of existing experiments by using *inheritance*. This allows you to reuse code and avoid duplication.  See the section below for details.

**Using Inheritance for Experiment Variations**

Inheritance is a powerful object-oriented programming technique that allows you to create new classes (called *subclasses* or *derived classes*) that inherit the attributes and methods of existing classes (called *base classes* or *parent classes*).

In the context of the `Scientific Data Simulator`, you can use inheritance to:

*   **Create specialized versions of existing experiments:** For example, you could create a `PredatorPreyCalibrationExperiment` that inherits from the `PredatorPreyExperiment` and adds functionality for loading and comparing with observed data (as shown in Example 6).
*   **Add new features to existing experiments:**  You could add a new method to an existing `ExperimentLogic` class without modifying the original code.
*   **Share common logic between multiple experiments:** If you have several experiments that share some common functionality, you can create a base class that contains the shared logic and then have the individual experiment classes inherit from that base class.

**Example: Predator-Prey Calibration**

The `PredatorPreyCalibrationExperiment` (in `experiments/predator_prey_calibration/logic.py`) provides a concrete example of inheritance.  Here's how it works:

.. code-block:: python

    from simulator.base import ExperimentLogic
    from simulator.utils import DataDescriptor, DataType
    from simulator.data_handler import load_csv
    from experiments.predator_prey.logic import PredatorPreyExperiment  # Import the base class
    import numpy as np

    class PredatorPreyCalibrationExperiment(PredatorPreyExperiment):  # Inherit
        def __init__(self, config):
            super().__init__(config)  # Call the base class constructor
            self.observed_data_info = None
            if 'observed_data_path' in config:
                self.observed_data_info = load_csv(config['observed_data_path'])
        def get_results(self):
            results = super().get_results() # Get result from parent class
            if self.observed_data_info:
                results['observed_data'] = self.observed_data_info
            return results

*   **`class PredatorPreyCalibrationExperiment(PredatorPreyExperiment):`:** This line defines the new class and indicates that it *inherits* from `PredatorPreyExperiment`.
*   **`super().__init__(config)`:** This line calls the constructor of the base class (`PredatorPreyExperiment`). This is *essential* to ensure that the base class is properly initialized.
*   **Adding New Functionality:** The `PredatorPreyCalibrationExperiment` adds a new attribute (`observed_data_info`) and overrides method `get_results()` to handle observed data.

By inheriting from `PredatorPreyExperiment`, the `PredatorPreyCalibrationExperiment` automatically gets all the methods and attributes of the base class (like `initialize`, `run_step`, and the core Lotka-Volterra simulation logic). This avoids code duplication and makes the code more maintainable.

**Key Principles of Inheritance:**

*   **"Is-A" Relationship:**  The subclass *is a* specialized version of the base class.  A `PredatorPreyCalibrationExperiment` *is a* type of `PredatorPreyExperiment`.
*   **Code Reuse:**  Avoid repeating code by inheriting common functionality from a base class.
*   **Polymorphism:**  You can use instances of the subclass wherever instances of the base class are expected.  The `SimulatorEngine` can work with *any* `ExperimentLogic` subclass, regardless of whether it's the base class or a derived class.

.. _api_reference:

API Reference
=============

[TODO: This will be generated automatically by Sphinx from your docstrings.]

.. _contributing:

Contributing
============
[TODO]


Key changes and explanations:

*   **New Section:** Added a new section titled "Using Inheritance for Experiment Variations" within the "Extending the Framework" section.
*   **Explanation of Inheritance:**  Provides a clear explanation of inheritance in the context of the `Scientific Data Simulator`.
*   **Concrete Example:**  Uses the `PredatorPreyCalibrationExperiment` as a concrete example to illustrate how inheritance works.
*   **Key Principles:**  Highlights the key principles of inheritance ("is-a" relationship, code reuse, polymorphism).
* **Creating new experiments:** Added link to :ref:`experiment_logic` in "Adding New Experiments".

This addition to the documentation significantly improves its value to users who want to go beyond the basic examples and create more sophisticated simulations. It also reinforces the importance of good object-oriented design principles in scientific software development. The example code is clear and well-commented, and the explanation of the key principles is concise and easy to understand.
