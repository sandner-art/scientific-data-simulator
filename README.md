# Scientific Data Simulator

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
<!-- Add other badges as needed (e.g., build status, code coverage, DOI) -->

## Overview

The Scientific Data Simulator is a Python framework designed for creating and managing reproducible scientific simulations and generating synthetic data. It aims to address the challenges of reproducibility, extensibility, and ease of use in computational science. The framework emphasizes modularity, allowing users to easily define and extend simulation models, incorporate various data sources, and integrate with external tools. A key feature is the optional integration of Large Language Models (LLMs) to assist in experiment design, lowering the barrier to entry for users and accelerating the research process.

## Key Features

*   **Modular Design:** Built on a modular architecture with clear separation of concerns, making it easy to extend and customize.
*   **Extensibility:** Abstract base classes define standard interfaces for experiment logic, LLM interaction, and data handling.
*   **Reproducibility:** Comprehensive experiment records capture all relevant information about each simulation run (parameters, versions, data provenance, system information).
*   **LLM-Assisted Experiment Design:** Optionally use LLMs to help generate experiment code, reducing development time and promoting best practices.
*   **Design of Experiments (DOE):** Built-in support for DOE principles, enabling systematic exploration of parameter spaces.
*   **Data Management:** Promotes best practices in data management, including metadata management, data provenance tracking, and recommendations for data version control.
*   **Experiment Chaining:** Create pipelines of experiments, where the output of one experiment becomes the input of the next.
*   **Parameter Sweeps:** Easily define and execute parameter sweeps to explore the behavior of simulations.
*   **Data Preview:** Interactive data visualization and summary statistics for quick inspection of results.
*   **Jupyter Notebook Integration:** Seamlessly integrate with Jupyter Notebooks for interactive exploration and analysis.
*   **Open Source:** Released under the MIT License, encouraging collaboration and community contributions.

## Installation

It's highly recommended to use a virtual environment to manage the project's dependencies.  This prevents conflicts with other Python projects and ensures reproducibility.  Choose *one* of the following methods:

**A. Using `venv` (Recommended for most users):**

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    ```

    This creates a new virtual environment in a directory named `.venv` (you can choose a different name if you prefer).  It's a good practice to put the environment *inside* your project directory.  The leading `.` makes it a hidden directory on most systems.

2.  **Activate the environment:**

    *   **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```

    *   **Windows (cmd.exe):**
        ```
        .venv\Scripts\activate.bat
        ```

    *   **Windows (PowerShell):**
        ```
        .venv\Scripts\Activate.ps1
        ```

    You should see `(.venv)` (or your environment name) at the beginning of your terminal prompt, indicating that the environment is active.

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    This installs all required packages.

4. **Install in editable mode:**
    ```bash
    pip install -e .
    ```

**B. Using `conda` (If you use Anaconda or Miniconda):**

1.  **Create a conda environment:**

    ```bash
    conda create -n scientific-data-simulator python=3.9  # Or another Python version
    ```

    This creates a new conda environment named `scientific-data-simulator` (you can choose a different name).

2.  **Activate the environment:**

    ```bash
    conda activate scientific-data-simulator
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    Even within a conda environment, it is a good practice to install project specific dependencies using `pip` and `requirements.txt`
4. **Install in editable mode:**
    ```bash
    pip install -e .
    ```

## Usage

The core concept of the Scientific Data Simulator is the separation of the *simulation engine* from the *experiment logic*.

1.  **Define your experiment logic:** Create a Python class that inherits from the `ExperimentLogic` abstract base class (in `simulator/base.py`). Implement the required methods (`initialize`, `run_step`, `get_results`, and optionally `visualize`). Place this class in a file within the `experiments` directory (e.g., `experiments/my_experiment/logic.py`).

2.  **Create a configuration file:** Create a YAML file (e.g., `config.yaml`) to define the experiment parameters, input data sources, and other settings.

3.  **Run the simulation:** Use the provided example scripts in `examples/` folder.

## Running the Example

To run the included example simulation (a simple sine wave):

1.  **Make sure you have activated your virtual environment** (see the Installation instructions above).
2.  **Navigate to the project root directory** in your terminal:
    ```bash
    cd /path/to/scientific-data-simulator  # Replace with the actual path
    ```
3.  **Run the example script:**

    ```bash
    python -m examples.example_1.run_experiment
    ```
    This will execute the example and generate an `experiment_record.json` file and plot files in an `experiments_output` subdirectory.

## Project Structure

```
scientific_data_simulator/
├── simulator/           # The core engine
│   ├── __init__.py
│   ├── base.py          # Abstract base classes (ExperimentLogic, LLMClient)
│   ├── engine.py        # Core engine logic (execution, logging, etc.)
│   ├── config.py       # Configuration management
│   ├── data_handler.py # Data loading and saving
│   ├── visualization.py # Visualization adapters/wrappers
│   ├── utils.py         # Utility functions
│   ├── doe.py           # Design of Experiments functions
│   ├── llm_client.py    # LLM client abstraction
│   └── experiment_record.py  # ExperimentRecord class
│
├── experiments/         # Specific experiment implementations (ExperimentLogic)
│   ├── __init__.py
│   ├── example_experiment/  # For the reusable ExperimentLogic
│   │   ├── __init__.py
│   │   └── logic.py
│   └── ...
│
├── tests/
│   ├── __init__.py
│   ├── test_engine.py
│   ├── test_example_experiment.py  # Tests for the ExperimentLogic
│   └── ...
│
├── examples/            # Example usage scripts/notebooks
│   ├── example_1/        # NEW: Renamed for clarity
│   │   ├── run_experiment.py
│   │   └── config.yaml
│   └── example_notebook.ipynb  # Notebooks can stay at the top level
│
├── docs/
│   ├── conf.py
│   ├── index.rst
│   └── ...
│
├── .gitignore
├── requirements.txt
├── README.md
└── CITATION.cff
```

## Documentation

[TODO: Link to Sphinx documentation once it's built.]


From scientific-data-simulator folder run:

```bash
# From the project root:
python -m examples.example_1.run_experiment

# Or, explicitly specifying the config file:
python -m examples.example_1.run_experiment --config examples/example_1/config.yaml
```

## Citation

If you use Scientific Data Simulator in your research, please cite it as follows:

> Daniel Sandner. (2025). Scientific Data Simulator (Version 0.1.0) [Computer software].  https://github.com/your-username/scientific_data_simulator

You can find more detailed citation information in the [`CITATION.cff`](CITATION.cff) file.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)  <!-- Add this after you get a DOI -->
