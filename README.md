# scientific-data-simulator
The Scientific Data Simulator is a Python framework designed for creating and managing reproducible scientific simulations and generating synthetic data. 

# Scientific Data Simulator

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
<!-- Add other badges as needed (e.g., build status, code coverage) -->

## Overview

The Scientific Data Simulator is a Python framework designed for creating and managing reproducible scientific simulations and generating synthetic data.  It aims to address the challenges of reproducibility, extensibility, and ease of use in computational science. The framework emphasizes modularity, allowing users to easily define and extend simulation models, incorporate various data sources, and integrate with external tools. A key feature is the optional integration of Large Language Models (LLMs) to assist in experiment design, lowering the barrier to entry for users and accelerating the research process.

```console
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
├── experiments/         # Specific experiment implementations
│   ├── __init__.py
│   ├── example_experiment/  # Example: A simple experiment
│   │   ├── __init__.py
│   │   └── logic.py     # Implements ExperimentLogic
│   └── ... # Other experiments
│
├── tests/              # Unit and integration tests
│   ├── __init__.py
│   ├── test_engine.py
│   ├── test_example_experiment.py
│   └── ...
│
├── examples/            # Example usage scripts/notebooks
│   ├── run_example_experiment.py
│   ├── example_notebook.ipynb
│   └── ...
│
├── docs/                # Sphinx documentation (optional, but recommended)
│   ├── conf.py
│   ├── index.rst
│   └── ...
│
├── .gitignore           # Files and directories to ignore in Git
├── requirements.txt      # Project dependencies
└── README.md            # Project description and instructions

```

## Key Features

*   **Modular Design:**  Built on a modular architecture with clear separation of concerns, making it easy to extend and customize.
*   **Extensibility:**  Abstract base classes define standard interfaces for experiment logic, LLM interaction, and data handling.
*   **Reproducibility:** Comprehensive experiment records capture all relevant information about each simulation run (parameters, versions, data provenance, system information).
*   **LLM-Assisted Experiment Design:**  Optionally use LLMs to help generate experiment code, reducing development time and promoting best practices.
*   **Design of Experiments (DOE):**  Built-in support for DOE principles, enabling systematic exploration of parameter spaces.
*   **Data Management:**  Promotes best practices in data management, including metadata management, data provenance tracking, and recommendations for data version control.
*   **Experiment Chaining:**  Create pipelines of experiments, where the output of one experiment becomes the input of the next.
*   **Parameter Sweeps:** Easily define and execute parameter sweeps to explore the behavior of simulations.
*   **Data Preview:**  Interactive data visualization and summary statistics for quick inspection of results.
*   **Jupyter Notebook Integration:**  Seamlessly integrate with Jupyter Notebooks for interactive exploration and analysis.
*   **Open Source:** Released under the MIT License, encouraging collaboration and community contributions.

## Installation

```bash
pip install scientific-data-simulator  # This will work once the package is on PyPI
```
```bash
git clone https://github.com/sandner-art/scientific-data-simulator.git 
cd scientific-data-simulator
pip install -e .
```

From scientific-data-simulator folder run:

```bash
python -m examples.run_example_experiment --config examples/config.yaml
```

## Citation

If you use Scientific Data Simulator in your research, please cite it as follows:

> Daniel Sandner. (2025). Scientific Data Simulator (Version 0.1.0) [Computer software].  https://github.com/your-username/scientific_data_simulator

You can find more detailed citation information in the [`CITATION.cff`](CITATION.cff) file.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)  <!-- Add this after you get a DOI -->
