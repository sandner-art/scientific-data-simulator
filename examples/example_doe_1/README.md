# DOE Example 1: Linear Function Parameter Sweep

This example demonstrates using the `simulator.doe` module to perform a parameter sweep for the `LinearFunctionExperiment`.  It showcases:

*   Generating a full factorial design using `create_doe_table`.
*   Running simulations for each parameter combination using `run_parameter_sweep`.
*   Accessing and processing the results *manually* (without using `append_results_to_doe_table`).

## Files

*   **`run_doe_example.py`:** The script to run the DOE example.
*   **`config_base.yaml`:**  The base configuration file, defining parameters that are *not* varied in the sweep.

## Running the Example

To run this example, navigate to the project root directory in your terminal and execute:

```bash
python -m examples.example_doe_1.run_experiment --plot
```

You can also specify a different output directory and the output format using command-line arguments:

```bash
python -m examples.example_doe_1.run_experiment --output_dir my_results --output_format nested
python -m examples.example_doe_1.run_experiment --help # To get help
```
By default, result is saved in `parameter_sweep_results` directory.

## Output
The script will:
* Print DOE table.
* Print results of the experiment for each parameter combination.
* Create output files in experiment specific folders in `parameter_sweep_results` directory.
```

**How to Run:**

```bash
python -m examples.example_doe_1.run_doe_example
# or
python -m examples.example_doe_1.run_doe_example --output_format nested
# or
python -m examples.example_doe_1.run_doe_example --output_dir my_results
```

**Key Points:**

*   **`experiment_type` in `config_base.yaml`:**  This points to the `LinearFunctionExperiment` class, demonstrating that you can use *any* `ExperimentLogic` implementation with the DOE features.
*   **`param_ranges` in `run_doe_example.py`:** This defines the parameters to vary (`m` and `c`) and their possible values.
* **Command line arguments:** Added output directory and output format as command line arguments.
*   **`output_transform` Argument:** The `run_doe_example.py` script now takes an optional `--output_format` argument (`list` or `nested`), which is passed to `run_parameter_sweep`. This allows you to control the format of the returned results.
*   **Manual Result Processing:** The `run_doe_example.py` script shows how to *manually* access and process the results, both in the `list` and `nested` formats.  This demonstrates the flexibility of the framework.
*   **README:**  The `README.md` file clearly explains the purpose of the example, the files involved, and how to run the example.

This example provides a clear and concise demonstration of using the `simulator.doe` module's core functionality for designing and running experiments. It complements the previous parameter sweep example by showcasing a different way of working with the results and by using a different `ExperimentLogic` class. This completes a very useful, self-contained, and well-documented example.



---

- Additional info:

# The workflow for using the Design of Experiments (DOE) features within the `Scientific Data Simulator` framework

1.  **Defining Parameter Ranges:** Specifying the parameters you want to vary and their possible values.
2.  **Creating a DOE Table (Optional):** Generating a DataFrame that represents the parameter combinations (useful for visualization and analysis).
3.  **Running the Parameter Sweep:** Using the `run_parameter_sweep` function to execute the simulations for each parameter combination.
4.  **Analyzing the Results:**  Using the output of `run_parameter_sweep` (either a list or a nested dictionary) and the saved experiment records to analyze the results.
5.  **Appending results to table (optional):** Appending results to the DOE table.

**I. Workflow in Detail:**

Here's a step-by-step workflow, with code examples and explanations:

**Step 1: Define Parameter Ranges (`config.yaml` or Python Dictionary)**

You define the parameters you want to vary and their possible values in a Python dictionary.  *This is separate from your main `config.yaml` file*.  The `config.yaml` will define the *base* configuration, and the parameter ranges will define how to *vary* that base configuration.

```python
# Example: Parameter ranges for the PredatorPreyExperiment
param_ranges = {
    "prey_growth_rate": [0.05, 0.1, 0.15],  # Three different growth rates
    "prey_death_rate": [0.01, 0.02],       # Two different death rates
    # "initial_prey": [80, 100, 120] # Example, can be used, if you do not want to change other params.
}

# You can define your base configuration directly in your script for
# a parameter sweep, OR you can load it from a "base_config.yaml" file, which
# is a good practice to separate experiment and sweep definitions

base_config = {
    "experiment_type": "experiments.predator_prey.logic.PredatorPreyExperiment",
    "experiment_description": "Parameter sweep of Predator-Prey model.",
    "n_steps": 200,  # Fixed number of steps for all runs
    "initial_prey": 100,      # Fixed initial prey population
    "initial_predators": 20,  # Fixed initial predator population
    "predator_growth_rate": 0.01,   # Fixed predator growth rate
    "predator_death_rate": 0.05, # Fixed predator death rate
    "save_csv": True,  # Example: enable CSV output
}


```

**Step 2: (Optional) Create a DOE Table:**

This step is optional, but it's useful for visualizing the parameter combinations and for later analysis.

```python
import pandas as pd
from simulator.doe import create_doe_table

doe_table = create_doe_table(param_ranges)
print(doe_table)

#   prey_growth_rate  prey_death_rate
# 0              0.05             0.01
# 1              0.05             0.02
# 2              0.10             0.01
# 3              0.10             0.02
# 4              0.15             0.01
# 5              0.15             0.02

```

**Step 3: Run the Parameter Sweep:**

```python
from simulator.doe import run_parameter_sweep
from experiments.predator_prey.logic import PredatorPreyExperiment  # Import the ExperimentLogic class
import os

# Choose an output directory
output_dir = "parameter_sweep_results"
os.makedirs(output_dir, exist_ok=True) # Make sure the directory exists

# Run the sweep.  Pass the *class* itself, not an instance.
results = run_parameter_sweep(PredatorPreyExperiment, base_config, param_ranges, output_dir=output_dir)

# 'results' is now a list of dictionaries (because we used the default output_transform='list')
# Each dictionary in 'results' contains:
#   'params':  The parameter combination for that run.
#   'results': The results dictionary from the ExperimentLogic.get_results() method.
#   'record_id': The UUID of the experiment run.
```

**Step 4: Analyze the Results:**

You have several options for analyzing the results:

*   **Iterate through `results`:** You can loop through the `results` list and access the `params` and `results` for each run.

    ```python
    for run in results:
        params = run['params']
        results_dict = run['results']
        record_id = run['record_id']
        print(f"Parameters: {params}, Record ID: {record_id}")
        # Access and analyze the results in results_dict...
        # You could, for example, load the full ExperimentRecord using the record_id:
        # from simulator.persistence import load_experiment_record
        # record = load_experiment_record(output_dir, record_id)

    ```

*   **Use the `append_results_to_doe_table` function (Recommended):** This function combines the DOE table (from Step 2) with the flattened results, creating a single DataFrame for easy analysis.

    ```python
    from simulator.doe import append_results_to_doe_table
    results_table = append_results_to_doe_table(doe_table, results)
    print(results_table)
    # Now you can use Pandas functions to analyze the results, for example:
    # print(results_table.describe()) # descriptive statistics.
    # results_table.plot(x='prey_growth_rate', y='prey_population_50')
    ```

*   **Load Individual `ExperimentRecord` Files:** The results of *each* individual run are also saved as `experiment_record.json` files in the `output_dir`, in subdirectories named with the timestamp and UUID. You can load these files using `simulator.persistence.load_experiment_record` if you need to access all the details of a specific run.

**Step 5: (Optional) Further Analysis and Visualization**

With the `results_table` (if you use `append_results_to_doe_table`), you can now use all the power of Pandas and other libraries to perform further analysis and visualization:

*   **Descriptive Statistics:** Calculate summary statistics for different parameter combinations.
*   **Filtering and Grouping:**  Select subsets of the data based on parameter values.
*   **Plotting:** Create plots to visualize the relationship between parameters and results (e.g., scatter plots, line plots, heatmaps). You could use Matplotlib, Plotly, or Seaborn for this.
*   **Statistical Analysis:** Perform statistical tests (e.g., ANOVA) to determine the significance of different parameters.

**Complete, Runnable Example:**

```python
# run_parameter_sweep.py (Example script)
import os
import pandas as pd
from simulator.doe import run_parameter_sweep, create_doe_table, append_results_to_doe_table
from experiments.predator_prey.logic import PredatorPreyExperiment  # Import your ExperimentLogic

# Define your base configuration
base_config = {
    "experiment_type": "experiments.predator_prey.logic.PredatorPreyExperiment",
    "experiment_description": "Parameter sweep of Predator-Prey model.",
    "n_steps": 200,
    "initial_prey": 100,
    "initial_predators": 20,
    "predator_growth_rate": 0.01,
    "predator_death_rate": 0.05,
    "save_csv": True,  # Example: enable CSV output
}

# Define the parameter ranges to sweep
param_ranges = {
    "prey_growth_rate": [0.05, 0.1, 0.15],
    "prey_death_rate": [0.01, 0.02],
}

# Create output directory
output_dir = "parameter_sweep_results"
os.makedirs(output_dir, exist_ok=True)

# Run the parameter sweep
results = run_parameter_sweep(PredatorPreyExperiment, base_config, param_ranges, output_dir=output_dir)

# Create the DOE table
doe_table = create_doe_table(param_ranges)

# Append the results to the DOE table
results_table = append_results_to_doe_table(doe_table, results)

# Now you have the results in a convenient DataFrame:
print(results_table)

# --- Example Analysis ---
# Calculate the mean final prey population for each parameter combination
mean_final_prey = results_table.groupby(['prey_growth_rate', 'prey_death_rate'])['prey_population_200'].mean()
print("\nMean Final Prey Population:")
print(mean_final_prey)


# --- Example Visualization (using matplotlib) ---
import matplotlib.pyplot as plt

plt.figure()
for label, row in results_table.iterrows(): # Use iterrows to iterate through DataFrame rows
    param_combination = ", ".join(f"{k}={v}" for k, v in row[:2].items())  # First 2 columns are parameters
    plt.plot(row[2:], label=param_combination) # Plot from third column.
plt.xlabel("Step")
plt.ylabel("Prey population")
plt.title("Prey Population over Time for Different Parameter Combinations")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'prey_populations_sweep.png')) # save plot
plt.show()

```

To run this:

1.  **Save:** Save the above code as a Python file (e.g., `run_parameter_sweep.py`) in your project's root directory.
2.  **Run:** Execute the script from your project root:

    ```bash
    python run_parameter_sweep.py
    ```

This will:

*   Create a directory named `parameter_sweep_results`.
*   Run the `PredatorPreyExperiment` for all combinations of `prey_growth_rate` and `prey_death_rate`.
*   Save the results of *each* run to a separate subdirectory within `parameter_sweep_results` (with timestamped names).
*   Create a Pandas DataFrame (`results_table`) containing the DOE table and the flattened results.
*   Print the `results_table` to the console.
*   Calculate and print the mean final prey population for each parameter combination.
* Generate a plot and saves it.

This complete example provides a clear, practical, and runnable demonstration of how to use the DOE features of your framework.  It shows the entire workflow, from defining parameter ranges to running the sweep to analyzing and visualizing the results. This is a *very* important step in showcasing the capabilities of your simulator. It also demonstrates *good practices*: separating the parameter sweep logic from the core simulation logic, using Pandas for data analysis, and integrating with the existing `ExperimentRecord` and file saving mechanisms.
