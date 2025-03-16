# Parameter Sweep Example: Predator-Prey

This example demonstrates how to perform a parameter sweep using the `Scientific Data Simulator`.  It uses the `PredatorPreyExperiment` to simulate predator-prey population dynamics and explores the effect of varying the `prey_growth_rate` and `prey_death_rate` parameters.

## Files

*   **`run_sweep.py`:**  The script that sets up and runs the parameter sweep.  It:
    *   Loads a base configuration from `config_base.yaml`.
    *   Defines the parameter ranges to sweep.
    *   Calls the `run_parameter_sweep` function (from `simulator.doe`) to execute the simulations.
    *   Creates a DOE table using `create_doe_table`.
    *   Appends the results to the DOE table using `append_results_to_doe_table`.
    *   Performs some basic analysis and plotting (you can customize this).
*   **`config_base.yaml`:**  A base configuration file that defines the parameters that are *not* being varied in the sweep.  The parameters that *are* being varied are defined within the `run_sweep.py` script itself.

## Running the Example

To run this example, navigate to the project root directory in your terminal and execute:

```bash
python -m examples.parameter_sweep_example.run_sweep
```

This will create a directory named `parameter_sweep_results` (or the directory you specify with the `--output_dir` argument) containing:

*   Subdirectories for *each* individual simulation run (named with timestamps and UUIDs).  These subdirectories will contain the `experiment_record.json` file and the generated plots for each run.
*   Output from the script to the console (the DOE table with results, and potentially some analysis results).

You can then analyze the results in the `parameter_sweep_results` directory, using the `experiment_record.json` files, the generated plots, and the printed output.
```

**How to Run:**

1.  **Save:** Save the code above as `examples/parameter_sweep_example/run_sweep.py` and `examples/parameter_sweep_example/config_base.yaml`.
2.  **Execute:** From the project root directory, run:

    ```bash
    python -m examples.parameter_sweep_example.run_sweep
    ```

**Key Points:**

*   **Self-Contained:** This example is self-contained within the `examples/parameter_sweep_example/` directory.
*   **`config_base.yaml`:**  This file contains the *base* configuration.  The parameters that are being varied in the sweep are defined *within* the `run_sweep.py` script itself. This is a common pattern: a base configuration file for fixed parameters, and then the sweep parameters defined separately.
*   **`run_sweep.py`:** This script does the following:
    *   **Argument Parsing:** Uses `argparse` to handle command-line arguments (e.g., `--config` and `--output_dir`).
    *   **Loads Base Config:** Loads the `config_base.yaml` file.
    *   **Defines Parameter Ranges:** Defines the `param_ranges` dictionary, specifying the parameters to vary and their values.
    *   **Creates Output Directory:** Creates the `parameter_sweep_results` directory (or uses the directory specified by `--output_dir`).
    *   **Calls `run_parameter_sweep`:** Calls the `run_parameter_sweep` function from your `simulator.doe` module, passing the `PredatorPreyExperiment` class, the base configuration, the parameter ranges, and the output directory.
    *   **Creates DOE Table:** Calls `create_doe_table` to create a Pandas DataFrame representing the parameter combinations.
    *   **Appends Results:** Calls `append_results_to_doe_table` to combine the DOE table with the simulation results.
    *   **Example Analysis and Plotting:** Includes *example* code to perform basic analysis (calculating the mean final prey population) and generate a plot showing the results.  You would adapt this section to your specific analysis needs.
*   **Modularity:** The example uses the `PredatorPreyExperiment` *without modification*.  It demonstrates how to use existing `ExperimentLogic` implementations with the DOE features.
* **README:** Added clear explanation.

This example provides a complete, runnable demonstration of how to use the parameter sweep functionality.  It shows how to define parameter ranges, run the sweep, analyze the results, and generate plots. It's a good starting point for creating your own parameter sweep experiments. This is now a self-contained, well-documented, and runnable example that demonstrates the parameter sweep functionality of your framework.  It follows all the best practices we've discussed and provides a clear template for users to create their own parameter sweeps.
