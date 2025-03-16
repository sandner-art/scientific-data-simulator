# Example 6: Predator-Prey Model Calibration

This example demonstrates calibrating the parameters of the `PredatorPreyExperiment` using "observed" data loaded from a CSV file.

## Files

*   `run_experiment.py`: The script to run the example.
*   `config.yaml`: The configuration file.
*   `observed_data.csv`: Sample observed data of prey and predator populations over time.

## Configuration

The `config.yaml` file specifies the following:
*   `experiment_type`: Points to the `PredatorPreyExperiment` class.
*   `experiment_description`:  A description of the experiment.
*   Initial values for the model parameters (which you will adjust manually).
*   `observed_data_path`: The path to the CSV file containing the observed data.

## Running the Example and Calibration

1.  Run the example:
    ```bash
    python -m examples.example_6.run_experiment
    ```
2.  Inspect the generated plots (in the `experiments_output` directory).  Compare the simulated `prey_population` and `predator_population` with the loaded `observed_data`.
3.  **Manually adjust** the parameters in `config.yaml` (e.g., `prey_growth_rate`, `prey_death_rate`, etc.) to try to get a better fit between the simulation and the observed data.
4.  Re-run the experiment and repeat steps 2 and 3 until you are satisfied with the fit.

**Note:** This example demonstrates a *manual* calibration process.  For more sophisticated calibration, you would typically use optimization algorithms (which could be integrated into the framework in the future).