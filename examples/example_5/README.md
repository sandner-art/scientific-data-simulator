# Example 5: Data Analysis and Visualization

This example demonstrates loading data from a CSV file, performing basic analysis, and generating plots.

## Files

*   `run_experiment.py`:  The script to run the example.
*   `config.yaml`:  The configuration file.
*   `data.csv`: Sample CSV data file with `time` and `value` columns.

## Configuration

The `config.yaml` file specifies the following:
* `experiment_type`:  Points to `DataAnalysisExperiment`
* `experiment_description`: Description of the experiment.
*   `input_data_path`: The path to the CSV data file.

## Running the Example

```bash
python -m examples.example_5.run_experiment
```

This will generate an experiment_record.json file and plots in the experiments_output directory.

