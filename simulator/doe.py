# simulator/doe.py

import itertools
import pandas as pd
from typing import Dict, Any, List, Union, Tuple
import numpy as np
from .utils import DataDescriptor, DataType  # Import DataDescriptor and DataType
from .data_handler import create_descriptor_from_data # Corrected import
from .base import ExperimentLogic
from .persistence import save_experiment_record  # For saving results
from .experiment_record import ExperimentRecord # For creating records
import os
import logging

logger = logging.getLogger(__name__)

def generate_parameter_combinations(param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Generates all combinations of parameters from a dictionary of parameter ranges.

    Args:
        param_ranges: A dictionary where keys are parameter names and values
                      are lists of possible values for that parameter.

    Returns:
        A list of dictionaries, where each dictionary represents a single
        combination of parameter values.
    """
    if not param_ranges:
        raise ValueError("param_ranges cannot be empty.")

    keys = param_ranges.keys()
    value_combinations = itertools.product(*param_ranges.values())
    return [dict(zip(keys, values)) for values in value_combinations]


def run_parameter_sweep(experiment_logic_class: type[ExperimentLogic], base_config: Dict[str, Any],
                        param_ranges: Dict[str, List[Any]],
                        output_dir: str = "experiments_output",
                        output_transform: str = 'list') -> Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Runs a parameter sweep for a given ExperimentLogic instance.

    Args:
        experiment_logic_class: The *class* of the ExperimentLogic to use (not an instance).
        base_config: A dictionary of base configuration parameters.
        param_ranges: A dictionary of parameter ranges to sweep.
        output_dir:  The base output directory.  Individual runs will be stored in subdirectories.
        output_transform: How to format final output, 'list' or 'nested'.

    Returns:
        If `output_transform` == 'list':
          A list of dictionaries, where each inner dictionary contains:
            'params': parameter combination
            'results': the results dictionary, as returned by get_results()
            'record_id': the experiment ID (useful for finding the output files)

        If `output_transform` == 'nested':
            A dictionary where keys are parameter combination names, and values
            are dictionaries containing the `results` (same as returned by get_results).

        In either case, the results of *each* individual run are saved to disk
        using the standard `ExperimentRecord` and `save_experiment_record` mechanism.
    """

    if not issubclass(experiment_logic_class, ExperimentLogic):
        raise TypeError("experiment_logic_class must be a subclass of ExperimentLogic")

    if not param_ranges:
        raise ValueError("param_ranges cannot be empty.")

    combinations = generate_parameter_combinations(param_ranges)
    results_list = []
    results_nested = {}

    for combination in combinations:
        # Create a copy of the base config and update with the current combination
        config = base_config.copy()
        config.update(combination)
        # Create an *instance* of the ExperimentLogic class
        experiment_logic_instance = experiment_logic_class(config)

        # Initialize and run the experiment with the updated config
        state = experiment_logic_instance.initialize(config)
        if hasattr(experiment_logic_instance, "run_step"):
            for step in range(config.get("n_steps", 1)):
                state = experiment_logic_instance.run_step(state, step)
        results = experiment_logic_instance.get_results()

        # Create an ExperimentRecord and save the results to disk
        record = ExperimentRecord(config, experiment_logic_class)
        for data_name, data_info in results.items():
            record.add_output_data(data_name, data_info["data"], data_info["descriptor"])

        record_id = record.experiment_id # Get ID
        save_experiment_record(record, output_dir)
        logger.info(f"Parameter sweep run completed. Experiment ID: {record_id}")


        if output_transform == 'list':
            results_list.append({'params': combination, 'results': results, 'record_id': record_id})
        elif output_transform == 'nested':
            # Create a descriptive name for the combination (for the nested dict)
            combination_name = ", ".join(f"{k}={v}" for k, v in combination.items())
            results_nested[combination_name] = results
        else:
            raise ValueError("Invalid output_transform value. Must be 'list' or 'nested'.")


    return results_list if output_transform == 'list' else results_nested


def create_doe_table(param_ranges: Dict[str, List[Any]], design_type: str = 'full_factorial') -> pd.DataFrame:
    """
    Creates a Design of Experiments (DOE) table.

    Args:
        param_ranges: Dictionary of parameter ranges (as in generate_parameter_combinations)
        design_type:  Currently only supports 'full_factorial'

    Returns:
        A Pandas DataFrame representing the DOE table.
    """
    if not param_ranges:
      raise ValueError("param_ranges cannot be empty for DOE table creation.")
    if design_type == 'full_factorial':
        combinations = generate_parameter_combinations(param_ranges)
        df = pd.DataFrame(combinations)
        return df
    else:
        raise ValueError(f"Unsupported design_type: {design_type}")



def append_results_to_doe_table(doe_table: pd.DataFrame, results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Appends simulation results to a DOE table.  Assumes 'list' output from `run_parameter_sweep`.

    Args:
        doe_table: The original DOE table (DataFrame).
        results: The results from `run_parameter_sweep` (with output_transform='list').

    Returns:
        A new DataFrame with the results appended.  Handles different result types.
    """
    if not isinstance(doe_table, pd.DataFrame):
        raise TypeError("doe_table must be a pandas DataFrame")

    if not all('results' in r for r in results):
        raise ValueError("results must contain 'results' key, ensure output_transform='list'")

    # Convert parameter combinations in DOE table to a list of dictionaries
    params_list = doe_table.to_dict('records')

    if len(params_list) != len(results):
        raise ValueError("The number of results must match the number of rows in the DOE table.")

    results_list = [r['results'] for r in results] # take only results dict

    # Flatten nested results (if any), keeping track of data types
    flattened_results = []
    for res in results_list:
        flat_res = {}
        for data_name, data_info in res.items():
            data = data_info['data']
            if isinstance(data, (np.ndarray, list)):
                # If data is array or list create a column for each element.
                for i, val in enumerate(data):
                    flat_res[f"{data_name}_{i}"] = val
            elif isinstance(data, pd.DataFrame):
                # Flatten DataFrame.
                df_dict = data.to_dict(orient='records')  # Convert to list of dicts
                for i, row in enumerate(df_dict):
                    for key, val in row.items():
                        flat_res[f"{data_name}_{key}_{i}"] = val # Add index
            elif isinstance(data, dict):
                for key, val in data.items():
                    flat_res[f"{data_name}_{key}"] = val # Unique names for dict
            else:  # Scalar values
                flat_res[data_name] = data
        flattened_results.append(flat_res)

    results_df = pd.DataFrame(flattened_results)
    combined_df = pd.concat([doe_table, results_df], axis=1)
    return combined_df