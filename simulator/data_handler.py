# simulator/data_handler.py
import csv
import json
import numpy as np
import pandas as pd
from .utils import DataDescriptor, DataType
from typing import Dict, Any, Union, List, Optional
import os  # Import os


def load_csv(file_path: str, delimiter: str = ',', header: bool = True) -> Dict[str, Any]:
    """
    Loads data from a CSV file.

    Args:
        file_path: Path to the CSV file.
        delimiter: The delimiter character (default: ',').
        header: Whether the CSV file has a header row (default: True).

    Returns:
        A dictionary containing the data and a DataDescriptor.  The data
        will be a Pandas DataFrame.
    """
    try:
        if header:
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            df = pd.read_csv(file_path, delimiter=delimiter, header=None)
            # If no header, assign default column names
            df.columns = [f"col_{i}" for i in range(df.shape[1])]


        descriptor = DataDescriptor(
            name=os.path.basename(file_path),  # Use filename as default name
            data_type=DataType.DATAFRAME,
            shape=df.shape,
            group="input_data", # or other
            # You might add more metadata extraction here (e.g., units from a header row)
        )
        return {"data": df, "descriptor": descriptor}

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {e}") from e

def load_json(file_path: str) ->  Dict[str, Any]:
    """Loads data from a JSON file.
    Returns dict containing loaded data, and `DataDescriptor` instance.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Try to guess data shape
            if isinstance(data, list):
                shape = (len(data),)
            else:
                shape = None
            descriptor = DataDescriptor(name=os.path.basename(file_path), data_type=DataType.LIST, shape=shape, group='input_data')
        return {"data": data, "descriptor": descriptor}

    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error during loading JSON file: {e}") from e


def load_numpy(file_path: str) -> Dict[str, Any]:
    """
    Loads data from a NumPy .npy or .npz file.
    Returns:
        A dictionary containing loaded data and DataDescriptor.
    """
    try:
        data = np.load(file_path)
        if isinstance(data, np.ndarray):
            descriptor =  DataDescriptor(name=os.path.basename(file_path), data_type=DataType.NDARRAY, shape = data.shape, group='input_data')
            return {"data": data, "descriptor": descriptor}
        elif isinstance(data, np.lib.npyio.NpzFile):
            # Handle .npz files (archives containing multiple arrays)
            result_data = {}
            for key in data.files: # Iterate through arrays with names in archive
                arr = data[key]
                descriptor = DataDescriptor(name=key, data_type=DataType.NDARRAY, shape=arr.shape, group='input_data')
                result_data[key] =  {"data": arr, "descriptor": descriptor}
            return result_data # return dict of data
        else:
            raise TypeError(f"Unsupported NumPy file format: {type(data)}")

    except FileNotFoundError:
        raise FileNotFoundError(f"NumPy file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading NumPy file: {e}") from e


def create_descriptor_from_data(data: Any, name:str, group: str = 'data', plot_type: Optional[str] = None, x_axis: Optional[str] = None) -> DataDescriptor:
    """
    Creates a DataDescriptor for data.

    Args:
        data: data for creating descriptor.
        name: Name of data.
        group: Data group (default: 'data').

    Returns:
        DataDescriptor for provided data.
    """
    if isinstance(data, np.ndarray):
        return DataDescriptor(name, DataType.NDARRAY, shape = data.shape, group=group, plot_type=plot_type, x_axis=x_axis)
    elif isinstance(data, pd.DataFrame):
        return DataDescriptor(name, DataType.DATAFRAME, shape = data.shape, group=group, plot_type=plot_type, x_axis=x_axis)
    elif isinstance(data, list):
        return DataDescriptor(name, DataType.LIST, group=group, plot_type=plot_type, x_axis=x_axis) # No shape
    elif isinstance(data, float):
        return DataDescriptor(name, DataType.FLOAT, group=group, plot_type=plot_type, x_axis=x_axis)  # No shape
    elif isinstance(data, int):
        return DataDescriptor(name, DataType.INT, group=group, plot_type=plot_type, x_axis=x_axis)  # No shape
    elif isinstance(data, str):
        return DataDescriptor(name, DataType.STRING, group=group, plot_type=plot_type, x_axis=x_axis)  # No shape
    else:
        raise TypeError(f"Unsupported data type for descriptor creation: {type(data)}")

def save_csv(data: Union[pd.DataFrame, Dict[str, Any]], file_path: str,
             delimiter: str = ',', index: bool = False) -> None:
    """
    Saves data to a CSV file.  Handles both DataFrames and results dictionaries.

    Args:
        data: The data to save (either a Pandas DataFrame or a dictionary of results).
        file_path: The path to the CSV file to be created.
        delimiter:  Delimiter to use.
        index: Add index to output or not.
    """
    try:
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            # Convert results dictionary to DataFrame
            df = pd.DataFrame()
            for data_name, data_info in data.items():
                if isinstance(data_info, dict) and 'data' in data_info: # Check if the structure is right
                    if isinstance(data_info['data'], (np.ndarray, list)):
                         df[data_name] = pd.Series(data_info['data']) # use pandas series
                    elif isinstance(data_info['data'], (int, float, str)):
                        df[data_name] = [data_info['data']] # scalar values to list

        else:
            raise TypeError(f"Unsupported data type for CSV export: {type(data)}")
        df.to_csv(file_path, sep=delimiter, index=index)


    except Exception as e:
        raise RuntimeError(f"Error saving data to CSV: {e}") from e