# simulator/data_handler.py
import csv
import json
import numpy as np
import pandas as pd
from .utils import DataDescriptor, DataType, Role
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

def results_to_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert a results dictionary to one tidy DataFrame.

    * 1-D arrays and lists become columns; columns of different lengths are
      aligned on their index and padded with NaN.
    * DataFrame results contribute their columns as ``<name>.<column>``.
    * Scalars (int, float, str, numpy scalars) are broadcast to a constant column.
    * Higher-dimensional arrays are skipped (store them with the data files).
    * Sampled observations (see ``observation_index``) are placed on the rows of
      the points they were sampled at, with NaN elsewhere, so every row refers to
      a single coordinate value.
    """
    columns: Dict[str, pd.Series] = {}
    scalars: Dict[str, Any] = {}
    index_info = results.get("observation_index")
    sampled_rows = None
    if isinstance(index_info, dict) and isinstance(index_info.get("data"), np.ndarray):
        sampled_rows = np.asarray(index_info["data"], dtype=int)
        full_length = max((len(i["data"]) for i in results.values()
                           if isinstance(i, dict) and isinstance(i.get("data"), np.ndarray)
                           and i["data"].ndim == 1), default=0)

    def _place(name, values):
        descriptor = results[name].get("descriptor")
        role = getattr(descriptor, "role", None)
        if (sampled_rows is not None and len(values) == len(sampled_rows) and len(values) != full_length
                and (name == "observation_index" or role in (Role.OBSERVATION, Role.COORDINATE))):
            spread = np.full(full_length, np.nan)
            spread[sampled_rows] = values
            return pd.Series(spread)
        return pd.Series(values)

    for data_name, data_info in results.items():
        if not (isinstance(data_info, dict) and 'data' in data_info):
            continue
        value = data_info['data']
        if isinstance(value, pd.DataFrame):
            for col in value.columns:
                columns[f"{data_name}.{col}"] = value[col].reset_index(drop=True)
        elif isinstance(value, np.ndarray) and value.ndim > 1:
            continue
        elif isinstance(value, (np.ndarray, list, tuple, pd.Series)):
            columns[data_name] = _place(data_name, np.asarray(value) if not isinstance(value, pd.Series)
                                        else value.to_numpy())
        elif isinstance(value, (int, float, str, np.generic)):
            scalars[data_name] = value.item() if isinstance(value, np.generic) else value
    df = pd.concat(columns, axis=1) if columns else pd.DataFrame(index=range(1 if scalars else 0))
    for name, value in scalars.items():
        df[name] = value
    return df


def save_csv(data: Union[pd.DataFrame, Dict[str, Any]], file_path: str,
             delimiter: str = ',', index: bool = False) -> None:
    """
    Saves data to a CSV file. Handles both DataFrames and results dictionaries
    (see :func:`results_to_dataframe` for how results are flattened).
    """
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        df = results_to_dataframe(data)
    else:
        raise TypeError(f"Unsupported data type for CSV export: {type(data)}")
    try:
        df.to_csv(file_path, sep=delimiter, index=index)
    except Exception as e:
        raise RuntimeError(f"Error saving data to CSV: {e}") from e
