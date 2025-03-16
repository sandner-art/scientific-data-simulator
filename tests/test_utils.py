# tests/test_utils.py
import pytest
from simulator.utils import DataDescriptor, DataType
import numpy as np  # Import numpy
import pandas as pd

def test_data_descriptor_creation():
    """Test creating DataDescriptor instances."""
    descriptor = DataDescriptor("test_data", DataType.NDARRAY, shape=(10, 2), units="meters", group="my_group", plot_type="line")
    assert descriptor.name == "test_data"
    assert descriptor.data_type == DataType.NDARRAY
    assert descriptor.shape == (10, 2)
    assert descriptor.units == "meters"
    assert descriptor.group == "my_group"
    assert descriptor.plot_type == "line"
    assert descriptor.x_axis is None  # Check default value

    descriptor2 = DataDescriptor("test_data2", DataType.LIST, x_axis="time")
    assert descriptor2.x_axis == "time"


def test_data_descriptor_to_dict():
    """Test the to_dict method of DataDescriptor."""
    descriptor = DataDescriptor("test_data", DataType.DATAFRAME, shape=(5, 3), units="seconds", group="data", plot_type="scatter", x_axis = "time")
    data_dict = descriptor.to_dict()
    assert isinstance(data_dict, dict)
    assert data_dict == {
        "name": "test_data",
        "data_type": "pd.DataFrame",  # Check that enum is converted to string
        "shape": (5, 3),
        "units": "seconds",
        "group": "data",
        "plot_type": "scatter",
        "x_axis": "time"
    }

def test_data_type_enum():
    """Test the DataType enum (basic check)."""
    assert DataType.FLOAT == "float"
    assert DataType.INT == "int"
    assert DataType.STRING == "str"
    assert DataType.LIST == "list"
    assert DataType.NDARRAY == "np.ndarray"
    assert DataType.DATAFRAME == "pd.DataFrame"
    assert len(DataType) == 6 # Check that all values present.