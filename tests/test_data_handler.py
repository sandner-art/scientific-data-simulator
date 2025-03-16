# tests/test_data_handler.py
import pytest
from simulator.data_handler import load_csv, load_json, load_numpy, save_csv, create_descriptor_from_data
from simulator.utils import DataDescriptor, DataType
import numpy as np
import pandas as pd
import os
import json
import csv

# Fixture to create a temporary directory for test files
@pytest.fixture
def temp_test_files(tmp_path):
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()

    # Create a sample CSV file
    csv_file = test_dir / "test.csv"
    with open(csv_file, "w", newline="") as f:  # Use newline="" for correct CSV writing
        writer = csv.writer(f)
        writer.writerow(["header1", "header2", "header3"])  # Header row
        writer.writerow([1, 2.5, "a"])
        writer.writerow([2, 3.7, "b"])

    # Create a CSV file without a header
    csv_no_header_file = test_dir / "test_no_header.csv"
    with open(csv_no_header_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([4,5.5, "c"])
        writer.writerow([6,7.1, "d"])

    # Create a sample JSON file
    json_file = test_dir / "test.json"
    with open(json_file, "w") as f:
        json.dump([{"key1": "value1", "key2": 10}, {"key1": "value2", "key2": 20}], f)

    # Create sample NumPy files
    npy_file = test_dir / "test.npy"
    np.save(npy_file, np.array([1, 2, 3]))

    npz_file = test_dir / "test.npz"
    np.savez(npz_file, arr1=np.array([4, 5, 6]), arr2=np.array([7, 8, 9]))

    return {
        "csv": str(csv_file),
        'csv_no_header': str(csv_no_header_file),
        "json": str(json_file),
        "npy": str(npy_file),
        "npz": str(npz_file),
        "test_dir": str(test_dir)  # Also return the test directory
    }

def test_load_csv_with_header(temp_test_files):
    """Test loading a CSV file with a header."""
    data_info = load_csv(temp_test_files["csv"])
    assert isinstance(data_info['data'], pd.DataFrame)
    assert data_info['data'].shape == (2, 3)
    assert list(data_info['data'].columns) == ["header1", "header2", "header3"]
    assert data_info['descriptor'].data_type == DataType.DATAFRAME
    assert data_info['descriptor'].shape == (2,3)

def test_load_csv_without_header(temp_test_files):
    data_info = load_csv(temp_test_files['csv_no_header'], header=False)
    assert isinstance(data_info['data'], pd.DataFrame)
    assert data_info['data'].shape == (2,3)
    assert list(data_info['data'].columns) == ['col_0', 'col_1', 'col_2'] # check generated names
    assert data_info['descriptor'].data_type == DataType.DATAFRAME


def test_load_csv_file_not_found():
    """Test loading a non-existent CSV file."""
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent_file.csv")

def test_load_json_valid(temp_test_files):
    """Test loading a valid JSON file."""
    data_info = load_json(temp_test_files["json"])
    assert isinstance(data_info['data'], list)
    assert len(data_info['data']) == 2
    assert data_info['data'][0]["key1"] == "value1"
    assert data_info['descriptor'].data_type == DataType.LIST

def test_load_json_file_not_found():
    """Test loading a non-existent JSON file."""
    with pytest.raises(FileNotFoundError):
        load_json("nonexistent_file.json")

def test_load_json_invalid(temp_test_files):
    invalid_json = os.path.join(temp_test_files['test_dir'] , 'invalid.json') # Corrected
    with open(invalid_json, 'w') as f:
        f.write("invalid json")  # Write invalid JSON content

    with pytest.raises(ValueError):  # Expect a ValueError for invalid JSON
        load_json(str(invalid_json))

def test_load_numpy_npy(temp_test_files):
    """Test loading a NumPy .npy file."""
    data_info = load_numpy(temp_test_files["npy"])
    assert isinstance(data_info['data'], np.ndarray)
    assert np.array_equal(data_info['data'], np.array([1, 2, 3]))
    assert data_info['descriptor'].data_type == DataType.NDARRAY

def test_load_numpy_npz(temp_test_files):
    """Test loading a NumPy .npz file."""
    data_info = load_numpy(temp_test_files["npz"])
    assert isinstance(data_info, dict)
    assert "arr1" in data_info
    assert "arr2" in data_info
    assert np.array_equal(data_info["arr1"]['data'], np.array([4, 5, 6]))
    assert np.array_equal(data_info["arr2"]['data'], np.array([7, 8, 9]))
    assert data_info['arr1']['descriptor'].data_type == DataType.NDARRAY


def test_load_numpy_file_not_found():
    """Test loading a non-existent NumPy file."""
    with pytest.raises(FileNotFoundError):
        load_numpy("nonexistent_file.npy")

def test_create_descriptor_from_data():
    # Test with different data types
    data_np = np.array([1,2,3])
    descriptor = create_descriptor_from_data(data_np, "test_array", group="test")
    assert isinstance(descriptor, DataDescriptor)
    assert descriptor.data_type == DataType.NDARRAY

    data_df = pd.DataFrame({'a': [1,2], 'b': [3,4]})
    descriptor = create_descriptor_from_data(data_df, 'test_df')
    assert descriptor.data_type == DataType.DATAFRAME

    data_list = [1,2,3]
    descriptor = create_descriptor_from_data(data_list, 'test_list')
    assert descriptor.data_type == DataType.LIST

def test_save_csv(temp_test_files):
    """Test saving data to a CSV file."""
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    csv_file = os.path.join(temp_test_files['test_dir'], "output.csv")  # Use the test_dir fixture
    save_csv(data, csv_file)
    assert os.path.exists(csv_file)

    # Verify content by loading it back:
    loaded_data_info = load_csv(csv_file)
    loaded_data = loaded_data_info['data']
    pd.testing.assert_frame_equal(loaded_data, data)  # best way to compare dataframes

    # Check that saving a dictionary works
    results = {
        "time": {
            "data": np.array([1,2,3]),
            "descriptor": DataDescriptor("time", DataType.NDARRAY, shape=(3,), units="seconds", group="time_series")
        },
        "value": {
            "data": np.array([4,3,5]),
            "descriptor": DataDescriptor("value", DataType.NDARRAY, shape=(3,), units="arbitrary", group="time_series", plot_type="line")
        }
    }
    csv_file2 = os.path.join(temp_test_files['test_dir'], "output2.csv")
    save_csv(results, csv_file2)
    assert os.path.exists(csv_file2)
    df = pd.read_csv(csv_file2)
    assert 'time' in df.columns
    assert 'value' in df.columns