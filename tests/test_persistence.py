# tests/test_persistence.py
import pytest
import os
import json
from simulator.persistence import save_experiment_record, load_experiment_record
from simulator.experiment_record import ExperimentRecord
from simulator.utils import DataDescriptor, DataType
import numpy as np
import pandas as pd
import shutil # For cleanup
from datetime import datetime

# Fixture to create a sample ExperimentRecord
@pytest.fixture
def example_record():
    config = {
        "experiment_type": "some_experiment.logic.SomeExperiment",
        "experiment_description": "Test experiment",
        "param1": 42,
    }
    record = ExperimentRecord(config, None)  # No logic class needed for persistence tests
    record.add_input_data_descriptor("input", DataDescriptor("input", DataType.STRING))
    record.add_output_data("output_array", np.array([1, 2, 3]), DataDescriptor("output_array", DataType.NDARRAY))
    record.add_output_data("output_df", pd.DataFrame({'col1': [4,5], 'col2': [6,7]}), DataDescriptor('output_df', DataType.DATAFRAME))
    record.add_log_message("Test log message")
    record.set_system_info({"os": "testOS"})
    record.set_software_versions({"numpy": "1.2.3"})
    record.set_end_time()
    return record

# Fixture for test directory
@pytest.fixture
def temp_output_dir(tmp_path):  # Use pytest's built-in tmp_path fixture
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    yield str(output_dir) # Return as string.
    shutil.rmtree(output_dir) # Clean up


def test_save_and_load_record(example_record, temp_output_dir):
    """Test saving and loading an ExperimentRecord."""

    # Save the record
    saved_dir = save_experiment_record(example_record, temp_output_dir)

    # Check return
    assert os.path.isdir(saved_dir)
    assert example_record.experiment_id in saved_dir

    # Load the record
    loaded_record = load_experiment_record(temp_output_dir, example_record.experiment_id)

    # Compare the original and loaded records
    assert loaded_record.experiment_id == example_record.experiment_id
    assert loaded_record.experiment_description == example_record.experiment_description
    assert loaded_record.start_time == example_record.start_time
    assert loaded_record.end_time == example_record.end_time  # Compare datetime objects directly
    assert loaded_record.config == example_record.config
    assert loaded_record.experiment_logic_class_name == example_record.experiment_logic_class_name
    assert loaded_record.experiment_logic_module == example_record.experiment_logic_module
    assert loaded_record.system_info == example_record.system_info
    assert loaded_record.software_versions == example_record.software_versions
    assert loaded_record.log_messages == example_record.log_messages

    # Compare input descriptors (using to_dict for comparison)
    assert len(loaded_record.input_data_descriptors) == len(example_record.input_data_descriptors)
    for name, descriptor in example_record.input_data_descriptors.items():
        assert name in loaded_record.input_data_descriptors
        assert loaded_record.input_data_descriptors[name].to_dict() == descriptor.to_dict()

    # Compare output data (using to_dict for descriptors and handling NumPy arrays/DataFrames)
    assert len(loaded_record.output_data) == len(example_record.output_data)
    for name, data_info in example_record.output_data.items():
        assert name in loaded_record.output_data
        assert loaded_record.output_data[name]["descriptor"].to_dict() == data_info["descriptor"].to_dict()

        # Handle different data types
        if isinstance(data_info['data'], np.ndarray):
            assert np.array_equal(loaded_record.output_data[name]["data"], data_info["data"])
        elif isinstance(data_info['data'], pd.DataFrame):
            pd.testing.assert_frame_equal(loaded_record.output_data[name]["data"], data_info['data']) # compare dataframes
        else:
            assert loaded_record.output_data[name]["data"] == data_info["data"]


def test_load_record_not_found(temp_output_dir):
    """Test loading a record with a non-existent experiment ID."""
    with pytest.raises(FileNotFoundError):
        load_experiment_record(temp_output_dir, "nonexistent_id")