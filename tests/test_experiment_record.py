# tests/test_experiment_record.py
import pytest
from simulator.experiment_record import ExperimentRecord
from simulator.utils import DataDescriptor, DataType
import datetime
import numpy as np
import pandas as pd

@pytest.fixture
def example_config():
    return {
        "experiment_type": "some_experiment.logic.SomeExperiment",
        "experiment_description": "Test experiment description",
        "param1": 42,
        "param2": "test_string",
    }

def test_experiment_record_creation(example_config):
    """Test creating an ExperimentRecord instance."""
    record = ExperimentRecord(example_config, None)  # No experiment logic needed

    assert record.experiment_id is not None  # UUID generated
    assert isinstance(record.start_time, datetime.datetime)
    assert record.end_time is None  # Not set yet
    assert record.config == example_config
    assert record.experiment_logic_class_name == ""  # No logic class provided
    assert record.experiment_logic_module == "" # No logic class provided
    assert record.experiment_description == "Test experiment description"
    assert record.input_data_descriptors == {}
    assert record.output_data == {}
    assert record.log_messages == []
    assert record.system_info == {}
    assert record.software_versions == {}
    assert record.llm_usage == {}

def test_experiment_record_add_data(example_config):
    """Test adding input/output data and descriptors."""
    record = ExperimentRecord(example_config, None)

    # Input data descriptor
    input_desc = DataDescriptor("input_data", DataType.NDARRAY, shape=(10,))
    record.add_input_data_descriptor("input_data", input_desc)
    assert "input_data" in record.input_data_descriptors
    assert record.input_data_descriptors["input_data"] == input_desc

    # Output data and descriptor
    output_data = np.array([1, 2, 3])
    output_desc = DataDescriptor("output_data", DataType.NDARRAY, shape=(3,))
    record.add_output_data("output_data", output_data, output_desc)
    assert "output_data" in record.output_data
    assert record.output_data["output_data"]["data"] is output_data  # Check for identity
    assert record.output_data["output_data"]["descriptor"] == output_desc


def test_experiment_record_add_log_message(example_config):
    record = ExperimentRecord(example_config, None)
    record.add_log_message("Test message 1")
    record.add_log_message("Test message 2")
    assert len(record.log_messages) == 2
    assert record.log_messages[0] == "Test message 1"
    assert record.log_messages[1] == "Test message 2"

def test_experiment_record_set_end_time(example_config):
    record = ExperimentRecord(example_config, None)
    record.set_end_time()
    assert isinstance(record.end_time, datetime.datetime)

def test_experiment_record_set_system_info(example_config):
    record = ExperimentRecord(example_config, None)
    system_info = {"os": "linux", "cpu": "x86"}
    record.set_system_info(system_info)
    assert record.system_info == system_info

def test_experiment_record_set_software_versions(example_config):
    record = ExperimentRecord(example_config, None)
    software_versions = {"numpy": "1.23.4", "pandas": "2.0.0"}
    record.set_software_versions(software_versions)
    assert record.software_versions == software_versions

def test_experiment_record_set_llm_usage(example_config):
    record = ExperimentRecord(example_config, None)
    llm_usage = {"tokens": 100, "cost": 0.01}
    record.set_llm_usage(llm_usage)
    assert record.llm_usage == llm_usage

def test_experiment_record_to_dict(example_config):
    """Test serializing the ExperimentRecord to a dictionary."""
    record = ExperimentRecord(example_config, None)
    # Add different types for testing
    record.add_input_data_descriptor("input_desc", DataDescriptor("input_desc", DataType.STRING))
    record.add_output_data("output_ndarray", np.array([1,2,3]),DataDescriptor("output_ndarray", DataType.NDARRAY, shape=(3,)))
    record.add_output_data("output_dataframe", pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}), DataDescriptor("output_dataframe", DataType.DATAFRAME))
    record.add_output_data("output_list", [1, 'a', 2.3], DataDescriptor("output_list", DataType.LIST))
    record.add_output_data("output_float", 5.67, DataDescriptor("output_float", DataType.FLOAT))
    record.set_end_time()  # set end time
    record.set_system_info({'os': 'linux'})
    record_dict = record.to_dict()

    assert isinstance(record_dict, dict)
    assert "experiment_id" in record_dict
    assert "start_time" in record_dict
    assert "end_time" in record_dict
    assert "config" in record_dict
    assert "experiment_logic_class_name" in record_dict
    assert "input_data_descriptors" in record_dict
    assert "output_data" in record_dict
    assert "log_messages" in record_dict
    assert "system_info" in record_dict
    assert isinstance(record_dict["output_data"]["output_ndarray"]["data"], list) # Check numpy array conversion
    assert isinstance(record_dict["output_data"]["output_dataframe"]["data"], list) # Check dataframe conversion
    assert isinstance(record_dict["output_data"]["output_list"]["data"], list)
    assert isinstance(record_dict["output_data"]["output_float"]["data"], float)
    assert record_dict['system_info']['os'] == 'linux'