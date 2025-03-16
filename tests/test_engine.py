# tests/test_engine.py
import pytest
from simulator.engine import SimulatorEngine
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
from simulator.experiment_record import ExperimentRecord
import os
import json
import shutil  # Import shutil for directory removal
import yaml

# Create a dummy ExperimentLogic for testing - NO!  Use a real one.
# class DummyExperiment(ExperimentLogic):
#     ...

# Fixture to create a temporary directory for each test
@pytest.fixture
def temp_test_dir(tmp_path):
    test_dir = tmp_path / "test_output"
    test_dir.mkdir()
    yield str(test_dir)  # Provide the directory path as a string
    shutil.rmtree(test_dir)  # Clean up after the test


def test_run_experiment_valid_config(temp_test_dir):
    """Test running an experiment with a valid configuration."""
    engine = SimulatorEngine(output_dir=temp_test_dir)
    config_path = os.path.join(temp_test_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            "experiment_type": "experiments.example_experiment.logic.ExampleExperiment",  # CORRECT
            "experiment_description": "Test experiment",
            "n_steps": 3,
            "amplitude": 5,  # Correct parameter for ExampleExperiment
        }, f)

    experiment_id = engine.run_experiment(str(config_path))
    assert experiment_id is not None

    # Check that the experiment record was created
    # and that the experiment output directory exists.
    # --- Corrected: Search for the directory with the UUID ---
    experiment_dir = None
    for item in os.listdir(temp_test_dir):
        item_path = os.path.join(temp_test_dir, item)
        if os.path.isdir(item_path) and experiment_id in item:
            experiment_dir = item_path
            break

    assert experiment_dir is not None, f"Experiment directory not found in {temp_test_dir}"

    # *Now* construct the record path correctly
    record_path = os.path.join(experiment_dir, "experiment_record.json")
    assert os.path.exists(record_path)
    # --- End Correction ---

    # Check content of the record
    # Load the record and check its contents
    record = engine.load_experiment_record(experiment_id)  # Use the returned ID
    assert record.config['amplitude'] == 5.0
    assert record.experiment_description == "Test experiment"
    assert len(record.output_data) == 2
    assert record.output_data['time']['descriptor'].name == "time"
    assert record.output_data['value']['data'][0] == 0.0



def test_run_experiment_invalid_logic(temp_test_dir):
    """Test running an experiment with an invalid ExperimentLogic class."""
    engine = SimulatorEngine(output_dir=temp_test_dir)
    config_path = os.path.join(temp_test_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump({
            "experiment_type": "test_engine.InvalidExperiment",  # Invalid class
            "experiment_description": "Test experiment",
        }, f)

    with pytest.raises(ImportError):
        engine.run_experiment(config_path)


def test_run_experiment_missing_config(temp_test_dir):
    """Test running an experiment with a missing config file."""
    engine = SimulatorEngine(output_dir=temp_test_dir)
    config_path = os.path.join(temp_test_dir, "missing_config.yaml")  # Non-existent file

    with pytest.raises(FileNotFoundError):
        engine.run_experiment(config_path)

def test_validate_results_invalid_data(temp_test_dir):
    """Test _validate_results with invalid data types."""
    engine = SimulatorEngine(output_dir=temp_test_dir)

    # Create a dummy results dictionary with an invalid data type
    invalid_results = {
        "test_data": {
            "data": "not a list",  # Should be a list according to the descriptor
            "descriptor": DataDescriptor("test_data", DataType.LIST)
        }
    }
    with pytest.raises(TypeError):
        engine._validate_results(invalid_results)

    invalid_results2 = {
        "test_data": {
            "data": "not a float",
            "descriptor": DataDescriptor("test_data", DataType.FLOAT)
        }
    }
    with pytest.raises(TypeError):
        engine._validate_results(invalid_results2)

def test_load_experiment_record_nonexistent(temp_test_dir):
    """Test loading a non-existent experiment record."""
    engine = SimulatorEngine(output_dir=temp_test_dir)
    with pytest.raises(FileNotFoundError):
        engine.load_experiment_record("nonexistent_id")