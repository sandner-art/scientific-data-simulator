# tests/test_predator_prey_calibration.py
import pytest
from experiments.predator_prey_calibration.logic import PredatorPreyCalibrationExperiment
import numpy as np
import pandas as pd
import os

@pytest.fixture
def calibration_config(tmp_path):
    # Create a sample observed_data.csv file
    data = {
        'time': [0, 1, 2],
        'prey_population': [100, 90, 85],
        'predator_population': [20, 25, 28]
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "observed_data.csv"
    df.to_csv(csv_file, index=False)


    return {
        'n_steps': 5,
        'initial_prey': 100,
        'initial_predators': 20,
        'prey_growth_rate': 0.1,
        'prey_death_rate': 0.02,
        'predator_growth_rate': 0.01,
        'predator_death_rate': 0.05,
        'observed_data_path': str(csv_file)  # Use the temporary file
    }

def test_calibration_initialize(calibration_config):
    experiment = PredatorPreyCalibrationExperiment(calibration_config)
    # Check if observed_data_info is loaded correctly
    assert experiment.observed_data_info is not None
    assert isinstance(experiment.observed_data_info['data'], pd.DataFrame)
    assert 'time' in experiment.observed_data_info['data'].columns
    assert 'prey_population' in experiment.observed_data_info['data'].columns
    assert 'predator_population' in experiment.observed_data_info['data'].columns

    # Check inherited attributes
    assert experiment.n_steps == 5

def test_calibration_get_results(calibration_config):
    experiment = PredatorPreyCalibrationExperiment(calibration_config)
    results = experiment.get_results()

    assert 'observed_data' in results
    assert isinstance(results['observed_data']['data'], pd.DataFrame)

    # check lengths, they should be trimmed now to observed data length.
    assert len(results['time']['data']) == len(results['observed_data']['data'])
    assert len(results['prey_population']['data']) == len(results['observed_data']['data'])
    assert len(results['predator_population']['data']) == len(results['observed_data']['data'])

def test_calibration_missing_file(tmp_path):
    # Test the case where the observed_data_path is incorrect
    config = {
        'n_steps': 5,
        'initial_prey': 100,
        'initial_predators': 20,
        'prey_growth_rate': 0.1,
        'prey_death_rate': 0.02,
        'predator_growth_rate': 0.01,
        'predator_death_rate': 0.05,
        'observed_data_path': str(tmp_path / "missing.csv")  # Non-existent file
    }
    experiment = PredatorPreyCalibrationExperiment(config)  # Should not raise an error
    assert experiment.observed_data_info is None  # observed_data_info should be None