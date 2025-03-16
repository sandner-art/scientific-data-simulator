# tests/test_data_analysis.py
import pytest
from experiments.data_analysis.logic import DataAnalysisExperiment
from simulator.data_handler import load_csv
import numpy as np
import pandas as pd
import os

@pytest.fixture
def analysis_config(tmp_path):
    # Create a sample CSV file
    data = {'time': [0, 1, 2], 'value': [10, 12, 11]}
    df = pd.DataFrame(data)
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)

    return {
        'input_data_path': str(csv_file)
    }

def test_data_analysis_initialize(analysis_config):
    experiment = DataAnalysisExperiment(analysis_config)
    experiment.initialize(analysis_config)  # Load the data
    assert experiment.data_info is not None
    assert isinstance(experiment.data, pd.DataFrame)
    assert 'time' in experiment.data.columns
    assert 'value' in experiment.data.columns

def test_data_analysis_get_results(analysis_config):
    experiment = DataAnalysisExperiment(analysis_config)
    experiment.initialize(analysis_config)
    results = experiment.get_results()

    assert 'time' in results
    assert 'value' in results
    assert 'mean' in results
    assert 'std' in results
    assert 'max' in results
    assert 'min' in results
    assert 'loaded_data' in results

    # Check a few values using sample standard deviation calculation
    assert np.isclose(results['mean']['data'], 11.0)
    assert np.isclose(results['std']['data'], np.std([10, 12, 11], ddof=1)) # Corrected STD