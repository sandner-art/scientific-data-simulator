# tests/test_linear_function.py

import pytest
from experiments.linear_function.logic import LinearFunctionExperiment
import numpy as np

@pytest.fixture
def linear_function_config():
    return {
        'n_points': 5,
        'm': 2.0,
        'c': 1.0,
        'x_min': 0.0,
        'x_max': 4.0
    }
def test_linear_function_initialize(linear_function_config):
    experiment = LinearFunctionExperiment(linear_function_config)
    # Check that attributes are set correctly
    assert experiment.n_points == 5
    assert experiment.m == 2.0
    assert experiment.c == 1.0
    assert experiment.x_min == 0.0
    assert experiment.x_max == 4.0

    # Check that initialize returns an empty dict (as expected)
    assert experiment.initialize(linear_function_config) == {}

def test_linear_function_run_step(linear_function_config):
     experiment = LinearFunctionExperiment(linear_function_config)
     # The linear function experiment doesn't use run_step,
     # so we just check that it exists and doesn't raise an error.
     assert experiment.run_step({},0) is None

def test_linear_function_get_results(linear_function_config):
    experiment = LinearFunctionExperiment(linear_function_config)
    results = experiment.get_results()
    assert 'x' in results
    assert 'y' in results
    assert results['x']['descriptor'].name == 'x'
    assert results['y']['descriptor'].name == 'y'

    assert isinstance(results['x']['data'], np.ndarray)
    assert isinstance(results['y']['data'], np.ndarray)
    assert len(results['x']['data']) == 5
    assert len(results['y']['data']) == 5

    # Check a few values
    assert np.isclose(results['x']['data'][0], 0.0)
    assert np.isclose(results['x']['data'][4], 4.0)
    assert np.isclose(results['y']['data'][0], 1.0)  # y = 2*0 + 1
    assert np.isclose(results['y']['data'][4], 9.0)  # y = 2*4 + 1