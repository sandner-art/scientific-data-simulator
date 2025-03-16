# tests/test_example_experiment.py
import pytest
from experiments.example_experiment.logic import ExampleExperiment
import numpy as np

@pytest.fixture
def example_config():
    return {
        'n_steps': 5,
        'amplitude': 2.0
    }

def test_initialize(example_config):
    experiment = ExampleExperiment(example_config)
    initial_state = experiment.initialize(example_config)
    assert initial_state == {'time': 0, 'value': 0}

def test_run_step(example_config):
    experiment = ExampleExperiment(example_config)
    state = {'time': 0, 'value': 0}
    new_state = experiment.run_step(state, 0)  # step argument is not used in this example
    assert new_state['time'] == 1
    assert np.isclose(new_state['value'], 2.0 * np.sin(1))  # Check with a tolerance

    state = new_state
    new_state = experiment.run_step(state, 1)
    assert new_state['time'] == 2
    assert np.isclose(new_state['value'], 2.0 * np.sin(2))

def test_get_results(example_config):
    experiment = ExampleExperiment(example_config)
    # Normally, you would run the initialization and steps here
    # to populate the internal data structures of the experiment.
    # For this simple example, we know what the results should be,
    # so we can test get_results directly.
    results = experiment.get_results()

    assert 'time' in results
    assert 'value' in results

    assert results['time']['descriptor'].name == 'time'
    assert results['value']['descriptor'].name == 'value'

    assert isinstance(results['time']['data'], np.ndarray)
    assert isinstance(results['value']['data'], np.ndarray)

    assert len(results['time']['data']) == 5
    assert len(results['value']['data']) == 5

    # Check a few values
    assert np.isclose(results['value']['data'][0], 0.0)
    assert np.isclose(results['value']['data'][1], 2.0 * np.sin(1))
    assert np.isclose(results['value']['data'][2], 2.0 * np.sin(2))