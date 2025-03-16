# tests/test_predator_prey.py
import pytest
from experiments.predator_prey.logic import PredatorPreyExperiment
import numpy as np

@pytest.fixture
def predator_prey_config():
    return {
        'n_steps': 5,
        'initial_prey': 100,
        'initial_predators': 20,
        'prey_growth_rate': 0.1,
        'prey_death_rate': 0.02,
        'predator_growth_rate': 0.01,
        'predator_death_rate': 0.05
    }

def test_predator_prey_initialize(predator_prey_config):
    experiment = PredatorPreyExperiment(predator_prey_config)
    initial_state = experiment.initialize(predator_prey_config)
    assert initial_state == {'prey': 100, 'predators': 20}

def test_predator_prey_run_step(predator_prey_config):
    experiment = PredatorPreyExperiment(predator_prey_config)
    state = {'prey': 100, 'predators': 20}

    new_state = experiment.run_step(state, 0)
    # Check that the new state is calculated correctly based on Lotka-Volterra
    expected_new_prey = max(0, 100 + (0.1 * 100) - (0.02 * 100 * 20))
    expected_new_predators = max(0, 20 + (0.01 * 100 * 20) - (0.05 * 20))
    assert np.isclose(new_state['prey'], expected_new_prey)
    assert np.isclose(new_state['predators'], expected_new_predators)

    # Test a second step, using output of first
    new_state2 = experiment.run_step(new_state, 1)
    expected_new_prey2 = max(0, expected_new_prey + (0.1 * expected_new_prey) - (0.02 * expected_new_prey * expected_new_predators))
    expected_new_predators2 = max(0, expected_new_predators + (0.01 * expected_new_prey * expected_new_predators) - (0.05 * expected_new_predators))
    assert np.isclose(new_state2['prey'], expected_new_prey2)
    assert np.isclose(new_state2['predators'], expected_new_predators2)

def test_predator_prey_get_results(predator_prey_config):
    experiment = PredatorPreyExperiment(predator_prey_config)
    results = experiment.get_results()

    assert 'time' in results
    assert 'prey_population' in results
    assert 'predator_population' in results
    assert results['time']['descriptor'].name == 'time'
    assert results['prey_population']['descriptor'].name == 'prey_population'
    assert results['predator_population']['descriptor'].name == 'predator_population'

    assert isinstance(results['time']['data'], np.ndarray)
    assert isinstance(results['prey_population']['data'], np.ndarray)
    assert isinstance(results['predator_population']['data'], np.ndarray)

    assert len(results['time']['data']) == predator_prey_config['n_steps'] + 1
    assert len(results['prey_population']['data']) == predator_prey_config['n_steps'] + 1
    assert len(results['predator_population']['data']) == predator_prey_config['n_steps'] + 1

    # Check initial values (since we're storing initial values in the lists)
    assert results['time']['data'][0] == 0
    assert results['prey_population']['data'][0] == predator_prey_config['initial_prey']
    assert results['predator_population']['data'][0] == predator_prey_config['initial_predators']