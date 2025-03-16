# tests/test_random_walk.py
import pytest
from experiments.example_random_walk.logic import RandomWalkExperiment
import numpy as np

@pytest.fixture
def random_walk_config():
    return {
        'n_steps': 10,
        'step_size': 2.0
    }

def test_random_walk_initialize(random_walk_config):
    experiment = RandomWalkExperiment(random_walk_config)
    initial_state = experiment.initialize(random_walk_config)
    assert initial_state == {'step': 0, 'position': 0}

def test_random_walk_run_step(random_walk_config):
    experiment = RandomWalkExperiment(random_walk_config)
    state = {'step': 0, 'position': 0}

    # We can't predict the *exact* position due to randomness,
    # but we can check the step count and that the position
    # changes by +/- step_size.
    new_state = experiment.run_step(state, 0)
    assert new_state['step'] == 1
    assert abs(new_state['position']) == 2.0  # It moved either +2 or -2

    new_state = experiment.run_step(new_state, 1)
    assert new_state['step'] == 2
    assert new_state['position'] in [-4.0, 0.0, 4.0] # Possible positions

def test_random_walk_get_results(random_walk_config, monkeypatch):
    # We will fix the random seed, to have determined results
    def mock_random_choice(*args, **kwargs):
        return [1, -1, 1, -1, 1, 1, -1, 1, 1, -1]

    monkeypatch.setattr(np.random, 'choice', mock_random_choice)

    experiment = RandomWalkExperiment(random_walk_config)
    # Set a seed and run experiment.
    results = experiment.get_results()

    assert 'step' in results
    assert 'position' in results
    assert results['step']['descriptor'].name == 'step'
    assert results['position']['descriptor'].name == 'position'
    assert isinstance(results['step']['data'], np.ndarray)
    assert isinstance(results['position']['data'], np.ndarray)
    assert len(results['step']['data']) == 11  # n_steps + 1
    assert len(results['position']['data']) == 11
    # Check a few values, with fixed seed we can check exact values
    assert np.array_equal(results['position']['data'], [0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 6.0, 4.0])