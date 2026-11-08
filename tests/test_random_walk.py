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

class _ScriptedRng:
    """Stand-in for numpy.random.Generator returning a fixed sequence of draws."""

    def __init__(self, draws):
        self._draws = iter(draws)

    def choice(self, options):
        return next(self._draws)


def test_random_walk_get_results(random_walk_config):
    experiment = RandomWalkExperiment(random_walk_config)
    experiment.rng = _ScriptedRng([1, -1, 1, -1, 1, 1, -1, 1, 1, -1])
    state = experiment.initialize(random_walk_config)
    for step in range(random_walk_config['n_steps']):
        state = experiment.run_step(state, step)
    results = experiment.get_results()

    assert results['step']['descriptor'].name == 'step'
    assert results['position']['descriptor'].name == 'position'
    assert len(results['step']['data']) == 11  # n_steps + 1
    assert np.array_equal(results['position']['data'], [0, 2.0, 0.0, 2.0, 0.0, 2.0, 4.0, 2.0, 4.0, 6.0, 4.0])


def _walk(config):
    experiment = RandomWalkExperiment(config)
    state = experiment.initialize(config)
    for step in range(config['n_steps']):
        state = experiment.run_step(state, step)
    return experiment.get_results()['position']['data']


def test_random_walk_same_seed_same_path(random_walk_config):
    cfg = dict(random_walk_config, n_steps=200, seed=1234)
    assert np.array_equal(_walk(cfg), _walk(cfg))
    assert not np.array_equal(_walk(cfg), _walk(dict(cfg, seed=1235)))


def test_random_walk_does_not_touch_global_state(random_walk_config):
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    _walk(dict(random_walk_config, seed=1))
    assert np.random.random() == before


def test_random_walk_increment_statistics():
    """Increments are +/-a with p = 1/2: mean 0, variance a^2 (5-sigma bounds)."""
    n, a = 20000, 2.0
    increments = np.diff(_walk({'n_steps': n, 'step_size': a, 'seed': 7}))
    assert set(np.unique(increments)) <= {-a, a}
    assert abs(increments.mean()) < 5 * a / np.sqrt(n)
    assert increments.var() == pytest.approx(a ** 2, rel=0.01)
