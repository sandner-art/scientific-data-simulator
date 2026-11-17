# tests/test_mesa_adapter.py - the Mesa adapter and the agent-based pack (planted effects)
import json
import math
import os

import numpy as np
import pytest

mesa = pytest.importorskip("mesa")

from scidatasim.adapters.mesa import MesaExperiment, construct  # noqa: E402
from scidatasim.engine import SimulatorEngine, simulate  # noqa: E402
from scidatasim.experiments.social.logic import gamma_shape, saving_from_shape  # noqa: E402

PACK = "scidatasim.experiments.social.logic."

USER_MODEL = '''
import mesa
from scidatasim.adapters.mesa import init_mesa_model


class Walker(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.x = 0

    def step(self):
        self.x += self.model.random.choice((-1, 1))


class WalkModel(mesa.Model):
    def __init__(self, n=50, limit=None, rng=None):
        init_mesa_model(self, rng)
        self.limit = limit
        for _ in range(n):
            Walker(self)

    def step(self):
        self.agents.shuffle_do("step")
        if self.limit is not None and self.spread >= self.limit:
            self.running = False

    @property
    def spread(self):
        return float(sum(abs(a.x) for a in self.agents) / len(self.agents))


def mean_position(model):
    return float(sum(a.x for a in model.agents) / len(model.agents))


def positions(model):
    return [a.x for a in model.agents]


class NoSeed(mesa.Model):
    def __init__(self, n=3):
        super().__init__()
'''


@pytest.fixture
def user_models(tmp_path, monkeypatch):
    (tmp_path / "usermodels.py").write_text(USER_MODEL)
    monkeypatch.syspath_prepend(str(tmp_path))
    return "usermodels"


def _cfg(**extra):
    return {"experiment_type": "scidatasim.adapters.mesa.MesaExperiment", "model": "usermodels.WalkModel",
            "model_params": {"n": 50}, "n_steps": 20, "seed": 3, "generate_plots": False,
            "reporters": {"spread": "spread", "mean": "usermodels:mean_position"}, **extra}


def test_any_mesa_model_runs_reproducibly_with_roles(user_models):
    a, b = simulate(_cfg()), simulate(_cfg())
    assert np.array_equal(a.results["spread"]["data"], b.results["spread"]["data"])
    assert a.results["spread"]["data"].shape == (21,)  # step 0 and after each of 20 steps
    assert a.results["step"]["descriptor"].role.value == "coordinate"
    assert a.results["spread"]["descriptor"].role.value == "latent"
    c = simulate(_cfg(seed=4))
    assert not np.array_equal(a.results["spread"]["data"], c.results["spread"]["data"])
    params = a.ground_truth["parameters"]
    assert params["n"] == 50 and params["mesa_model"] == "usermodels.WalkModel" and params["mesa_version"]
    assert a.logic.fidelity({})["level"] == "unverified"


def test_snapshots_early_stop_and_observation(user_models):
    sim = simulate(_cfg(model_params={"n": 50, "limit": 2.0}, n_steps=200,
                        snapshots={"positions": "usermodels:positions"},
                        observation={"targets": ["spread"], "operators": [{"type": "gaussian", "sigma": 0.1}]}))
    steps = sim.results["step"]["data"]
    assert len(steps) < 201 and sim.results["spread"]["data"][-1] >= 2.0  # stopped when running turned False
    assert sim.results["positions"]["data"].shape == (50,)
    assert sim.results["spread_observed"]["descriptor"].role.value == "observation"


def test_adapter_errors(user_models):
    with pytest.raises(ValueError):
        simulate(_cfg(reporters={}))
    with pytest.raises(TypeError):
        simulate(_cfg(reporters={"all": "usermodels:positions"}))  # not a number
    import usermodels
    with pytest.raises(TypeError):
        construct(usermodels.NoSeed, {}, 1)
    with pytest.raises(ValueError):
        MesaExperiment({"reporters": {"x": "x"}})  # no model


# ------------------------------------------------------------------ the pack
def _pack(name, **params):
    return simulate({"experiment_type": PACK + name, "seed": params.pop("seed", 1), "generate_plots": False, **params})


def test_segregation_conserves_agents_and_responds_to_homophily():
    sim = _pack("SegregationExperiment", n_steps=30, homophily=0.5)
    grid = sim.results["grid"]["data"]
    model = sim.logic.model
    assert (grid >= 0).sum() == len(model.agents) == round(0.85 * 900)
    assert len({a.pos_rc for a in model.agents}) == len(model.agents)  # one resident per cell
    assert all(grid[a.pos_rc] == a.group for a in model.agents)
    calm = _pack("SegregationExperiment", n_steps=10, homophily=0.0)
    assert calm.logic.model.moves == 0 and np.ptp(calm.results["segregation"]["data"]) == 0
    low = np.mean([_pack("SegregationExperiment", n_steps=30, homophily=0.3, seed=s).results["segregation"]["data"][-1]
                   for s in range(3)])
    high = np.mean([_pack("SegregationExperiment", n_steps=30, homophily=0.6, seed=s).results["segregation"]["data"][-1]
                    for s in range(3)])
    assert high > low + 0.1
    with pytest.raises(ValueError):
        _pack("SegregationExperiment", n_steps=1, tolerance=0.3)  # not a parameter of the model


def _hazard_effect(results):
    """1 - (infections per susceptible-step, treated) / (the same, control)."""
    def rate(new, sus):
        return np.sum(results[new]["data"][1:]) / np.sum(results[sus]["data"][:-1])
    return 1 - rate("new_treated", "susceptible_treated") / rate("new_control", "susceptible_control")


def test_epidemic_trial_conserves_people_and_plants_the_effect():
    sim = _pack("EpidemicTrialExperiment", n_steps=150, effect=0.5)
    r = sim.results
    total = r["susceptible"]["data"] + r["infectious"]["data"] + r["recovered"]["data"]
    assert np.all(total == 400)
    assert np.all(r["susceptible_treated"]["data"] + r["susceptible_control"]["data"] == r["susceptible"]["data"])
    shield = _pack("EpidemicTrialExperiment", n_steps=150, effect=1.0, initial_infected=5)
    assert shield.results["new_treated"]["data"].sum() == 0
    estimates = {e: np.mean([_hazard_effect(_pack("EpidemicTrialExperiment", n_steps=150, effect=e, seed=s).results)
                             for s in range(8)]) for e in (0.0, 0.5)}
    assert abs(estimates[0.0]) < 0.12 and abs(estimates[0.5] - 0.5) < 0.12


def test_wealth_is_conserved_and_matches_the_gamma_law():
    from scipy.special import gammaln
    for saving, steps in ((0.0, 300), (0.5, 400)):
        sim = _pack("WealthExchangeExperiment", n_steps=steps, saving=saving, n_agents=2000)
        assert np.allclose(sim.results["total_wealth"]["data"], 2000.0)
        k = gamma_shape(saving)
        theory = math.exp(gammaln(k + 0.5) - gammaln(k) - math.log(k) - 0.5 * math.log(math.pi))  # Gini of Gamma(k)
        assert sim.results["gini"]["data"][-1] == pytest.approx(theory, abs=0.02)
    assert saving_from_shape(gamma_shape(0.37)) == pytest.approx(0.37)


def test_pack_bundles_ask_for_the_planted_effect(tmp_path):
    cfg = {"experiment_type": PACK + "EpidemicTrialExperiment", "seed": 2, "n_steps": 120, "effect": 0.3,
           "generate_plots": False, "bundle": {"qa": {"exclude": ["argmax", "value_at", "compare_max"]}},
           "observation": {"targets": ["new_treated", "new_control"], "operators": [{"type": "poisson", "scale": 1.0}]}}
    run = SimulatorEngine(str(tmp_path), verbose=False).run_config(cfg)
    bundle = os.path.join(run.experiment_dir, "bundle")
    qa = json.load(open(os.path.join(bundle, "qa.json")))
    planted = [q for q in qa if q["template"] == "planted_effect"]
    assert planted and planted[0]["answer"] == 0.3 and planted[0]["tolerance"] == 0.1
    public = json.load(open(os.path.join(bundle, "questions.json")))
    assert all("answer" not in q for q in public)
    assert run.record.provenance["fidelity"]["level"] == "reference"
