# scidatasim/experiments/social/logic.py
"""Experiments for the agent-based pack (social science, epidemiology, economics), run with Mesa.

Each wraps a model of :mod:`scidatasim.experiments.social.models` through
:class:`scidatasim.adapters.mesa.MesaExperiment`. The planted effect is a parameter of the config
(``homophily``, ``effect``, ``saving``); designs vary it, bundles ask for it back, and simulation
studies (:mod:`scidatasim.ademp`) compare estimators of it. Needs ``pip install
"scientific-data-simulator[mesa]"``.

    experiment_type: scidatasim.experiments.social.logic.EpidemicTrialExperiment
    n_steps: 120
    effect: 0.4          # planted: relative reduction of infection risk per contact when treated
    beta: 0.04
    observation: {operators: [{type: poisson, scale: 1.0}]}   # reported counts
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from scidatasim.adapters.mesa import MesaExperiment

_EXCLUDE = {"experiment_type", "experiment_description", "n_steps", "seed", "generate_plots",
            "static_plot_format", "save_csv", "observation", "law_shift", "storage", "bundle", "unit_check",
            "reporters", "snapshots", "stop_when_done", "model", "model_params"}


class _Pack(MesaExperiment):
    fidelity_level = "reference"
    params: Dict[str, Any] = {}

    def _model_params(self, config):
        unknown = set(config) - set(self.params) - _EXCLUDE
        if unknown:
            raise ValueError(f"{type(self).__name__}: unknown parameters {sorted(unknown)}; "
                             f"it takes {sorted(self.params)}")
        return {k: config.get(k, default) for k, default in self.params.items()}

    def _observed(self, results, name):
        """The observed series if the run has one, else the exact one (what a solver sees)."""
        for key, info in results.items():
            if key.startswith(name) and key != name and info["descriptor"].role.value == "observation":
                return np.asarray(info["data"], float)
        return np.asarray(results[name]["data"], float)


class SegregationExperiment(_Pack):
    """Schelling segregation. Planted: ``homophily``, the share of similar neighbours a resident wants."""
    from .models import SegregationModel as model_class
    params = {"width": 30, "height": 30, "density": 0.85, "minority": 0.35, "homophily": 0.5}
    reporters = {"segregation": "segregation", "unhappy": "unhappy"}
    snapshots = {"grid": "grid_array"}
    units = {"unhappy": "count"}
    fidelity_note = ("Exact discrete rules; tests check that agents and groups are conserved, that no cell holds "
                     "two residents, and that nobody moves with homophily 0.")

    def questions(self, results, rng) -> List[Dict[str, Any]]:
        h = float(self.model_params["homophily"])
        band = "low (below 0.35)" if h < 0.35 else ("high (above 0.55)" if h > 0.55 else "medium (0.35 to 0.55)")
        final = float(results["segregation"]["data"][-1])
        return [
            dict(template="planted_effect", question="Residents move when too few of their neighbours share their "
                 "group. How demanding is that preference: what share of similar neighbours do they want?",
                 answer_type="choice", choices=["low (below 0.35)", "medium (0.35 to 0.55)", "high (above 0.55)"],
                 answer=band, tolerance=None, tolerance_rule="exact", answerable_from="inference"),
            dict(template="final_segregation", question="What is the mean share of similar neighbours at the end?",
                 answer_type="number", answer=final, tolerance=0.02, tolerance_rule="0.02 absolute",
                 series="segregation", answerable_from="data"),
        ]


class EpidemicTrialExperiment(_Pack):
    """An epidemic with a randomised trial inside it. Planted: ``effect``, the relative reduction in
    the chance of infection per contact for treated agents."""
    from .models import EpidemicTrialModel as model_class
    params = {"n_agents": 400, "width": 20, "height": 20, "beta": 0.04, "gamma": 0.1, "initial_infected": 5,
              "treated_share": 0.5, "effect": 0.4}
    reporters = {"susceptible": "susceptible", "infectious": "infectious", "recovered": "recovered",
                 "new_treated": "new_treated", "new_control": "new_control",
                 "susceptible_treated": "susceptible_treated", "susceptible_control": "susceptible_control"}
    units = {k: "count" for k in reporters}
    fidelity_note = ("Exact discrete rules; tests check that S + I + R and the arms are conserved, that arms "
                     "match with effect 0, and that nobody treated is infected with effect 1.")

    def questions(self, results, rng) -> List[Dict[str, Any]]:
        peak = float(np.max(results["infectious"]["data"]))
        return [
            dict(template="planted_effect", question="Treated agents were chosen at random. By what fraction does "
                 "treatment lower the chance of infection per contact (0 = no effect, 1 = full protection)?",
                 answer_type="number", answer=float(self.model_params["effect"]), tolerance=0.1,
                 tolerance_rule="0.1 absolute", answerable_from="inference"),
            dict(template="peak_infectious", question="What is the largest number of infectious agents at one time?",
                 answer_type="number", answer=peak, tolerance=max(2.0, 0.05 * peak),
                 tolerance_rule="5 %, at least 2 agents", series="infectious", answerable_from="data"),
        ]


class WealthExchangeExperiment(_Pack):
    """Kinetic wealth exchange. Planted: ``saving``, the share of wealth each agent keeps in a trade."""
    from .models import WealthExchangeModel as model_class
    params = {"n_agents": 500, "saving": 0.3}
    reporters = {"gini": "gini", "top10_share": "top10_share", "total_wealth": "total_wealth"}
    snapshots = {"wealth": "wealth"}
    units = {"total_wealth": "dimensionless", "wealth": "dimensionless"}  # in units of the initial wealth per agent
    fidelity_note = ("Total wealth is conserved exactly (checked); with saving 0 the stationary distribution is "
                     "exponential (Gini 0.5), and for saving s close to a Gamma law with shape 1 + 3s / (1 - s), "
                     "both checked in tests.")

    def invariants(self, results):
        return {"total_wealth": results["total_wealth"]["data"]}

    def questions(self, results, rng) -> List[Dict[str, Any]]:
        return [
            dict(template="planted_effect", question="In each trade both agents keep a fixed share of their wealth "
                 "and split the rest at random. Estimate that share from the final wealth distribution.",
                 answer_type="number", answer=float(self.model_params["saving"]), tolerance=0.1,
                 tolerance_rule="0.1 absolute", answerable_from="inference"),
            dict(template="final_gini", question="What is the Gini coefficient of wealth at the end?",
                 answer_type="number", answer=float(results["gini"]["data"][-1]), tolerance=0.02,
                 tolerance_rule="0.02 absolute", series="gini", answerable_from="data"),
        ]


def gamma_shape(saving: float) -> float:
    """Shape of the Gamma law that approximates the stationary wealth distribution (Patriarca,
    Chakraborti and Kaski 2004): n = 1 + 3 s / (1 - s)."""
    return 1.0 + 3.0 * saving / (1.0 - saving)


def saving_from_shape(shape: float) -> float:
    """Inverse of :func:`gamma_shape`: s = (n - 1) / (n + 2)."""
    return (shape - 1.0) / (shape + 2.0)
