# experiments/predator_prey/logic.py
"""Lotka-Volterra predator-prey model.

    dx/dt = a x (1 - x/K) - b x y / (1 + b h x)
    dy/dt = c x y / (1 + b h x) - d y

x = prey, y = predators; a = prey_growth_rate, b = prey_death_rate,
c = predator_growth_rate, d = predator_death_rate. Optional:
K = prey_carrying_capacity (logistic prey; default: none),
h = handling_time with functional_response: holling2 (default: linear, h = 0).

Defaults reproduce the 0.1.0 behaviour exactly: forward Euler, dt = 1 and
negative populations clamped to 0. That is a *teaching / audit* model whose
trajectories carry integrator error (it does not conserve the Lotka-Volterra
invariant). Use ``integrator: rk4`` with a small ``dt`` for reference-quality
trajectories; tests/test_verification.py checks both behaviours.
"""
from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType
import numpy as np

INTEGRATORS = ("euler", "rk4")
FUNCTIONAL_RESPONSES = ("linear", "holling2")


class PredatorPreyExperiment(ExperimentLogic):
    inferable_parameters = ("prey_growth_rate", "prey_death_rate", "predator_growth_rate", "predator_death_rate")
    # Model time has no physical scale ("time_unit"); rates are per time unit.
    parameter_units = {
        "prey_growth_rate": "1/time_unit",
        "prey_death_rate": "1/(individual*time_unit)",
        "predator_growth_rate": "1/(individual*time_unit)",
        "predator_death_rate": "1/time_unit",
        "initial_prey": "individual",
        "initial_predators": "individual",
        "prey_carrying_capacity": "individual",
        "handling_time": "time_unit",
        "dt": "time_unit",
    }

    def __init__(self, config):
        self.n_steps = config['n_steps']
        self.initial_prey = config['initial_prey']
        self.initial_predators = config['initial_predators']
        self.prey_growth_rate = config['prey_growth_rate']
        self.prey_death_rate = config['prey_death_rate']
        self.predator_growth_rate = config['predator_growth_rate']
        self.predator_death_rate = config['predator_death_rate']

        self.dt = float(config.get('dt', 1.0))
        self.integrator = config.get('integrator', 'euler')
        self.clamp_negative = bool(config.get('clamp_negative', True))
        self.functional_response = config.get('functional_response', 'linear')
        self.handling_time = float(config.get('handling_time', 0.0))
        self.carrying_capacity = config.get('prey_carrying_capacity')
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}")
        if self.functional_response not in FUNCTIONAL_RESPONSES:
            raise ValueError(f"functional_response must be one of {FUNCTIONAL_RESPONSES}")
        if self.functional_response == 'linear' and self.handling_time != 0.0:
            raise ValueError("handling_time requires functional_response: holling2")
        if self.carrying_capacity is not None and self.carrying_capacity <= 0:
            raise ValueError("prey_carrying_capacity must be > 0")

        # Stored trajectories (protected, for use in subclasses)
        self._prey_populations = [self.initial_prey]
        self._predator_populations = [self.initial_predators]
        self._time_points = [0]

    # ------------------------------------------------------------------ declared properties
    def fidelity(self, config):
        if self.integrator == 'euler':
            return {"level": "audit",
                    "note": (f"Forward Euler with dt = {self.dt:g}"
                             + (" and clamping of negative populations" if self.clamp_negative else "")
                             + ": first-order integrator error; violates the Lotka-Volterra invariant "
                               "(by > 1 % at dt = 1). Kept as a deliberately weak model (0.1.0 behaviour) "
                               "to test whether methods notice model error. Use integrator: rk4 for reference data.")}
        note = (f"RK4 with dt = {self.dt:g}; agrees with a high-accuracy solver, converges at order 4 and "
                "conserves the Lotka-Volterra invariant (verified for dt <= 0.04).")
        if self.dt > 0.04:
            return {"level": "unverified", "note": note + " This dt is larger than the verified range."}
        if self.clamp_negative:
            note += " Clamping of negative values is enabled (inactive for positive trajectories)."
        return {"level": "reference", "note": note}

    def invariants(self, results):
        """Conserved quantity of the classical system (linear response, no carrying capacity):
        V = c x - d ln x + b y - a ln y."""
        if self.functional_response != 'linear' or self.carrying_capacity is not None:
            return {}
        x = np.asarray(results["prey_population"]["data"], dtype=float)
        y = np.asarray(results["predator_population"]["data"], dtype=float)
        if np.any(x <= 0) or np.any(y <= 0):
            return {}
        a, b = self.prey_growth_rate, self.prey_death_rate
        c, d = self.predator_growth_rate, self.predator_death_rate
        return {"lotka_volterra_V": c * x - d * np.log(x) + b * y - a * np.log(y)}

    def mechanism_graph(self, config):
        prey, pred = "prey_population", "predator_population"
        growth = "logistic growth" if self.carrying_capacity is not None else "exponential growth"
        response = "saturating (Holling II)" if self.functional_response == 'holling2' else "mass action"
        edges = [
            {"source": prey, "target": prey, "sign": "+", "parameter": "prey_growth_rate",
             "value": self.prey_growth_rate, "term": growth},
            {"source": pred, "target": prey, "sign": "-", "parameter": "prey_death_rate",
             "value": self.prey_death_rate, "term": f"predation, {response}"},
            {"source": prey, "target": pred, "sign": "+", "parameter": "predator_growth_rate",
             "value": self.predator_growth_rate, "term": f"conversion, {response}"},
            {"source": pred, "target": pred, "sign": "-", "parameter": "predator_death_rate",
             "value": self.predator_death_rate, "term": "mortality"},
        ]
        if self.carrying_capacity is not None:
            edges.append({"source": prey, "target": prey, "sign": "-", "parameter": "prey_carrying_capacity",
                          "value": self.carrying_capacity, "term": "self-limitation"})
        if self.functional_response == 'holling2':
            edges.append({"source": prey, "target": pred, "sign": "-", "parameter": "handling_time",
                          "value": self.handling_time, "term": "saturation of the functional response"})
        return {"directed": True,
                "nodes": [{"id": prey, "kind": "state"}, {"id": pred, "kind": "state"}],
                "edges": edges}

    # ------------------------------------------------------------------ model
    def derivatives(self, prey, predators):
        """Right-hand side (dx/dt, dy/dt)."""
        a, b = self.prey_growth_rate, self.prey_death_rate
        c, d = self.predator_growth_rate, self.predator_death_rate
        saturation = 1.0 + b * self.handling_time * prey if self.functional_response == 'holling2' else 1.0
        growth = a * prey if self.carrying_capacity is None else a * prey * (1.0 - prey / self.carrying_capacity)
        d_prey = growth - b * prey * predators / saturation
        d_predators = c * prey * predators / saturation - d * predators
        return d_prey, d_predators

    def _advance(self, prey, predators):
        h = self.dt
        if self.integrator == 'euler':
            dx, dy = self.derivatives(prey, predators)
            return prey + h * dx, predators + h * dy
        k1 = self.derivatives(prey, predators)
        k2 = self.derivatives(prey + 0.5 * h * k1[0], predators + 0.5 * h * k1[1])
        k3 = self.derivatives(prey + 0.5 * h * k2[0], predators + 0.5 * h * k2[1])
        k4 = self.derivatives(prey + h * k3[0], predators + h * k3[1])
        return (prey + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                predators + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))

    # ------------------------------------------------------------------ ExperimentLogic
    def initialize(self, config):
        self._prey_populations = [self.initial_prey]
        self._predator_populations = [self.initial_predators]
        self._time_points = [0]
        return {
            'prey': self.initial_prey,
            'predators': self.initial_predators,
        }

    def run_step(self, state, step):
        new_prey, new_predators = self._advance(state['prey'], state['predators'])
        if self.clamp_negative:
            new_prey, new_predators = max(0, new_prey), max(0, new_predators)

        self._prey_populations.append(new_prey)
        self._predator_populations.append(new_predators)
        self._time_points.append((step + 1) * self.dt if self.dt != 1.0 else step + 1)

        return {
            'prey': new_prey,
            'predators': new_predators,
        }

    def get_results(self):
        n = len(self._time_points)
        time_units = "time_unit"
        return {
            "time": {
                "data": np.array(self._time_points),
                "descriptor": DataDescriptor("time", DataType.NDARRAY, shape=(n,), units=time_units, group="time_series", role="coordinate")
            },
            "prey_population": {
                "data": np.array(self._prey_populations, dtype=float),
                "descriptor": DataDescriptor("prey_population", DataType.NDARRAY, shape=(n,), units="individuals", group="time_series", plot_type="line", x_axis="time", role="latent")
            },
            "predator_population": {
                "data": np.array(self._predator_populations, dtype=float),
                "descriptor": DataDescriptor("predator_population", DataType.NDARRAY, shape=(n,), units="individuals", group="time_series", plot_type="line", x_axis="time", role="latent")
            }
        }
