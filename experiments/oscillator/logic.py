# experiments/oscillator/logic.py
"""Damped (optionally Duffing) oscillator with physical units.

    m x'' + c x' + k x + beta x^3 = 0,     x(0) = x0,  x'(0) = v0

Parameters (SI units; any compatible units can be given in the config, e.g.
``stiffness: "2 kN/m"``): mass [kg], damping c [N*s/m], stiffness k [N/m],
nonlinearity beta [N/m^3], x0 [m], v0 [m/s], dt [s]; n_steps steps.

Integrators:

* ``exact``  - closed-form solution (beta = 0; under-, critically and over-damped)
* ``dop853`` - SciPy DOP853 adaptive solver, rtol 1e-10 / atol 1e-12
* ``euler``  - explicit Euler; a deliberately weak model (``audit``): it injects
  energy into an undamped oscillator, so amplitudes grow over time
* ``auto``   - ``exact`` if beta = 0, else ``dop853`` (default)

Outputs: time (coordinate), position and velocity (latent), mechanical energy
(label: E = m v^2/2 + k x^2/2 + beta x^4/4).
"""
import numpy as np

from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType

INTEGRATORS = ("auto", "exact", "dop853", "euler")


class OscillatorExperiment(ExperimentLogic):
    parameter_units = {
        "mass": "kg", "damping": "N*s/m", "stiffness": "N/m", "nonlinearity": "N/m**3",
        "x0": "m", "v0": "m/s", "dt": "s",
    }
    inferable_parameters = ("stiffness", "damping")

    def __init__(self, config):
        self.n_steps = int(config["n_steps"])
        self.mass = float(config.get("mass", 1.0))
        self.damping = float(config.get("damping", 0.0))
        self.stiffness = float(config["stiffness"])
        self.nonlinearity = float(config.get("nonlinearity", 0.0))
        self.x0 = float(config.get("x0", 1.0))
        self.v0 = float(config.get("v0", 0.0))
        self.dt = float(config.get("dt", 0.01))
        self.integrator = config.get("integrator", "auto")
        if self.mass <= 0 or self.stiffness <= 0 or self.dt <= 0:
            raise ValueError("mass, stiffness and dt must be > 0")
        if self.damping < 0:
            raise ValueError("damping must be >= 0")
        if self.integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}")
        if self.integrator == "exact" and self.nonlinearity != 0.0:
            raise ValueError("integrator 'exact' needs nonlinearity = 0 (no closed form for the Duffing term)")
        self.method = ("exact" if self.nonlinearity == 0.0 else "dop853") if self.integrator == "auto" else self.integrator

    # ------------------------------------------------------------------ declared properties
    def fidelity(self, config):
        if self.method == "exact":
            return {"level": "reference", "note": "Closed-form solution of the linear damped oscillator; "
                                                  "checked against DOP853 and energy balance in tests."}
        if self.method == "dop853":
            return {"level": "reference", "note": "SciPy DOP853, rtol 1e-10, atol 1e-12; agrees with the closed "
                                                  "form for beta = 0 and conserves energy without damping (tests)."}
        return {"level": "audit", "note": (f"Explicit Euler with dt = {self.dt:g} s: not energy-conserving; "
                                           "an undamped oscillator gains energy every step (amplitude grows). "
                                           "Kept as a deliberately weak model to test whether methods notice "
                                           "numerical model error.")}

    def invariants(self, results):
        """Mechanical energy is conserved without damping."""
        return {"energy": results["energy"]["data"]} if self.damping == 0.0 else {}

    def mechanism_graph(self, config):
        edges = [
            {"source": "velocity", "target": "position", "sign": "+", "parameter": None, "value": 1.0,
             "term": "kinematics dx/dt = v"},
            {"source": "position", "target": "velocity", "sign": "-", "parameter": "stiffness",
             "value": self.stiffness, "term": "linear restoring force -k x / m"},
            {"source": "velocity", "target": "velocity", "sign": "-", "parameter": "damping",
             "value": self.damping, "term": "viscous damping -c v / m"},
        ]
        if self.nonlinearity != 0.0:
            edges.append({"source": "position", "target": "velocity",
                          "sign": "-" if self.nonlinearity > 0 else "+", "parameter": "nonlinearity",
                          "value": self.nonlinearity, "term": "cubic (Duffing) force -beta x^3 / m"})
        return {"directed": True, "nodes": [{"id": "position", "kind": "state"}, {"id": "velocity", "kind": "state"}],
                "edges": edges}

    # ------------------------------------------------------------------ solutions
    def exact_solution(self, t):
        m, c, k, x0, v0 = self.mass, self.damping, self.stiffness, self.x0, self.v0
        w0 = np.sqrt(k / m)
        zeta = c / (2.0 * np.sqrt(k * m))
        if abs(zeta - 1.0) < 1e-9:  # critically damped
            a = v0 + w0 * x0
            e = np.exp(-w0 * t)
            return (x0 + a * t) * e, (a - w0 * (x0 + a * t)) * e
        if zeta < 1.0:  # under-damped (incl. undamped)
            wd = w0 * np.sqrt(1.0 - zeta ** 2)
            kk = (v0 + zeta * w0 * x0) / wd
            e = np.exp(-zeta * w0 * t)
            cos, sin = np.cos(wd * t), np.sin(wd * t)
            x = e * (x0 * cos + kk * sin)
            v = e * (-zeta * w0 * (x0 * cos + kk * sin) + (-x0 * wd * sin + kk * wd * cos))
            return x, v
        s = np.sqrt(zeta ** 2 - 1.0)  # over-damped
        r1, r2 = -w0 * (zeta - s), -w0 * (zeta + s)
        a = (v0 - r2 * x0) / (r1 - r2)
        b = x0 - a
        return a * np.exp(r1 * t) + b * np.exp(r2 * t), a * r1 * np.exp(r1 * t) + b * r2 * np.exp(r2 * t)

    def _rhs(self, _t, z):
        x, v = z
        return [v, -(self.damping * v + self.stiffness * x + self.nonlinearity * x ** 3) / self.mass]

    def numerical_solution(self, t):
        from scipy.integrate import solve_ivp
        sol = solve_ivp(self._rhs, (t[0], t[-1]), [self.x0, self.v0], t_eval=t, method="DOP853",
                        rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError(f"DOP853 failed: {sol.message}")
        return sol.y[0], sol.y[1]

    def euler_solution(self, t):
        x, v = np.empty_like(t), np.empty_like(t)
        x[0], v[0] = self.x0, self.v0
        for i in range(1, len(t)):
            dx, dv = self._rhs(t[i - 1], (x[i - 1], v[i - 1]))
            x[i], v[i] = x[i - 1] + self.dt * dx, v[i - 1] + self.dt * dv
        return x, v

    # ------------------------------------------------------------------ ExperimentLogic
    def initialize(self, config):
        return {}

    def run_step(self, state, step):
        return state  # trajectories are computed on the full time grid in get_results

    def get_results(self):
        t = np.arange(self.n_steps + 1) * self.dt
        x, v = {"exact": self.exact_solution, "dop853": self.numerical_solution,
                "euler": self.euler_solution}[self.method](t)
        energy = 0.5 * self.mass * v ** 2 + 0.5 * self.stiffness * x ** 2 + 0.25 * self.nonlinearity * x ** 4
        return {
            "time": {"data": t, "descriptor": DataDescriptor("time", DataType.NDARRAY, units="s",
                                                             group="time_series", role="coordinate")},
            "position": {"data": x, "descriptor": DataDescriptor("position", DataType.NDARRAY, units="m",
                                                                 group="time_series", plot_type="line",
                                                                 x_axis="time", role="latent")},
            "velocity": {"data": v, "descriptor": DataDescriptor("velocity", DataType.NDARRAY, units="m/s",
                                                                 group="velocity", x_axis="time", role="latent")},
            "energy": {"data": energy, "descriptor": DataDescriptor("energy", DataType.NDARRAY, units="J",
                                                                    group="energy", x_axis="time", role="label")},
        }
