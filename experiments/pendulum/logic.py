# experiments/pendulum/logic.py
"""Plane pendulum (point mass on a massless rod), optionally damped.

    theta'' + (c / m) theta' + (g / L) sin(theta) = 0,   theta(0) = theta0,  theta'(0) = omega0

Parameters (SI; any compatible units may be given): length L [m], gravity g [m/s^2],
mass m [kg], damping c [kg/s] (torque -c L^2 theta'), theta0 [rad] (or ``theta0_deg``),
omega0 [rad/s], dt [s]; n_steps steps.

``approximation``:

* ``none`` (default, ``reference``) - the nonlinear equation, integrated with SciPy DOP853
  (rtol 1e-11, atol 1e-12). Undamped, released from rest, the period is checked against the
  exact value T = 4 sqrt(L/g) K(sin^2(theta0/2)) (complete elliptic integral of the first kind).
* ``small_angle`` (``audit``) - sin(theta) replaced by theta: the exact solution of the
  *linearised* equation. The integration is exact, but the model is structurally wrong for
  large swings: its period 2 pi sqrt(L/g) does not depend on amplitude, and is about 7 % short at
  theta0 = 60 degrees and 15 % short at 90 degrees. Kept as a deliberately weak model to test
  whether methods notice model-form (not numerical) error.

Outputs: time (coordinate), angle and angular_velocity (latent), energy (label: per unit mass,
E/m = L^2 omega^2 / 2 + g L (1 - cos theta), the true pendulum energy for both approximations),
and ``period`` (label: the period the chosen model predicts; damping-free formula).
"""
import numpy as np

from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType

APPROXIMATIONS = ("none", "small_angle")


def exact_period(length, gravity, theta0):
    """Period of the undamped nonlinear pendulum released from rest at ``theta0`` (|theta0| < pi)."""
    from scipy.special import ellipk
    return 4.0 * np.sqrt(length / gravity) * ellipk(np.sin(abs(theta0) / 2.0) ** 2)


class PendulumExperiment(ExperimentLogic):
    parameter_units = {"length": "m", "gravity": "m/s**2", "mass": "kg", "damping": "kg/s",
                       "theta0": "rad", "omega0": "rad/s", "dt": "s"}
    inferable_parameters = ("length", "damping")

    def __init__(self, config):
        self.n_steps = int(config["n_steps"])
        self.length = float(config.get("length", 1.0))
        self.gravity = float(config.get("gravity", 9.81))
        self.mass = float(config.get("mass", 1.0))
        self.damping = float(config.get("damping", 0.0))
        if "theta0_deg" in config and "theta0" in config:
            raise ValueError("give theta0 or theta0_deg, not both")
        self.theta0 = (np.deg2rad(float(config["theta0_deg"])) if "theta0_deg" in config
                       else float(config.get("theta0", 0.5)))
        self.omega0 = float(config.get("omega0", 0.0))
        self.dt = float(config.get("dt", 0.01))
        self.approximation = config.get("approximation", "none")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.length <= 0 or self.gravity <= 0 or self.mass <= 0 or self.dt <= 0:
            raise ValueError("length, gravity, mass and dt must be > 0")
        if self.damping < 0:
            raise ValueError("damping must be >= 0")
        if self.approximation not in APPROXIMATIONS:
            raise ValueError(f"approximation must be one of {APPROXIMATIONS}")

    # ------------------------------------------------------------------ declared properties
    def fidelity(self, config):
        if self.approximation == "small_angle":
            deg = np.rad2deg(abs(self.theta0))
            err = ""
            if self.omega0 == 0.0 and abs(self.theta0) < np.pi:
                rel = 1.0 - 2.0 * np.pi * np.sqrt(self.length / self.gravity) / exact_period(
                    self.length, self.gravity, self.theta0)
                err = f" At theta0 = {deg:.1f} deg the predicted period is {100 * rel:.1f} % too short."
            return {"level": "audit", "note": (
                "Small-angle (linearised) pendulum, sin(theta) ~ theta: exact solution of the wrong equation. "
                "Its period does not depend on amplitude." + err +
                " Kept as a deliberately weak model to test detection of model-form error.")}
        return {"level": "reference", "note": (
            "Nonlinear pendulum, SciPy DOP853 (rtol 1e-11, atol 1e-12); the undamped period matches the "
            "elliptic-integral formula and energy is conserved without damping (tests).")}

    def invariants(self, results):
        if self.damping == 0.0 and self.approximation == "none":
            return {"energy": results["energy"]["data"]}
        return {}

    def mechanism_graph(self, config):
        restoring = "-(g/L) theta" if self.approximation == "small_angle" else "-(g/L) sin(theta)"
        return {"directed": True,
                "nodes": [{"id": "angle", "kind": "state"}, {"id": "angular_velocity", "kind": "state"}],
                "edges": [
                    {"source": "angular_velocity", "target": "angle", "sign": "+", "parameter": None,
                     "value": 1.0, "term": "kinematics dtheta/dt = omega"},
                    {"source": "angle", "target": "angular_velocity", "sign": "-", "parameter": "length",
                     "value": self.length, "term": f"gravity torque {restoring}"},
                    {"source": "angular_velocity", "target": "angular_velocity", "sign": "-",
                     "parameter": "damping", "value": self.damping, "term": "viscous damping -(c/m) omega"},
                ]}

    def period(self):
        if self.approximation == "small_angle" or (self.theta0 == 0.0 and self.omega0 == 0.0):
            return 2.0 * np.pi * np.sqrt(self.length / self.gravity)
        energy = 0.5 * self.length ** 2 * self.omega0 ** 2 + self.gravity * self.length * (1 - np.cos(self.theta0))
        cos_amp = 1.0 - energy / (self.gravity * self.length)
        if cos_amp <= -1.0:
            return float("nan")  # rotating, not swinging
        return float(exact_period(self.length, self.gravity, np.arccos(cos_amp)))

    # ------------------------------------------------------------------ solutions
    def _rhs(self, _t, z):
        theta, omega = z
        restoring = theta if self.approximation == "small_angle" else np.sin(theta)
        return [omega, -(self.damping / self.mass) * omega - (self.gravity / self.length) * restoring]

    def linear_solution(self, t):
        """Closed form of the linearised equation (under-, critically or over-damped)."""
        w0 = np.sqrt(self.gravity / self.length)
        gamma = self.damping / (2.0 * self.mass)
        x0, v0 = self.theta0, self.omega0
        if abs(gamma - w0) < 1e-12 * w0:
            a = v0 + gamma * x0
            e = np.exp(-gamma * t)
            return (x0 + a * t) * e, (a - gamma * (x0 + a * t)) * e
        if gamma < w0:
            wd = np.sqrt(w0 ** 2 - gamma ** 2)
            b = (v0 + gamma * x0) / wd
            e = np.exp(-gamma * t)
            c, s = np.cos(wd * t), np.sin(wd * t)
            x = e * (x0 * c + b * s)
            return x, -gamma * x + e * (-x0 * wd * s + b * wd * c)
        s = np.sqrt(gamma ** 2 - w0 ** 2)
        r1, r2 = -gamma + s, -gamma - s
        a = (v0 - r2 * x0) / (r1 - r2)
        b = x0 - a
        return a * np.exp(r1 * t) + b * np.exp(r2 * t), a * r1 * np.exp(r1 * t) + b * r2 * np.exp(r2 * t)

    def numerical_solution(self, t):
        from scipy.integrate import solve_ivp
        sol = solve_ivp(self._rhs, (t[0], t[-1]), [self.theta0, self.omega0], t_eval=t, method="DOP853",
                        rtol=1e-11, atol=1e-12)
        if not sol.success:
            raise RuntimeError(f"DOP853 failed: {sol.message}")
        return sol.y[0], sol.y[1]

    # ------------------------------------------------------------------ ExperimentLogic
    def initialize(self, config):
        return {}

    def run_step(self, state, step):
        return state  # computed on the full grid in get_results

    def get_results(self):
        t = np.arange(self.n_steps + 1) * self.dt
        theta, omega = (self.linear_solution(t) if self.approximation == "small_angle"
                        else self.numerical_solution(t))
        energy = 0.5 * self.length ** 2 * omega ** 2 + self.gravity * self.length * (1.0 - np.cos(theta))
        return {
            "time": {"data": t, "descriptor": DataDescriptor("time", DataType.NDARRAY, units="s",
                                                             group="time_series", role="coordinate")},
            "angle": {"data": theta, "descriptor": DataDescriptor("angle", DataType.NDARRAY, units="rad",
                                                                  group="time_series", plot_type="line",
                                                                  x_axis="time", role="latent")},
            "angular_velocity": {"data": omega, "descriptor": DataDescriptor(
                "angular_velocity", DataType.NDARRAY, units="rad/s", group="angular_velocity", x_axis="time",
                role="latent")},
            "energy": {"data": energy, "descriptor": DataDescriptor("energy", DataType.NDARRAY, units="J/kg",
                                                                    group="energy", x_axis="time", role="label")},
            "period": {"data": float(self.period()), "descriptor": DataDescriptor(
                "period", DataType.FLOAT, units="s", group="summary", role="label")},
        }
