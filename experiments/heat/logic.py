# experiments/heat/logic.py
"""Heat (diffusion) equation on a periodic domain, with a known exact solution.

    u_t = alpha * Laplacian(u),    x in [0, L)^d (periodic),  d = 1 or 2

Initial condition: a sum of Fourier modes on top of a constant mean,

    u(x, 0) = mean + sum_j [a_j cos(k_j . x) + b_j sin(k_j . x)],   k_j = 2 pi n_j / L,

with integer wave vectors n_j (0 < max|n_j| <= k_max < nx / 2, so every mode is
resolved on the grid). The modes are drawn from ``self.rng`` (``n_modes``,
``k_max``, ``amplitude``) or given explicitly (``modes``). Each mode decays as
``exp(-alpha |k_j|^2 t)``, so the exact solution is known everywhere.

Parameters (SI units; compatible units may be given, e.g. ``alpha: "1 mm^2/s"``):
alpha [m^2/s], length L [m], dt [s]; dims, nx (grid points per dimension),
n_steps, integrator, n_modes, k_max, amplitude and mean (field units, ``units``,
default ``K``), probe_index (grid index of the probe point, default ``nx // 3``
and ``nx // 7``: an off-symmetry point where low modes rarely vanish).

Integrators:

* ``exact`` - the spectral exact solution evaluated on the grid (``reference``)
* ``ftcs``  - explicit forward-time centred-space finite differences (``audit``):
  first order in time, second order in space, and unstable when
  ``r = alpha dt / dx^2`` exceeds ``1 / (2 d)``. A deliberately weak model for
  testing whether methods notice numerical error and instability.

Outputs: ``time`` and ``x`` (and ``y``) coordinates; ``u`` the full field
(latent, shape ``(n_steps + 1, nx)`` or ``(n_steps + 1, nx, nx)``, indexed
``u[t, i, j] = u(t, x_i, y_j)``); ``u_probe`` the field at the probe point (latent
time series); labels ``mean`` (spatial mean, conserved) and ``l2_norm``
(discrete L2 norm ``sqrt(sum u^2 dx^d)``, non-increasing).

Configuration example::

    experiment_type: experiments.heat.logic.HeatExperiment
    dims: 2
    nx: 32
    length: 1.0          # m
    alpha: "1 mm^2/s"
    dt: 1.0              # s
    n_steps: 200
    integrator: exact    # or ftcs
    n_modes: 4
    k_max: 3
    # modes: [{k: [1, 0], a: 1.0, b: 0.0}, {k: [2, 1], a: 0.0, b: 0.5}]  # instead of random modes
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from simulator.base import ExperimentLogic
from simulator.utils import DataDescriptor, DataType

INTEGRATORS = ("exact", "ftcs")


class HeatExperiment(ExperimentLogic):
    parameter_units = {"alpha": "m**2/s", "length": "m", "dt": "s"}
    inferable_parameters = ("alpha",)

    def __init__(self, config: Dict[str, Any]):
        self.dims = int(config.get("dims", 1))
        self.nx = int(config.get("nx", 64))
        self.length = float(config.get("length", 1.0))
        self.alpha = float(config.get("alpha", 0.01))
        self.dt = float(config.get("dt", 0.01))
        self.n_steps = int(config["n_steps"])
        self.integrator = config.get("integrator", "exact")
        self.n_modes = int(config.get("n_modes", 4))
        self.k_max = int(config.get("k_max", 4))
        self.amplitude = float(config.get("amplitude", 1.0))
        self.mean = float(config.get("mean", 1.0))
        self.units = str(config.get("units", "K"))
        self._given_modes = config.get("modes")
        self._seed = config.get("seed")

        if self.dims not in (1, 2):
            raise ValueError("dims must be 1 or 2")
        if self.nx < 4:
            raise ValueError("nx must be >= 4")
        if self.length <= 0 or self.dt <= 0 or self.alpha < 0:
            raise ValueError("length and dt must be > 0 and alpha >= 0")
        if self.n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        if self.integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}")
        if self._given_modes is None:
            if self.n_modes < 1 or not (1 <= self.k_max < self.nx / 2):
                raise ValueError("need n_modes >= 1 and 1 <= k_max < nx / 2 (modes must be resolved on the grid)")

        probe = config.get("probe_index", [self.nx // 3, self.nx // 7][:self.dims])
        probe = [int(probe)] * self.dims if np.isscalar(probe) else [int(p) for p in probe]
        if len(probe) != self.dims or not all(0 <= p < self.nx for p in probe):
            raise ValueError(f"probe_index must be {self.dims} grid index/indices in [0, nx)")
        self.probe_index = tuple(probe)

        self.dx = self.length / self.nx
        self.r = self.alpha * self.dt / self.dx ** 2  # FTCS stability number
        self.r_limit = 1.0 / (2 * self.dims)
        self.modes: Optional[List[Dict[str, Any]]] = None
        if self._given_modes is not None:
            self.modes = self._parse_modes(self._given_modes)

    # ------------------------------------------------------------------ initial condition
    def _parse_modes(self, modes) -> List[Dict[str, Any]]:
        parsed = []
        for m in modes:
            k = m["k"]
            k = [int(k)] if np.isscalar(k) else [int(v) for v in k]
            if len(k) != self.dims:
                raise ValueError(f"mode wave vector {k} must have {self.dims} component(s)")
            if not any(k) or max(abs(v) for v in k) >= self.nx / 2:
                raise ValueError(f"mode wave vector {k} must be non-zero with |n| < nx / 2")
            parsed.append({"k": k, "a": float(m.get("a", 0.0)), "b": float(m.get("b", 0.0))})
        return parsed

    def _draw_modes(self, rng: np.random.Generator) -> List[Dict[str, Any]]:
        modes = []
        for _ in range(self.n_modes):
            if self.dims == 1:
                k = [int(rng.integers(1, self.k_max + 1))]
            else:
                k = [0, 0]
                while not any(k):
                    k = [int(v) for v in rng.integers(-self.k_max, self.k_max + 1, size=2)]
            a, b = rng.normal(0.0, self.amplitude, size=2)
            modes.append({"k": k, "a": float(a), "b": float(b)})
        return modes

    def _resolve_modes(self) -> List[Dict[str, Any]]:
        if self.modes is None:
            self.modes = self._draw_modes(self.get_rng({"seed": self._seed}))
        return self.modes

    # ------------------------------------------------------------------ grid and solutions
    def coordinates(self) -> np.ndarray:
        return np.arange(self.nx) * self.dx

    def _mesh(self) -> List[np.ndarray]:
        x = self.coordinates()
        return [x] if self.dims == 1 else list(np.meshgrid(x, x, indexing="ij"))

    def _mode_patterns(self):
        """(wave vectors |k|^2, spatial patterns a cos(k.x) + b sin(k.x)) of all modes."""
        mesh = self._mesh()
        k2, patterns = [], []
        for m in self._resolve_modes():
            kvec = [2.0 * np.pi * n / self.length for n in m["k"]]
            phase = sum(kc * xc for kc, xc in zip(kvec, mesh))
            patterns.append(m["a"] * np.cos(phase) + m["b"] * np.sin(phase))
            k2.append(sum(kc ** 2 for kc in kvec))
        return np.array(k2), np.array(patterns)

    def exact_solution(self, t: np.ndarray) -> np.ndarray:
        """Exact field at times ``t``: shape ``(len(t), nx[, nx])``."""
        k2, patterns = self._mode_patterns()
        decay = np.exp(-self.alpha * np.outer(np.asarray(t, dtype=float), k2))  # (n_t, n_modes)
        return self.mean + np.tensordot(decay, patterns, axes=(1, 0))

    def ftcs_solution(self) -> np.ndarray:
        """FTCS: u <- u + r * sum_d (u[i+1] - 2 u[i] + u[i-1]) along each axis."""
        u = self.exact_solution(np.array([0.0]))[0]
        out = np.empty((self.n_steps + 1,) + u.shape)
        out[0] = u
        with np.errstate(over="ignore", invalid="ignore"):
            for n in range(1, self.n_steps + 1):
                lap = -2.0 * self.dims * u
                for axis in range(self.dims):
                    lap = lap + np.roll(u, 1, axis=axis) + np.roll(u, -1, axis=axis)
                u = u + self.r * lap
                out[n] = u
        return out

    # ------------------------------------------------------------------ declared properties
    def fidelity(self, config):
        if self.integrator == "exact":
            return {"level": "reference", "note": (
                "Spectral exact solution of the periodic heat equation (each Fourier mode decays as "
                "exp(-alpha |k|^2 t)), evaluated on the grid; tests check the PDE residual, conservation "
                "of the mean, the decay of the L2 norm and FTCS convergence to it.")}
        note = (f"Explicit FTCS finite differences with r = alpha dt / dx^2 = {self.r:.4g} "
                f"(stability limit 1/(2d) = {self.r_limit:g}): first order in time, second order in space, "
                "so the field carries discretisation error. Kept as a deliberately weak model to test "
                "whether methods notice numerical error and instability.")
        if self.r > self.r_limit:
            note += (f" UNSTABLE in this configuration: r exceeds the limit, so grid-scale components "
                     f"(including round-off) grow by |1 - 4 d r| = {abs(1 - 4 * self.dims * self.r):.3g} per "
                     "step and the field blows up.")
        return {"level": "audit", "note": note}

    def invariants(self, results):
        """The spatial mean is conserved (periodic domain, no sources)."""
        return {"mean": results["mean"]["data"]}

    def mechanism_graph(self, config):
        return {"directed": True, "nodes": [{"id": "u", "kind": "field"}],
                "edges": [{"source": "u", "target": "u", "sign": "-", "parameter": "alpha",
                           "value": self.alpha,
                           "term": "diffusion alpha * Laplacian(u): each Fourier mode decays at rate alpha |k|^2"}]}

    def ground_truth_parameters(self, config):
        params = super().ground_truth_parameters(config)
        params["modes"] = [dict(m, k=list(m["k"])) for m in self._resolve_modes()]
        params["dx"] = self.dx
        params["stability_number"] = self.r
        return params

    # ------------------------------------------------------------------ ExperimentLogic
    def initialize(self, config):
        self._resolve_modes()  # draws from self.rng, which the engine has set by now
        return {}

    def run_step(self, state, step):
        return state  # the field is computed on the full time grid in get_results

    def get_results(self):
        t = np.arange(self.n_steps + 1) * self.dt
        u = self.exact_solution(t) if self.integrator == "exact" else self.ftcs_solution()
        spatial = tuple(range(1, self.dims + 1))
        with np.errstate(over="ignore", invalid="ignore"):
            mean = u.mean(axis=spatial)
            l2 = np.sqrt(np.sum(u ** 2, axis=spatial) * self.dx ** self.dims)
        probe = u[(slice(None),) + self.probe_index]
        x = self.coordinates()
        l2_units = f"{self.units}*m**0.5" if self.dims == 1 else f"{self.units}*m"

        def desc(name, **kw):
            return DataDescriptor(name, DataType.NDARRAY, **kw)

        results = {
            "time": {"data": t, "descriptor": desc("time", units="s", group="time_series", role="coordinate")},
            "x": {"data": x, "descriptor": desc("x", units="m", group="field", role="coordinate")},
        }
        if self.dims == 2:
            results["y"] = {"data": x.copy(), "descriptor": desc("y", units="m", group="field", role="coordinate")}
        results.update({
            "u": {"data": u, "descriptor": desc("u", units=self.units, group="field", role="latent")},
            "u_probe": {"data": np.ascontiguousarray(probe), "descriptor": desc(
                "u_probe", units=self.units, group="time_series", plot_type="line", x_axis="time", role="latent")},
            "mean": {"data": mean, "descriptor": desc("mean", units=self.units, group="mean", x_axis="time",
                                                       role="label")},
            "l2_norm": {"data": l2, "descriptor": desc("l2_norm", units=l2_units, group="l2_norm", x_axis="time",
                                                        role="label")},
        })
        return results
