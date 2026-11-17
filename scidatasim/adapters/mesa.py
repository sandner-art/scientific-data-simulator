# scidatasim/adapters/mesa.py
"""Run any Mesa model as a simulator experiment.

`Mesa <https://mesa.readthedocs.io>`_ is the standard Python framework for agent-based models.
:class:`MesaExperiment` wraps a ``mesa.Model`` subclass, so a model written for Mesa gets the
simulator's recorded seeds, output roles, observation operators, bundles, designs, leak audit and
simulation studies without being rewritten::

    experiment_type: scidatasim.adapters.mesa.MesaExperiment
    model: mymodels.city.CityModel        # a mesa.Model subclass (dotted path)
    model_params: {density: 0.8, homophily: 0.4}
    n_steps: 100
    reporters:                            # model-level series, recorded at step 0 and after each step
      happy: n_happy                      #   an attribute of the model ...
      segregation: mymodels.city:segregation_index   # ... or a function of the model
    snapshots:                            # arrays recorded after the last step (fields)
      grid: grid_array
    stop_when_done: true                  # stop when model.running turns False (series end there)

The model's constructor receives its ``model_params`` and a seed (as ``rng`` on Mesa versions that
take it, else as ``seed``), drawn from the run's model stream, so a run is reproducible from its
recorded seed. Reporters must give numbers; snapshots give arrays. The parameters are the ground
truth; outputs are latent (exact) values, to be observed through ``observation`` operators.
Mesa models from elsewhere are labelled ``unverified``: no verification tests are registered for
them here. Mesa 3.0 or newer is needed (``pip install "scientific-data-simulator[mesa]"``).
"""
from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..base import ExperimentLogic
from ..utils import DataDescriptor, DataType, Role


def _mesa():
    try:
        import mesa
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError('Mesa models need Mesa 3: pip install "scientific-data-simulator[mesa]"') from e
    major = int(str(mesa.__version__).split(".")[0])
    if major < 3:
        raise ImportError(f"Mesa {mesa.__version__} is too old: the adapter needs Mesa 3 or newer")
    return mesa


def mesa_version() -> Optional[str]:
    try:
        import mesa
    except ImportError:
        return None
    return str(mesa.__version__)


def init_mesa_model(model, seed: Optional[int]) -> None:
    """Call ``mesa.Model.__init__`` with ``seed``, as ``rng`` where Mesa takes it (3.2 and newer;
    ``seed`` is deprecated there) and as ``seed`` before. Use in a model's own ``__init__``."""
    mesa = _mesa()
    if "rng" in inspect.signature(mesa.Model.__init__).parameters:
        mesa.Model.__init__(model, rng=seed)
    else:  # Mesa 3.0 / 3.1
        mesa.Model.__init__(model, seed=seed)


def _load(path: str):
    """``module.Name`` or ``module:Name``."""
    module, sep, name = path.partition(":")
    if not sep:
        module, _, name = path.rpartition(".")
    if not module or not name:
        raise ValueError(f"{path!r} is not a dotted path to a class or function")
    return getattr(importlib.import_module(module), name)


def _getter(spec: str) -> Callable[[Any], Any]:
    if ":" in spec:
        return _load(spec)
    return lambda model: getattr(model, spec)


def construct(model_cls, params: Dict[str, Any], seed: int):
    """Build ``model_cls(**params)`` passing ``seed`` the way its constructor accepts it."""
    signature = inspect.signature(model_cls.__init__).parameters
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.values())
    if "rng" in signature or ("seed" not in signature and accepts_kwargs and _takes_rng()):
        return model_cls(**params, rng=seed)
    if "seed" in signature or accepts_kwargs:
        return model_cls(**params, seed=seed)
    raise TypeError(f"{model_cls.__name__} must accept 'rng' or 'seed' (the run's seed) in its constructor")


def _takes_rng() -> bool:
    return "rng" in inspect.signature(_mesa().Model.__init__).parameters


class MesaExperiment(ExperimentLogic):
    """A Mesa model run under the simulator (see the module docstring)."""

    fidelity_level = "unverified"
    fidelity_note = "A Mesa model from outside the simulator: no verification tests are registered for it here."

    #: Subclasses (model packs) set these instead of taking them from the config.
    model_class: Optional[type] = None
    reporters: Dict[str, Any] = {}
    snapshots: Dict[str, Any] = {}
    units: Dict[str, str] = {}

    def __init__(self, config: Dict[str, Any]):
        _mesa()
        cls = self.model_class
        if cls is None:
            if "model" not in config:
                raise ValueError("MesaExperiment needs 'model': a dotted path to a mesa.Model subclass")
            cls = _load(config["model"])
        self.model_cls = cls
        self.model_params = dict(self._model_params(config))
        reporters = config.get("reporters", self.reporters)
        snapshots = config.get("snapshots", self.snapshots)
        if not reporters:
            raise ValueError("MesaExperiment needs at least one reporter (a model attribute or function)")
        self._reporters = {name: spec if callable(spec) else _getter(spec) for name, spec in reporters.items()}
        self._snapshots = {name: spec if callable(spec) else _getter(spec) for name, spec in snapshots.items()}
        self.stop_when_done = bool(config.get("stop_when_done", True))
        self.model = None
        self.series: Dict[str, list] = {}
        self.steps_run = 0

    def _model_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return config.get("model_params") or {}

    def ground_truth_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        params = {k: v for k, v in self.model_params.items()
                  if isinstance(v, (bool, int, float, str)) or (
                      isinstance(v, list) and all(isinstance(x, (int, float)) for x in v))}
        params["mesa_model"] = f"{self.model_cls.__module__}.{self.model_cls.__qualname__}"
        params["mesa_version"] = mesa_version()
        return params

    def _record(self) -> None:
        for name, get in self._reporters.items():
            value = get(self.model)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
                raise TypeError(f"reporter {name!r} must give a number, got {type(value).__name__}")
            self.series[name].append(float(value))

    def initialize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        seed = int(self.get_rng(config).integers(2 ** 62))
        self.model = construct(self.model_cls, self.model_params, seed)
        self.series = {name: [] for name in self._reporters}
        self.steps_run = 0
        self._record()
        return {}

    def run_step(self, state: Dict[str, Any], step: int) -> Dict[str, Any]:
        if self.stop_when_done and not getattr(self.model, "running", True):
            return state
        self.model.step()
        self.steps_run += 1
        self._record()
        return state

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        steps = np.arange(self.steps_run + 1, dtype=float)
        out = {"step": {"data": steps, "descriptor": DataDescriptor(
            "step", DataType.NDARRAY, shape=steps.shape, units="step", group="time_series", x_axis="step",
            role=Role.COORDINATE)}}
        for name, values in self.series.items():
            data = np.asarray(values, dtype=float)
            out[name] = {"data": data, "descriptor": DataDescriptor(
                name, DataType.NDARRAY, shape=data.shape, units=self.units.get(name, "dimensionless"),
                group="time_series", plot_type="line", x_axis="step", role=Role.LATENT)}
        for name, get in self._snapshots.items():
            data = np.asarray(get(self.model), dtype=float)
            out[name] = {"data": data, "descriptor": DataDescriptor(
                name, DataType.NDARRAY, shape=data.shape, units=self.units.get(name, "dimensionless"),
                group="field" if data.ndim >= 2 else "histogram",
                plot_type=None if data.ndim >= 2 else "histogram", role=Role.LATENT)}
        return out
