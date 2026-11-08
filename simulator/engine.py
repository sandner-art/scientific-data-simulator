# simulator/engine.py
from __future__ import annotations

import copy
import importlib
import logging
import numbers
import os
import platform
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type

import numpy as np
import pandas as pd
import psutil

from .base import ExperimentLogic
from .config import apply_overrides, get_by_path, load_config
from .experiment_record import ExperimentRecord
from .persistence import (experiment_dir_for, load_experiment_record, save_experiment_record,
                          update_record_json)
from .provenance import code_provenance, collect_software_versions, config_sha256, simulator_version
from .rng import RunRandomness
from .observation import apply_observation, validate_observation_config
from .schema import parse_law_shift, validate_engine_config
from .bundle import build_bundle, resolve_bundle_config
from .storage import resolve_storage_config
from .utils import FIDELITY_LEVELS, GROUND_TRUTH_ROLES, DataDescriptor, DataType, Role

logger = logging.getLogger(__name__)


@dataclass
class RunOutput:
    """Everything produced by one engine run."""
    experiment_id: str
    experiment_dir: str
    record: ExperimentRecord
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SimulatorEngine:
    """
    The core engine for running scientific simulations.

    ``run_experiment(config_path)`` runs a YAML-configured experiment;
    ``run_config(config)`` runs an already-loaded configuration dictionary and
    is the single code path used by parameter sweeps as well.
    """

    def __init__(self, output_dir: str = "experiments_output", verbose: bool = True):
        self.output_dir = output_dir
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("SimulatorEngine initialized. Output directory: %s", self.output_dir)

    # ------------------------------------------------------------------ public API
    def run_experiment(self, config_path: str) -> str:
        """Runs an experiment from a YAML configuration file and returns its ID."""
        config = load_config(config_path)
        return self.run_config(config).experiment_id

    def run_config(self, config: Dict[str, Any],
                   experiment_logic_class: Optional[Type[ExperimentLogic]] = None,
                   generate_plots: Optional[bool] = None) -> RunOutput:
        """Runs one experiment from a configuration dictionary.

        Args:
            config: Experiment configuration (as loaded from YAML).
            experiment_logic_class: Optional class to use instead of resolving
                ``config['experiment_type']``.
            generate_plots: Override the ``generate_plots`` config option.
        """
        config = copy.deepcopy(config)
        replaced_type = None
        if experiment_logic_class is None:
            experiment_logic_class = self._get_experiment_logic_class(config)
        elif not (isinstance(experiment_logic_class, type) and issubclass(experiment_logic_class, ExperimentLogic)):
            raise TypeError("experiment_logic_class must be a subclass of ExperimentLogic")
        else:
            # An explicitly passed class wins; the stored recipe must name it.
            class_path = f"{experiment_logic_class.__module__}.{experiment_logic_class.__qualname__}"
            if config.get("experiment_type") not in (None, class_path):
                replaced_type = config["experiment_type"]
            config["experiment_type"] = class_path
        engine_config = validate_engine_config(config)
        observation_config = validate_observation_config(config.get("observation"))
        law_shift = parse_law_shift(config.get("law_shift"))
        resolve_storage_config(config.get("storage"))
        bundle_config = resolve_bundle_config(config.get("bundle"))

        # One recorded seed per run; written back into the config so that the
        # stored config re-runs identically.
        randomness = RunRandomness.from_config_seed(config.get("seed"))
        config["seed"] = randomness.seed
        streams = randomness.streams()

        record = ExperimentRecord(config, experiment_logic_class)
        if replaced_type:
            record.add_log_message(f"experiment_type '{replaced_type}' replaced by the explicitly "
                                   f"passed class '{config['experiment_type']}'")
        record.randomness = randomness.to_dict()
        record.provenance = {
            "synthetic": bool(getattr(experiment_logic_class, "synthetic", True)),
            "config_sha256": config_sha256(config),
            "simulator_version": simulator_version(),
            "code": code_provenance(experiment_logic_class),
        }
        experiment_dir = experiment_dir_for(record, self.output_dir)
        os.makedirs(experiment_dir, exist_ok=True)
        logger.info("Starting experiment: %s (seed %d)", record.experiment_id, randomness.seed)

        record.set_system_info(_system_info())
        try:
            record.set_software_versions(collect_software_versions())
        except Exception as e:  # provenance must never abort a run
            record.add_log_message(f"Failed to get software versions: {e}")
            logger.warning("Failed to get software versions: %s", e)

        try:
            # Counterfactual law shift: the model sees the shifted values; the
            # record keeps the original recipe and both values in ground_truth.
            # The model always gets its own copy, so it cannot alter the stored recipe.
            model_config = copy.deepcopy(config)
            shift_record = None
            if law_shift:
                model_config = apply_overrides(config, law_shift["changes"])
                shift_record = {"description": law_shift["description"], "changes": {
                    key: {"original": _get_or_none(config, key), "shifted": value}
                    for key, value in law_shift["changes"].items()}}

            experiment_logic = experiment_logic_class(model_config)
            experiment_logic.rng = streams["model"]
            fidelity = _fidelity(experiment_logic, model_config)
            record.provenance["fidelity"] = fidelity
            if fidelity["level"] != "reference":
                record.add_log_message(f"Model fidelity '{fidelity['level']}': {fidelity['note']}")
                if fidelity["level"] in ("audit", "unverified"):
                    logger.warning("Model fidelity '%s': %s", fidelity["level"], fidelity["note"])
            state = experiment_logic.initialize(model_config)
            for step in range(int(model_config.get("n_steps", 1))):
                state = experiment_logic.run_step(state, step)

            results = experiment_logic.get_results()
            results = self._validate_results(results)

            realized_observation = None
            if observation_config:
                added, realized_observation = apply_observation(results, observation_config,
                                                                streams["observation"])
                clash = set(added) & set(results)
                if clash:
                    raise ValueError(f"Observation outputs {sorted(clash)} clash with model outputs")
                results = {**results, **self._validate_results(added)}

            for data_name, data_info in results.items():
                record.add_output_data(data_name, data_info['data'], data_info['descriptor'])

            record.ground_truth = _ground_truth(record, experiment_logic, model_config, results,
                                                shift_record, realized_observation)

            # Use validated values: "no"/"false" strings from YAML or $ENV are booleans here.
            plots = engine_config.generate_plots if generate_plots is None else generate_plots
            if plots:
                self._generate_plots(results, experiment_dir, engine_config.static_plot_format, record)

            if engine_config.save_csv:
                try:
                    from .data_handler import save_csv
                    save_csv(results, os.path.join(experiment_dir, "results.csv"))
                except Exception as e:
                    record.add_log_message(f"Could not save CSV: {e}")
                    logger.error("Could not save CSV: %s", e)

            record.set_end_time()
        except Exception as e:
            record.add_log_message(f"Experiment failed: {e}")
            record.add_log_message(traceback.format_exc())
            record.set_end_time()
            try:
                save_experiment_record(record, self.output_dir)  # save even on failure
            except Exception as save_error:  # never mask the original error
                logger.error("Could not save record of failed run: %s", save_error)
            logger.error("Experiment failed: %s", e, exc_info=True)
            raise

        # Raises if data files cannot be written; a metadata-only record is still left behind.
        experiment_dir = save_experiment_record(record, self.output_dir)

        if bundle_config:
            try:
                record.bundle = build_bundle(experiment_dir, record, results, experiment_logic, model_config,
                                             bundle_config, streams["bundle"])
            except Exception as e:
                record.add_log_message(f"Bundle generation failed: {type(e).__name__}: {e}")
                update_record_json(experiment_dir, record, "log_messages")
                raise
            update_record_json(experiment_dir, record, "bundle")
        logger.info("Experiment completed: %s (%s)", record.experiment_id, experiment_dir)
        if self.verbose:
            _print_output_summary(experiment_dir)
        return RunOutput(record.experiment_id, experiment_dir, record, results)

    def load_experiment_record(self, experiment_id: str) -> ExperimentRecord:
        """Loads an experiment record from disk, using the persistence module."""
        return load_experiment_record(self.output_dir, experiment_id)

    # ------------------------------------------------------------------ internals
    def _get_experiment_logic_class(self, config: Dict[str, Any]) -> Type[ExperimentLogic]:
        """Loads the ExperimentLogic class named by ``config['experiment_type']``."""
        try:
            experiment_type = config["experiment_type"]
        except KeyError:
            raise KeyError("Configuration must define 'experiment_type' "
                           "(dotted path to an ExperimentLogic subclass).") from None
        module_name, _, class_name = experiment_type.rpartition(".")
        if not module_name:
            raise ImportError(f"'{experiment_type}' is not a dotted path 'module.ClassName'.")
        module = importlib.import_module(module_name)
        try:
            experiment_logic_class = getattr(module, class_name)
        except AttributeError:
            raise ImportError(f"Module '{module_name}' has no class '{class_name}'.") from None
        if not (isinstance(experiment_logic_class, type) and issubclass(experiment_logic_class, ExperimentLogic)):
            raise TypeError(f"'{experiment_type}' must be a subclass of ExperimentLogic.")
        return experiment_logic_class

    def _validate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Validates results returned by an ExperimentLogic.

        Checks structure, that values match the declared ``DataType`` (numpy
        scalars are accepted for FLOAT/INT), and that declared array/DataFrame
        shapes match the data. Returns the results with missing shapes filled
        in on *copies* of the descriptors.
        """
        if not isinstance(results, dict):
            raise TypeError("get_results() must return a dictionary.")
        validated: Dict[str, Dict[str, Any]] = {}
        for data_name, data_info in results.items():
            if not isinstance(data_info, dict):
                raise TypeError(f"Result for '{data_name}' must be a dictionary.")
            if "data" not in data_info:
                raise KeyError(f"Result for '{data_name}' must contain a 'data' key.")
            if "descriptor" not in data_info:
                raise KeyError(f"Result for '{data_name}' must contain a 'descriptor' key.")
            descriptor = data_info["descriptor"]
            if not isinstance(descriptor, DataDescriptor):
                raise TypeError(f"Descriptor for '{data_name}' must be a DataDescriptor instance.")

            data = data_info['data']
            expected = descriptor.data_type
            ok = {
                DataType.FLOAT: isinstance(data, numbers.Real) and not isinstance(data, (bool, np.bool_)),
                DataType.INT: isinstance(data, numbers.Integral) and not isinstance(data, (bool, np.bool_)),
                DataType.STRING: isinstance(data, str),
                DataType.LIST: isinstance(data, list),
                DataType.NDARRAY: isinstance(data, np.ndarray),
                DataType.DATAFRAME: isinstance(data, pd.DataFrame),
            }[expected]
            if not ok:
                raise TypeError(f"Data for '{data_name}' must be of type {expected.value} "
                                f"(based on descriptor), got {type(data).__name__}.")

            if expected in (DataType.NDARRAY, DataType.DATAFRAME):
                if descriptor.shape is None:
                    # copy, so a descriptor shared across runs is never modified
                    validated[data_name] = {**data_info, "descriptor": descriptor.replace(shape=tuple(data.shape))}
                    continue
                elif tuple(descriptor.shape) != tuple(data.shape):
                    raise ValueError(f"Descriptor shape {tuple(descriptor.shape)} for '{data_name}' "
                                     f"does not match data shape {tuple(data.shape)}.")
            validated[data_name] = data_info
        return validated

    def _generate_plots(self, results, experiment_dir, static_format, record) -> None:
        from .visualization import generate_plots
        try:
            generate_plots(results, experiment_dir, static_format=static_format)
        except ValueError:
            raise  # invalid configuration, e.g. unknown static_plot_format
        except Exception as e:  # e.g. kaleido/Chrome missing: keep the data, note the failure
            record.add_log_message(f"Plot generation failed: {e}")
            logger.warning("Plot generation failed: %s", e)


def _get_or_none(config: Dict[str, Any], key: str) -> Any:
    try:
        return copy.deepcopy(get_by_path(config, key))
    except (KeyError, IndexError, TypeError):
        return None


def _fidelity(logic: ExperimentLogic, model_config: Dict[str, Any]) -> Dict[str, Any]:
    info = dict(logic.fidelity(model_config))
    if info.get("level") not in FIDELITY_LEVELS:
        raise ValueError(f"{type(logic).__name__}.fidelity() returned unknown level {info.get('level')!r}; "
                         f"use one of {sorted(FIDELITY_LEVELS)}")
    info.setdefault("note", None)
    info["meaning"] = FIDELITY_LEVELS[info["level"]]
    return info


def _ground_truth(record, logic, model_config, results, shift_record, realized_observation) -> Dict[str, Any]:
    """Everything a benchmark needs to score inference against the truth,
    kept apart from the observations themselves."""
    fields = {name: {"role": info["descriptor"].role.value if info["descriptor"].role else None,
                     "units": info["descriptor"].units,
                     "data_type": info["descriptor"].data_type.value}
              for name, info in results.items()}
    by_role = lambda roles: sorted(n for n, f in fields.items() if f["role"] in roles)  # noqa: E731
    return {
        "experiment_id": record.experiment_id,
        "experiment_type": model_config.get("experiment_type"),
        "seed": record.randomness.get("seed"),
        "synthetic": record.provenance.get("synthetic", True),
        "fidelity": record.provenance.get("fidelity"),
        "parameters": logic.ground_truth_parameters(model_config),
        "law_shift": shift_record,
        "observation": realized_observation,
        "fields": fields,
        "ground_truth_fields": by_role({r.value for r in GROUND_TRUTH_ROLES}),
        "observation_fields": by_role({Role.OBSERVATION.value}),
    }


def _system_info() -> Dict[str, Any]:
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "ram": f"{round(psutil.virtual_memory().total / (1024.0 ** 3))}GB",
        "python_version": platform.python_version(),
    }


def _print_output_summary(experiment_dir: str) -> None:
    print("\nExperiment completed successfully. Output files:")
    for root, _dirs, files in os.walk(experiment_dir):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            size = float(os.path.getsize(file_path))
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if size < 1024.0:
                    break
                size /= 1024.0
            print(f"  - {os.path.relpath(file_path, experiment_dir)}: {size:.2f} {unit}")
    print(f"Output directory: {experiment_dir}\n")
