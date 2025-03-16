# simulator/engine.py
import os
import json
import platform
import importlib
import sys
import logging
import psutil
# import subprocess # Removed
from packaging import version # Added
import pkg_resources # Added

from .base import ExperimentLogic
from .experiment_record import ExperimentRecord
from .config import load_config
from .utils import DataDescriptor, DataType
from typing import Dict, Any, Type
from datetime import datetime # Added import


# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimulatorEngine:
    """
    The core engine for running scientific simulations.
    """

    def __init__(self, output_dir: str = "experiments_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"SimulatorEngine initialized. Output directory: {self.output_dir}")

    def run_experiment(self, config_path: str):
        """
        Runs an experiment based on the provided configuration file.

        Args:
            config_path: Path to the YAML configuration file.
        """
        config = load_config(config_path)
        experiment_logic_class = self._get_experiment_logic_class(config)

        record = ExperimentRecord(config, experiment_logic_class)
        logger.info(f"Starting experiment: {record.experiment_id}")

        # System Info
        system_info = {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "ram": str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + "GB",
            "python_version": platform.python_version(),
        }
        record.set_system_info(system_info)

        # Get Software versions
        try:
            installed_packages = {p.project_name: p.version for p in pkg_resources.working_set}
            record.set_software_versions(installed_packages)
        except Exception as e:
            record.add_log_message(f"Failed to get software versions: {e}")
            logger.warning(f"Failed to get software versions: {e}")

        try:
            # Initialize experiment logic
            experiment_logic = experiment_logic_class(config)
            state = experiment_logic.initialize(config)

            # Run simulation steps (if applicable)
            if hasattr(experiment_logic, "run_step"):
                for step in range(config.get("n_steps", 1)):  # Default to 1 step if not specified
                    state = experiment_logic.run_step(state, step)
                    logger.debug(f"Step {step}: State = {state}")

            # Get results
            results = experiment_logic.get_results()
            self._validate_results(results)

            # Add output to record
            for data_name, data_info in results.items():
                record.add_output_data(data_name, data_info['data'], data_info['descriptor'])

            # Optional Visualization
            experiment_logic.visualize(results)

            record.set_end_time()
            logger.info(f"Experiment completed: {record.experiment_id}")

        except Exception as e:
            record.add_log_message(f"Experiment failed: {e}")
            import traceback
            record.add_log_message(traceback.format_exc())
            self._save_experiment_record(record)  # Save even on failure
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise  # Re-raise the exception to inform the user

        self._save_experiment_record(record)  # save the record
        return record.experiment_id

    def _get_experiment_logic_class(self, config: Dict[str, Any]) -> Type[ExperimentLogic]:
        """Loads the ExperimentLogic class based on the configuration."""
        experiment_type = config["experiment_type"]
        module_name, class_name = experiment_type.rsplit(".", 1)
        module = importlib.import_module(module_name)
        experiment_logic_class = getattr(module, class_name)

        if not issubclass(experiment_logic_class, ExperimentLogic):
            raise TypeError(f"'{experiment_type}' must be a subclass of ExperimentLogic.")

        return experiment_logic_class

    def _validate_results(self, results: Dict[str, Dict[str, Any]]):
        """Validates the results returned by the ExperimentLogic."""
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

            # Type checking (add more checks as needed)
            data = data_info['data']
            if descriptor.data_type == DataType.FLOAT and not isinstance(data, float):
                raise TypeError(f"Data for '{data_name}' must be a float (based on descriptor).")
            # ... add checks for other data types as in previous examples ...

    def _save_experiment_record(self, record: ExperimentRecord):
        """Saves the ExperimentRecord to a JSON file."""
        experiment_dir = os.path.join(self.output_dir, record.experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        record_path = os.path.join(experiment_dir, "experiment_record.json")
        with open(record_path, "w") as f:
            json.dump(record.to_dict(), f, indent=4)
        logger.info(f"Experiment record saved to: {record_path}")

    def load_experiment_record(self, experiment_id: str) -> ExperimentRecord:
        """Loads an experiment record from disk."""
        record_path = os.path.join(self.output_dir, experiment_id, "experiment_record.json")
        with open(record_path, "r") as f:
            data = json.load(f)

        # Re-create DataDescriptor objects (important!)
        input_descriptors = {
            name: DataDescriptor(**desc_data)
            for name, desc_data in data["input_data_descriptors"].items()
        }
        output_data = {
            name: {
                "data": data_info["data"],
                "descriptor": DataDescriptor(**data_info["descriptor"])
            } for name, data_info in data["output_data"].items()
        }
        # recreate experiment record:
        record = ExperimentRecord({}, None)  # dummy values
        record.experiment_id = data['experiment_id']
        record.start_time = datetime.fromisoformat(data['start_time'])
        record.end_time = datetime.fromisoformat(data['end_time']) if data['end_time'] else None
        record.config = data['config']
        record.experiment_logic_class_name = data['experiment_logic_class_name']
        record.experiment_logic_module = data['experiment_logic_module']
        record.input_data_descriptors = input_descriptors
        record.output_data = output_data
        record.log_messages = data['log_messages']
        record.system_info = data['system_info']
        record.software_versions = data['software_versions']
        record.llm_usage = data['llm_usage']
        return record