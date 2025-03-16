# simulator/engine.py
import os
import json
import platform
import importlib
import sys
import logging
import psutil
from packaging import version
import pkg_resources

from .base import ExperimentLogic
from .experiment_record import ExperimentRecord
from .config import load_config
from .utils import DataDescriptor, DataType
from typing import Dict, Any, Type, Optional
from datetime import datetime
import pandas as pd  # Import pandas
import numpy as np  # Import numpy
from .visualization import generate_plots  # Import the new function


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

        # --- Corrected: Create experiment directory *before* anything else ---
        timestamp = record.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = os.path.join(self.output_dir, f"{timestamp}_{record.experiment_id}")
        os.makedirs(experiment_dir, exist_ok=True)  # Ensure directory exists
        # --- End Correction ---

        # System Info
        system_info = {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "ram": str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + "GB",
            "python_version": platform.python_version(),
        }
        record.set_system_info(system_info)

        # Get Software versions (using requirements.txt)
        try:
            installed_packages = {p.key: p.version for p in pkg_resources.working_set}
            relevant_versions = {}
            with open("requirements.txt", "r") as req_file:
                required_packages = [line.strip() for line in req_file if line.strip() and not line.startswith("#")]

            for package_name in required_packages:
                req = pkg_resources.Requirement.parse(package_name)
                if req.key in installed_packages:
                    relevant_versions[req.key] = installed_packages[req.key]
                else:
                    logger.warning(f"Package required not found: {req.project_name}")
            record.set_software_versions(relevant_versions)

        except Exception as e:
            record.add_log_message(f"Failed to get software versions: {e}")
            logger.warning(f"Failed to get software versions: {e}")

        try:
            # Initialize experiment logic
            experiment_logic = experiment_logic_class(config)
            state = experiment_logic.initialize(config)

            # Run simulation steps (if applicable)
            if hasattr(experiment_logic, "run_step"):
                for step in range(config.get("n_steps", 1)):  # Default to 1 step
                    state = experiment_logic.run_step(state, step)
                    logger.debug(f"Step {step}: State = {state}")

            # Get results
            results = experiment_logic.get_results()
            self._validate_results(results)

            # Add output to record
            for data_name, data_info in results.items():
                record.add_output_data(data_name, data_info['data'], data_info['descriptor'])

            # --- MODIFIED: Generate plots ---
            static_format = config.get('static_plot_format')
            if static_format == 'null': # convert to None
                static_format = None
            generate_plots(results, experiment_dir, static_format=static_format)  # Get from config
            # --- END MODIFIED ---

            # Save result to csv
            if config.get('save_csv', False): # Check for save_csv option.
                try:
                    from .data_handler import save_csv # Import here to avoid circular import
                    save_csv(results, os.path.join(experiment_dir, "results.csv"))
                except Exception as e:
                    logger.error(f"Could not save to csv: {e}")

            record.set_end_time()
            logger.info(f"Experiment completed: {record.experiment_id}")

        except Exception as e:
            record.add_log_message(f"Experiment failed: {e}")
            import traceback
            record.add_log_message(traceback.format_exc())
            self._save_experiment_record(record)  # Save even on failure
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise  # Re-raise

        experiment_id = self._save_experiment_record(record) # get experiment id

        # --- ADDED: Output file summary ---
        print("\nExperiment completed successfully. Output files:")
        for filename in os.listdir(experiment_dir):
            file_path = os.path.join(experiment_dir, filename)
            if os.path.isfile(file_path):
                try:
                    file_size = os.path.getsize(file_path)  # Get size in bytes
                    # Format size for readability (optional)
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if file_size < 1024.0:
                            break
                        file_size /= 1024.0
                    size_str = f"{file_size:.2f} {unit}"
                except OSError:
                    size_str = "Error getting size"
                print(f"  - {filename}: {size_str}")
        print(f"Output directory: {experiment_dir}\n")
        # --- END ADDED ---
        return experiment_id # return id

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
            if descriptor.data_type == DataType.INT and not isinstance(data, int):
                raise TypeError(f"Data for '{data_name}' must be an int (based on descriptor).")
            if descriptor.data_type == DataType.STRING and not isinstance(data, str):
                raise TypeError(f"Data for '{data_name}' must be a string (based on descriptor).")
            if descriptor.data_type == DataType.LIST and not isinstance(data, list):
                raise TypeError(f"Data for '{data_name}' must be a list (based on descriptor).")
            if descriptor.data_type == DataType.NDARRAY and not isinstance(data, np.ndarray):
                raise TypeError(f"Data for '{data_name}' must be a numpy array (based on descriptor).")
            if descriptor.data_type == DataType.DATAFRAME and not isinstance(data, pd.DataFrame):
                raise TypeError(f"Data for '{data_name}' must be a pandas DataFrame (based on descriptor).")

            # ... add checks for other data types as in previous examples ...

    def _save_experiment_record(self, record: ExperimentRecord):
        """Saves the ExperimentRecord to a JSON file."""
        timestamp = record.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = os.path.join(self.output_dir, f"{timestamp}_{record.experiment_id}")
        record_path = os.path.join(experiment_dir, "experiment_record.json")
        with open(record_path, "w") as f:
            json.dump(record.to_dict(), f, indent=4)
        logger.info(f"Experiment record saved to: {record_path}")
        return record.experiment_id # Return experiment ID


    def load_experiment_record(self, experiment_id: str) -> ExperimentRecord:
        """Loads an experiment record from disk."""
        # record_path = os.path.join(self.output_dir, experiment_id, "experiment_record.json") # Incorrect
        # Find the experiment directory (it will have the timestamp prefix)
        experiment_dir = None
        for item in os.listdir(self.output_dir):
            item_path = os.path.join(self.output_dir, item)
            if os.path.isdir(item_path) and experiment_id in item:
                experiment_dir = item_path
                break
        if experiment_dir is None:
            raise FileNotFoundError(f"Experiment directory with ID '{experiment_id}' not found in '{self.output_dir}'.")
        record_path = os.path.join(experiment_dir, "experiment_record.json")
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
        record = ExperimentRecord(data['config'], None)  # Pass the *loaded* config
        record.experiment_id = data['experiment_id']
        record.start_time = datetime.fromisoformat(data['start_time'])
        record.end_time = datetime.fromisoformat(data['end_time']) if data['end_time'] else None
        record.experiment_logic_class_name = data['experiment_logic_class_name']
        record.experiment_logic_module = data['experiment_logic_module']
        record.experiment_description = data['experiment_description']  # Load description
        record.input_data_descriptors = input_descriptors
        record.output_data = output_data
        record.log_messages = data['log_messages']
        record.system_info = data['system_info']
        record.software_versions = data['software_versions']
        record.llm_usage = data['llm_usage']
        return record