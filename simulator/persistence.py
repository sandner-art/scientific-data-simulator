# simulator/persistence.py
import os
import json
from .experiment_record import ExperimentRecord
from .utils import DataDescriptor  # Import DataDescriptor
from typing import Dict, Any
from datetime import datetime

def save_experiment_record(record: ExperimentRecord, output_dir: str):
    """Saves the ExperimentRecord to a JSON file."""
    timestamp = record.start_time.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join(output_dir, f"{timestamp}_{record.experiment_id}")
    os.makedirs(experiment_dir, exist_ok=True)
    record_path = os.path.join(experiment_dir, "experiment_record.json")
    with open(record_path, "w") as f:
        json.dump(record.to_dict(), f, indent=4)
    return experiment_dir # Return the full path

def load_experiment_record(output_dir: str, experiment_id: str) -> ExperimentRecord:
    """Loads an experiment record from disk."""
    experiment_dir = None
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and experiment_id in item:
            experiment_dir = item_path
            break
    if experiment_dir is None:
        raise FileNotFoundError(f"Experiment directory with ID '{experiment_id}' not found in '{output_dir}'.")
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