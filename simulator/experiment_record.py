# simulator/experiment_record.py
from typing import Optional, Dict, Any, List, Type
from datetime import datetime
import uuid
from .utils import DataDescriptor  # Import from utils
import numpy as np  # Import numpy
import pandas as pd

class ExperimentRecord:
    """
    Represents a complete record of a single experiment run.
    """

    def __init__(self, config: Dict[str, Any], experiment_logic_class: Optional[Type] = None):
        self.experiment_id: str = str(uuid.uuid4())  # Unique ID
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        self.config: Dict[str, Any] = config  # Store the configuration
        self.experiment_logic_class_name: str = "" # Provide defaults
        self.experiment_logic_module: str = "" # Provide defaults
        if experiment_logic_class: # Check if experiment logic class provided
            self.experiment_logic_class_name = experiment_logic_class.__name__
            self.experiment_logic_module = experiment_logic_class.__module__
        self.experiment_description: str = config.get("experiment_description", "")
        self.input_data_descriptors: Dict[str, DataDescriptor] = {}  # Descriptors used
        self.output_data: Dict[str, Dict[str, Any]] = {}  # Store results (as before)
        self.log_messages: List[str] = []  # Store log messages.
        self.system_info: Dict[str, Any] = {}  # add this.
        self.software_versions: Dict[str, str] = {}  # and this.
        self.llm_usage: Dict[str, Any] = {} # LLM usage.
        self.randomness: Dict[str, Any] = {}  # seed and derived streams (see simulator.rng)
        self.provenance: Dict[str, Any] = {}  # config hash, code hash, git commit, flags
        self.ground_truth: Dict[str, Any] = {}  # parameters, law shift, observation model, field roles
        self.bundle: Dict[str, Any] = {}  # manifest of the multimodal run bundle (see simulator.bundle)

    def add_input_data_descriptor(self, name: str, descriptor: DataDescriptor):
        self.input_data_descriptors[name] = descriptor

    def add_output_data(self, name: str, data: Any, descriptor: DataDescriptor):
        self.output_data[name] = {"data": data, "descriptor": descriptor}

    def add_log_message(self, message: str):
        self.log_messages.append(message)

    def set_end_time(self):
        self.end_time = datetime.now()

    def set_system_info(self, system_info: Dict[str, Any]):
        self.system_info = system_info

    def set_software_versions(self, software_versions: Dict[str, str]):
        self.software_versions = software_versions

    def set_llm_usage(self, llm_usage: Dict[str, Any]):
        self.llm_usage = llm_usage

    @property
    def seed(self) -> Optional[int]:
        return self.randomness.get("seed")

    def to_dict(self, inline_data: bool = True) -> Dict[str, Any]:
        """Serialize the record to a JSON-friendly dictionary.

        With ``inline_data=False`` output values are omitted (``"data": None``);
        :func:`simulator.persistence.save_experiment_record` then stores them in
        data files and adds a ``storage`` entry per output.
        """
        # Convert everything to JSON-serializable types
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "config": self.config,
            "experiment_logic_class_name": self.experiment_logic_class_name,
            "experiment_logic_module": self.experiment_logic_module,
            "experiment_description": self.experiment_description,
            "input_data_descriptors": {
                name: descriptor.to_dict()  # Use a to_dict method
                for name, descriptor in self.input_data_descriptors.items()
            },
            "output_data": {
                name: {
                    "data": _convert_to_serializable(data_info["data"]) if inline_data else None,
                    "descriptor": data_info["descriptor"].to_dict()  # Use a to_dict method
                } for name, data_info in self.output_data.items()
            },
            "log_messages": self.log_messages,
            "system_info": self.system_info,
            "software_versions": self.software_versions,
            'llm_usage': self.llm_usage,
            'randomness': self.randomness,
            'provenance': self.provenance,
            'ground_truth': self.ground_truth,
            'bundle': self.bundle,
        }


def _convert_to_serializable(data):
    """Recursively convert numpy/pandas objects to JSON-serializable Python types."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient='records')
    if isinstance(data, pd.Series):
        return data.tolist()
    if isinstance(data, np.generic):  # numpy scalars (np.int64, np.float64, np.bool_)
        return data.item()
    if isinstance(data, dict):
        return {k: _convert_to_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_convert_to_serializable(item) for item in data]
    return data
