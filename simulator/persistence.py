# simulator/persistence.py
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .experiment_record import ExperimentRecord, _convert_to_serializable
from .storage import read_output, write_outputs
from .utils import DataDescriptor, DataType


def experiment_dir_for(record: ExperimentRecord, output_dir: str) -> str:
    """Directory in which a record (and its data files) is stored."""
    timestamp = record.start_time.strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(output_dir, f"{timestamp}_{record.experiment_id}")


def _json_default(obj: Any) -> Any:
    converted = _convert_to_serializable(obj)
    if converted is obj:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    return converted


def save_experiment_record(record: ExperimentRecord, output_dir: str,
                           storage: Optional[Dict[str, Any]] = None) -> str:
    """Saves the record to ``<output_dir>/<timestamp>_<id>/``.

    Arrays and DataFrames go to checksummed data files (see
    :mod:`simulator.storage`; ``storage`` defaults to ``record.config['storage']``),
    the record to ``experiment_record.json`` and, when present, the ground truth
    to ``ground_truth.json``. Returns the experiment directory.
    """
    experiment_dir = experiment_dir_for(record, output_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    if storage is None:
        storage = (record.config or {}).get("storage")
    try:
        manifest = write_outputs(record.output_data, experiment_dir, storage)
    except Exception as e:
        # Leave a metadata-only record next to any partial files, then re-raise.
        record.add_log_message(f"Writing data files failed: {type(e).__name__}: {e}")
        with open(os.path.join(experiment_dir, "experiment_record.json"), "w", encoding="utf-8") as f:
            json.dump(record.to_dict(inline_data=False), f, indent=4, default=_json_default)
        raise
    record_dict = record.to_dict(inline_data=False)
    for name, data_info in record.output_data.items():
        entry = record_dict["output_data"][name]
        if manifest.get(name) is None:
            entry["data"] = _convert_to_serializable(data_info["data"])
        else:
            entry["storage"] = manifest[name]

    record_path = os.path.join(experiment_dir, "experiment_record.json")
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record_dict, f, indent=4, default=_json_default)
    if record.ground_truth:
        # Exported separately so observations can be shared without the answers.
        with open(os.path.join(experiment_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
            json.dump(record.ground_truth, f, indent=2, default=_json_default)
    return experiment_dir


def update_record_json(experiment_dir: str, record: ExperimentRecord, *keys: str) -> None:
    """Rewrite selected top-level keys of an existing experiment_record.json
    (data files are left untouched)."""
    path = os.path.join(experiment_dir, "experiment_record.json")
    with open(path, "r", encoding="utf-8") as f:
        stored = json.load(f)
    current = record.to_dict(inline_data=False)
    for key in keys:
        stored[key] = current[key]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stored, f, indent=4, default=_json_default)


def restore_data(data: Any, descriptor: DataDescriptor) -> Any:
    """Rebuild the in-memory type declared by ``descriptor`` from its JSON form."""
    if data is None:
        return None
    if descriptor.data_type == DataType.NDARRAY:
        return np.asarray(data)
    if descriptor.data_type == DataType.DATAFRAME:
        return pd.DataFrame.from_records(data) if isinstance(data, list) else pd.DataFrame(data)
    if descriptor.data_type == DataType.FLOAT:
        return float(data)
    if descriptor.data_type == DataType.INT:
        return int(data)
    return data


def find_experiment_dir(output_dir: str, experiment_id: str) -> str:
    for item in sorted(os.listdir(output_dir)):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.endswith(experiment_id):
            return item_path
    raise FileNotFoundError(f"Experiment directory with ID '{experiment_id}' not found in '{output_dir}'.")


def load_experiment_record(output_dir: str, experiment_id: str, verify: bool = True,
                           mmap: bool = False) -> ExperimentRecord:
    """Loads an experiment record from disk, restoring arrays and DataFrames.

    Args:
        verify: Check every data file against its recorded SHA-256
            (raises :class:`simulator.storage.ChecksumError` on mismatch).
        mmap: Memory-map ``.npy`` arrays instead of reading them (lazy loading).
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist.")
    experiment_dir = find_experiment_dir(output_dir, experiment_id)
    record_path = os.path.join(experiment_dir, "experiment_record.json")

    with open(record_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    input_descriptors = {
        name: DataDescriptor.from_dict(desc_data)
        for name, desc_data in data.get("input_data_descriptors", {}).items()
    }
    output_data = {}
    for name, data_info in data.get("output_data", {}).items():
        descriptor = DataDescriptor.from_dict(data_info["descriptor"])
        if data_info.get("storage"):
            value = read_output(experiment_dir, data_info["storage"], descriptor, verify=verify, mmap=mmap)
        else:
            value = restore_data(data_info.get("data"), descriptor)
        output_data[name] = {"data": value, "descriptor": descriptor}

    record = ExperimentRecord(data['config'], None)
    record.experiment_id = data['experiment_id']
    record.start_time = datetime.fromisoformat(data['start_time'])
    record.end_time = datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
    record.experiment_logic_class_name = data.get('experiment_logic_class_name', "")
    record.experiment_logic_module = data.get('experiment_logic_module', "")
    record.experiment_description = data.get('experiment_description', "")
    record.input_data_descriptors = input_descriptors
    record.output_data = output_data
    record.log_messages = data.get('log_messages', [])
    record.system_info = data.get('system_info', {})
    record.software_versions = data.get('software_versions', {})
    record.llm_usage = data.get('llm_usage', {})
    record.randomness = data.get('randomness', {})
    record.provenance = data.get('provenance', {})
    record.ground_truth = data.get('ground_truth', {})
    record.bundle = data.get('bundle', {})
    return record
