# simulator/utils.py
from typing import Optional, Union, Tuple, Dict, Any
from enum import Enum
import numpy as np # For checking array type

class DataType(str, Enum):
    FLOAT = "float"
    INT = "int"
    STRING = "str"
    LIST = "list"
    NDARRAY = "np.ndarray"
    DATAFRAME = "pd.DataFrame"
    # Add other common types as needed

# simulator/utils.py
class DataDescriptor:
    def __init__(self,
                 name: str,
                 data_type: DataType,
                 shape: Optional[Tuple[int, ...]] = None,
                 units: Optional[str] = None,
                 group: str = "default",
                 plot_type: Optional[str] = None,
                 x_axis: Optional[str] = None):  # Add x_axis
        self.name = name
        self.data_type = data_type
        self.shape = shape
        self.units = units
        self.group = group
        self.plot_type = plot_type
        self.x_axis = x_axis  # Add x_axis

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'data_type': str(self.data_type),
            'shape': self.shape,
            'units': self.units,
            'group': self.group,
            'plot_type': self.plot_type,
            'x_axis': self.x_axis  # Add x_axis
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a configuration file (placeholder for now).
    """
    # Implementation will be in config.py, but we need this here for now.
    raise NotImplementedError("This function will be implemented in config.py")