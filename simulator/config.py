# simulator/config.py
import yaml
import os
from typing import Dict, Any
from dotenv import load_dotenv

def load_config(config_path: str, secrets_path: str = ".env") -> Dict[str, Any]:
    """
    Loads the experiment configuration from a YAML file and environment variables.

    Args:
        config_path: Path to the YAML configuration file.
        secrets_path: Path to the .env file (or a file with environment variables).

    Returns:
        A dictionary containing the merged configuration.
    """

    # Load environment variables from .env file (if it exists)
    load_dotenv(secrets_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Overlay environment variables (allows overriding config file values)
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("$"):
            env_var_name = value[1:]  # Remove the '$'
            env_var_value = os.getenv(env_var_name)
            if env_var_value is not None:
                config[key] = env_var_value
            else:
                # If it starts with '$' it must be provided by env.
                raise ValueError(
                    f"Environment variable '{env_var_name}' not found, "
                    f"but it is required by the configuration."
                )
    return config


def get_api_key(key_name: str, dotenv_path=".env") -> str:
    """
        Retrieves an API key from environment variables.
        Prioritizes .env files if present, then falls back to system environment.

    """
    # Attempt to load from .env file
    load_dotenv(dotenv_path)

    # Try to get the key from the environment
    api_key = os.getenv(key_name)

    if api_key is None:
        raise ValueError(
            f"API key '{key_name}' not found in environment variables. "
            f"Please set it using: export {key_name}=YOUR_API_KEY  or in .env file"
        )
    return api_key