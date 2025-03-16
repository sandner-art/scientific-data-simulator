# tests/test_config.py
import pytest
from simulator.config import load_config, get_api_key
import os
import yaml

# Fixture to create temporary config files
@pytest.fixture
def temp_config_file(tmp_path):
    config_file = tmp_path / "config.yaml"
    def _create_config_file(content):
        with open(config_file, "w") as f:
            yaml.dump(content, f)
        return str(config_file)  # Return as string for load_config
    return _create_config_file

def test_load_config_valid(temp_config_file):
    """Test loading a valid YAML configuration file."""
    config_path = temp_config_file({"param1": 10, "param2": "value"})
    config = load_config(config_path)
    assert config["param1"] == 10
    assert config["param2"] == "value"

def test_load_config_missing_file():
    """Test loading a non-existent configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config.yaml")

def test_load_config_invalid_yaml(temp_config_file):
    """Test loading an invalid YAML configuration file."""
    config_path = temp_config_file("not yaml") # Pass string, to avoid error during creation
    with open(config_path, 'w') as f: # Overwrite content of the config.
        f.write("invalid_yaml_content: -: : -")
    with pytest.raises(yaml.YAMLError):
        load_config(config_path)


def test_load_config_environment_variable(temp_config_file, monkeypatch):
    """Test loading a config file with environment variable substitution."""
    config_path = temp_config_file({"param1": "$TEST_VAR", "param2": "value"})

    # Set a temporary environment variable using monkeypatch
    monkeypatch.setenv("TEST_VAR", "env_value")  # Use monkeypatch

    config = load_config(config_path)
    assert config["param1"] == "env_value"
    assert config["param2"] == "value"

def test_load_config_missing_environment_variable(temp_config_file):
    config_path = temp_config_file({"param_with_missing_env_var": "$MISSING_VAR"})
    with pytest.raises(ValueError) as excinfo:
      config = load_config(config_path)
    assert "MISSING_VAR" in str(excinfo.value)  # Check for variable name in error

def test_get_api_key_exists(monkeypatch):
    """Test retrieving an API key from environment variables."""
    monkeypatch.setenv("MY_API_KEY", "test_key_value")
    api_key = get_api_key("MY_API_KEY")
    assert api_key == "test_key_value"

def test_get_api_key_not_exists(monkeypatch):
    """Test retrieving a non-existent API key."""

    with pytest.raises(ValueError) as excinfo:
        get_api_key("NONEXISTENT_KEY")
    assert "NONEXISTENT_KEY" in str(excinfo.value)

def test_get_api_key_dotenv(tmp_path, monkeypatch):
    """Test get_api_key with a .env file."""
    dotenv_path = tmp_path / ".env"
    with open(dotenv_path, "w") as f:
        f.write("DOTENV_TEST_KEY=dotenv_test_value\n")

    # Unset the environment variable, if by any accident is set.
    monkeypatch.delenv("DOTENV_TEST_KEY", raising=False)
    api_key = get_api_key("DOTENV_TEST_KEY", dotenv_path=str(dotenv_path))
    assert api_key == "dotenv_test_value"