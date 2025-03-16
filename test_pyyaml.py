# test_pyyaml.py
import importlib.metadata
import sys

print(f"Python executable: {sys.executable}")

try:
    version = importlib.metadata.version('pyyaml')
    print(f"PyYAML version: {version}")
except importlib.metadata.PackageNotFoundError:
    print("PyYAML is not installed.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

try:
    import yaml
    print(f"yaml version is {yaml.__version__}")
except Exception as e:
  print(f"Error during import: {e}")