## Testing

The `Scientific Data Simulator` includes a comprehensive suite of unit tests. To run the tests, you need to have `pytest` installed.  If you followed the installation instructions, `pytest` should already be installed in your virtual environment.

From the project root directory, run:

```bash
pytest
```

or

```bash
pytest --log-file=tests/testlog/test_output.log
```

This command will automatically discover and run all tests in the `tests/` directory.



## **Key Points and Explanations:**

*   **`pytest`:**  We're using the `pytest` testing framework.  `pytest` is the *de facto* standard for testing in Python. It's powerful, flexible, and easy to use.
*   **Test File Structure:** Tests are organized into files named `test_<module_name>.py` within the `tests/` directory.
*   **Test Functions:**  Test functions *must* start with the prefix `test_`.  `pytest` automatically discovers and runs these functions.
*   **Assertions:**  Use `assert` statements to check that the actual results of your code match the expected results.  If an assertion fails, `pytest` will report the failure.
*   **Fixtures (`@pytest.fixture`):** Fixtures are functions that provide a fixed baseline for your tests. They are used to:
    *   Set up test data.
    *   Create instances of objects that are needed by multiple tests.
    *   Clean up resources after tests are finished.

    In `test_engine.py`, the `temp_test_dir` fixture creates a temporary directory for each test function.  This ensures that each test runs in an isolated environment and that any files created by the test are automatically cleaned up.  The `tmp_path` fixture (provided by `pytest`) is used to create the temporary directory.
* **`test_engine.py`:**
    * `DummyExperiment`: A very simple `ExperimentLogic` implementation used *only* for testing the engine. We don't want to test engine using other examples.
    * `test_run_experiment_valid_config`: Tests the core `run_experiment` method with a valid configuration. It checks if experiment ID is returned, record is saved, and data are saved to record.
    * `test_run_experiment_invalid_logic`: Checks that an appropriate exception is raised if an invalid `ExperimentLogic` class is specified.
    * `test_run_experiment_missing_config`: Checks that a `FileNotFoundError` is raised if the configuration file doesn't exist.
    * `test_validate_results_invalid_data`: Checks that validation works.
    *`test_load_experiment_record_nonexistent`: Checks loading non-existent file.
*   **`tests/test_example_experiment.py`:**
    *   `example_config`: A fixture that provides a sample configuration dictionary.
    *   `test_initialize`: Tests the `initialize` method.
    *   `test_run_step`: Tests the `run_step` method.
    *   `test_get_results`: Tests the `get_results` method.  It checks the structure and content of the returned data.
* **Running tests:** Now you can create other tests.

