# Testing Strategy for Scientific Data Simulator

This directory contains the unit tests for the `Scientific Data Simulator`.  We use the `pytest` testing framework to ensure the correctness and reliability of the code.

## Why Testing is Important

*   **Correctness:** Tests verify that the code behaves as expected and produces the correct results.
*   **Regression Prevention:** Tests help prevent new changes from breaking existing functionality.
*   **Maintainability:**  A good test suite makes it easier to refactor and maintain the codebase over time.
*   **Reproducibility:** Tests contribute to the overall reproducibility of the project by ensuring that the core simulation logic is reliable.
*   **Confidence:** Tests provide confidence that the simulator is producing valid results.

## Running Tests

To run all tests, navigate to the project root directory and execute:

```bash
pytest
```

This will automatically discover and run all test files (named `test_*.py` or `*_test.py`) within the `tests/` directory.

To run tests and generate log file:
```bash
pytest --log-file=tests/testlog/test_output.log
```

## Test Structure

The tests are organized into separate files, each focusing on a specific module or component of the simulator:

*   `test_engine.py`: Tests the core `SimulatorEngine` class (experiment execution, configuration loading, record saving/loading).
*   `test_experiment_record.py`: Tests the `ExperimentRecord` class (data serialization/deserialization).
*   `test_config.py`: Tests the configuration loading functions.
*   `test_utils.py`: Tests utility functions and classes (e.g., `DataDescriptor`, `DataType`).
*   `test_data_handler.py`: Tests data loading and saving functions (e.g., `load_csv`, `load_json`, `load_numpy`, `save_csv`).
*   `test_visualization.py`: Tests the plotting functions.
*   `test_<experiment_name>.py`: Tests for specific `ExperimentLogic` implementations (e.g., `test_example_experiment.py`, `test_predator_prey.py`).  These tests focus on the *scientific logic* of each experiment.

## Writing Tests

*   **Test-Driven Development (TDD):** Ideally, tests should be written *before* the code they are testing (Test-Driven Development). This helps clarify requirements and leads to better design.
*   **Unit Tests:** Each test function should focus on testing a *single* unit of code (e.g., a function or a method).
*   **Assertions:** Use `assert` statements to check that the actual results of your code match the expected results.
*   **Fixtures:** Use `pytest` fixtures (`@pytest.fixture`) to set up common test data or configurations.
*   **Edge Cases:** Test edge cases and boundary conditions to ensure that your code handles unexpected inputs gracefully.
*   **Error Handling:** Test that exceptions are raised correctly when expected (e.g., `with pytest.raises(ValueError): ...`).


