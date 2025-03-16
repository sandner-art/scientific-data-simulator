@echo off
REM run_example.bat

REM ---------------------------------------------------------------------
REM Scientific Data Simulator - Example Runner
REM ---------------------------------------------------------------------
REM This script runs an example experiment from the Scientific Data
REM Simulator project.  It activates the virtual environment and then
REM executes the specified example script using `python -m`.
REM
REM Usage:
REM    run_example.bat <example_name>
REM
REM    <example_name> : The name of the example to run (without the .py extension).
REM                     For example: example_1, example_2, etc.
REM
REM Example:
REM    run_example.bat example_3
REM ---------------------------------------------------------------------

REM Check if an example name was provided
IF "%~1"=="" (
    echo Error: You must provide an example name.
    echo Usage: run_example.bat ^<example_name^>
    pause
    exit /b 1
)

REM Set the example name
SET EXAMPLE_NAME=%1

REM Activate the virtual environment (adjust the path if necessary)
echo Activating virtual environment...
CALL .venv\Scripts\activate.bat

REM Check if the environment was activated
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to activate virtual environment.
    echo Make sure you have created the environment using: python -m venv .venv
    pause
    exit /b 1
)

REM Run the specified example
echo Running example: %EXAMPLE_NAME%...
python -m examples.%EXAMPLE_NAME%.run_experiment

REM Check for errors during execution
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Experiment execution failed.
    pause
    exit /b 1
)

echo Experiment completed successfully.  Check the experiments_output directory for results.
pause