@echo off
REM Run an example: run_example.bat ^<example_name^> [--quick] [--set name=value ...]
REM List the examples: run_example.bat (no arguments)
setlocal
set "SDS_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%SDS_PY%" (
  echo No .venv found. Install first: scripts\install.cmd
  if not "%SDS_NOPAUSE%"=="1" pause
  exit /b 1
)
if "%~1"=="" (
  echo Usage: run_example.bat ^<example_name^> [--quick] [--set name=value ...]
  echo.
  "%SDS_PY%" -m simulator examples list
  if not "%SDS_NOPAUSE%"=="1" pause
  exit /b 1
)
"%SDS_PY%" -m simulator examples run %*
set "SDS_EXIT=%ERRORLEVEL%"
if not "%SDS_NOPAUSE%"=="1" pause
exit /b %SDS_EXIT%
