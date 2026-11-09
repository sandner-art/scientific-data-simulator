@echo off
REM Start the browser UI (installed with: scripts\install.cmd ui). Close this window to stop it.
setlocal
set "SDS_PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%SDS_PY%" (
  echo No .venv found. Install first: scripts\install.cmd ui
  pause
  exit /b 1
)
cd /d "%~dp0.."
"%SDS_PY%" -m simulator ui %*
set "SDS_EXIT=%ERRORLEVEL%"
if not "%SDS_EXIT%"=="0" pause
exit /b %SDS_EXIT%
