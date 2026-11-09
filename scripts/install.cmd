@echo off
REM Install Scientific Data Simulator (Windows). Usage: scripts\install.cmd [core^|standard^|all^|dev]
REM Runs install.ps1 with the execution policy bypassed for this run only. Double-clicking it works too.
setlocal
set "SDS_PROFILE=%~1"
if "%SDS_PROFILE%"=="" set "SDS_PROFILE=standard"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" -Setup %SDS_PROFILE%
set "SDS_EXIT=%ERRORLEVEL%"
if not "%SDS_NOPAUSE%"=="1" pause
exit /b %SDS_EXIT%
