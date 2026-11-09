@echo off
REM Update Scientific Data Simulator (Windows). Usage: scripts\update.cmd [core^|standard^|all^|dev]
REM Runs update.ps1 with the execution policy bypassed for this run only.
setlocal EnableDelayedExpansion
(
  if "%~1"=="" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1"
  ) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" -Setup %~1
  )
  set "SDS_EXIT=!ERRORLEVEL!"
  if not "%SDS_NOPAUSE%"=="1" pause
  exit /b !SDS_EXIT!
)
