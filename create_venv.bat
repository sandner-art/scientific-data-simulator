@echo off
REM Kept for compatibility: use scripts\install.cmd [core^|standard^|all^|dev].
call "%~dp0scripts\install.cmd" dev
exit /b %ERRORLEVEL%
