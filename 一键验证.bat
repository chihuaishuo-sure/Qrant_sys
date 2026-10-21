@echo off
setlocal
cd /d "%~dp0"
title Qrant A-share validation
if not exist ".venv\Scripts\python.exe" goto nosetup
chcp 65001 >nul
".venv\Scripts\python.exe" validate_ashare.py %*
echo.
echo [Qrant] Done. Report is in the reports folder.
start "" "%~dp0reports"
pause
exit /b 0
:nosetup
echo [Qrant] Please double-click the launcher bat once first to install dependencies, then run this again.
pause
exit /b 1
