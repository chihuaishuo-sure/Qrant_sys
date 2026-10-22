@echo off
setlocal
cd /d "%~dp0"
title Qrant scheduled task
if not exist ".venv\Scripts\python.exe" goto nosetup
chcp 65001 >nul
rem Every weekday at 16:30 (after A-share close and data update). Signals only, never places orders.
schtasks /Create /F /TN "Qrant_Daily" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:30 ^
  /TR "\"%~dp0.venv\Scripts\python.exe\" \"%~dp0auto_run.py\" auto"
if errorlevel 1 goto fail
echo.
echo [Qrant] Installed task "Qrant_Daily": weekdays 16:30. Reports go to reports\daily\
echo [Qrant] Run once now to test:  .venv\Scripts\python.exe auto_run.py daily
echo [Qrant] To remove: double-click 卸载定时任务.bat
pause
exit /b 0
:nosetup
echo [Qrant] Please double-click the launcher bat once first to install dependencies, then run this again.
pause
exit /b 1
:fail
echo [Qrant] Failed to create the scheduled task.
echo [Qrant] If it says "Access is denied": right-click this file and choose "Run as administrator".
pause
exit /b 1
