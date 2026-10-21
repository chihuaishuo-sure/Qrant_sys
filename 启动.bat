@echo off
setlocal
cd /d "%~dp0"
title Qrant

where python >nul 2>nul
if errorlevel 1 goto nopython

if exist ".venv\Scripts\python.exe" goto venvok
echo [Qrant] First run: creating virtual environment .venv ...
python -m venv .venv
if errorlevel 1 goto fail
:venvok

if exist ".venv\deps_ok.txt" goto run
echo [Qrant] Installing dependencies via Tsinghua mirror, about 2-5 minutes ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto retry
goto depsok
:retry
echo [Qrant] Mirror failed, retrying with default PyPI ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
:depsok
echo ok> ".venv\deps_ok.txt"

:run
echo [Qrant] Starting... your browser will open http://localhost:8501
echo [Qrant] Close this window to stop the system.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501
goto end

:nopython
echo [Qrant] Python was not found.
echo Please install Python 3.10 or newer from https://www.python.org/downloads/
echo IMPORTANT: tick "Add python.exe to PATH" during installation, then run this file again.
pause
exit /b 1

:fail
echo [Qrant] Setup failed. Check your network, then delete the .venv folder and try again.
pause
exit /b 1

:end
pause
