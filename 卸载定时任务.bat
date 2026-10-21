@echo off
chcp 65001 >nul
schtasks /Delete /F /TN "Qrant_Daily"
pause
