@echo off
setlocal
cd /d "%~dp0"
PowerShell -ExecutionPolicy Bypass -File "%~dp0start_full_demo.ps1" -Port 8507

endlocal