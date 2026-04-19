@echo off
setlocal
cd /d "%~dp0"
PowerShell -ExecutionPolicy Bypass -File "%~dp0start_poc1_readonly.ps1" -Port 8506

endlocal