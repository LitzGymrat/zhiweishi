@echo off
setlocal
cd /d "%~dp0"
PowerShell -ExecutionPolicy Bypass -File "%~dp0start_poc1_public_tunnel.ps1" -Port 8506
set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
	echo.
	echo 脚本已退出，退出码：%EXIT_CODE%
	echo 如果窗口是一闪而过，请直接在这个窗口里看报错信息。
	pause
)
endlocal