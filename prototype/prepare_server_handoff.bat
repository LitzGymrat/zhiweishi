@echo off
setlocal
cd /d "%~dp0"
PowerShell -ExecutionPolicy Bypass -File "%~dp0prepare_server_handoff.ps1"
set EXIT_CODE=%ERRORLEVEL%
echo.
if "%EXIT_CODE%"=="0" (
    echo 打包完成，请看上面输出的目录和压缩包路径。
) else (
    echo 打包失败，退出码：%EXIT_CODE%
)
pause
endlocal & exit /b %EXIT_CODE%