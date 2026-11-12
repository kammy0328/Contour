@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\python\python.exe"
if not exist "%PY%" ( echo DIAP python not found: %PY% & pause & exit /b 1 )
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo DIAP step training: one collected project at a time (GPU). Close DIAP first.
"%PY%" "%~dp0step_train.py" %*
echo.
pause
