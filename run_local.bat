@echo off
REM Signora - one-command local setup ^& run (Windows)
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.11 -c "import sys" >nul 2>nul
    if errorlevel 1 goto nopy
    set PY=py -3.11
) else (
    python -c "import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>nul
    if errorlevel 1 goto nopy
    set PY=python
)

if not exist venv (
    echo Creating virtual environment...
    %PY% -m venv venv
    venv\Scripts\python -m pip install --upgrade pip
    echo Installing dependencies - first run takes a few minutes...
    venv\Scripts\pip install -r requirements-local.txt
    if errorlevel 1 ( echo Install failed. & pause & exit /b 1 )
)

venv\Scripts\python run_local.py
pause
exit /b 0

:nopy
echo Python 3.11 is required - TensorFlow 2.13 does not support 3.12 or newer.
echo Download it from https://www.python.org/downloads/release/python-3119/
echo and tick "Add python.exe to PATH" during install.
pause
exit /b 1
