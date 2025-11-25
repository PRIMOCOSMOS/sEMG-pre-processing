@echo off
REM sEMG Preprocessing Toolkit - Windows Launcher
REM Double-click this file to start the application

echo ============================================================
echo   sEMG Preprocessing Toolkit v0.4.0
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

REM Run the launcher script
python run_app.py

pause
