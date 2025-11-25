@echo off
REM sEMG Preprocessing Toolkit - Windows Launcher
REM Double-click this file to start the application

echo ============================================================
echo   sEMG Preprocessing Toolkit v0.4.0
echo ============================================================
echo.

REM Try to find Python in different locations
set PYTHON_CMD=

REM Method 1: Check if python is in PATH
where python >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
    goto :found_python
)

REM Method 2: Check if python3 is in PATH
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    goto :found_python
)

REM Method 3: Check common installation paths
if exist "C:\Python312\python.exe" (
    set PYTHON_CMD=C:\Python312\python.exe
    goto :found_python
)
if exist "C:\Python311\python.exe" (
    set PYTHON_CMD=C:\Python311\python.exe
    goto :found_python
)
if exist "C:\Python310\python.exe" (
    set PYTHON_CMD=C:\Python310\python.exe
    goto :found_python
)
if exist "C:\Python39\python.exe" (
    set PYTHON_CMD=C:\Python39\python.exe
    goto :found_python
)
if exist "C:\Python38\python.exe" (
    set PYTHON_CMD=C:\Python38\python.exe
    goto :found_python
)

REM Method 4: Check AppData local installations
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    goto :found_python
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
    goto :found_python
)

REM Method 5: Try py launcher (Windows Python Launcher)
where py >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py
    goto :found_python
)

REM No Python found
echo ERROR: Python is not installed or not found
echo.
echo Please install Python 3.8+ from https://www.python.org/
echo Make sure to check "Add Python to PATH" during installation
echo.
pause
exit /b 1

:found_python
echo Found Python: %PYTHON_CMD%
echo.

REM Run the launcher script
"%PYTHON_CMD%" run_app.py

if %errorlevel% neq 0 (
    echo.
    echo Application exited with an error.
)

pause
