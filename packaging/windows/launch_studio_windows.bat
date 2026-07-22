@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYEXE=.venv\Scripts\python.exe"
) else (
    echo Local environment was not found. Trying system Python.
    set "PYEXE=python"
)

"%PYEXE%" -c "import numpy, cv2, matplotlib, PIL, PySide6" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Required Python packages are not installed.
    echo Please run setup_windows.bat first, then try again.
    pause
    exit /b 1
)

"%PYEXE%" apps\ClassRoomSTORM_Studio.py
if errorlevel 1 (
    echo.
    echo ClassRoomSTORM did not launch successfully.
    echo Try running setup_windows.bat first.
    pause
)
