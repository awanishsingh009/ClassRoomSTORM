@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo.
echo ClassRoomSTORM Studio V1.1 - Windows setup
echo =========================================
echo.

call :find_python
if not defined PYTHON_CMD (
    call :offer_python_install
    exit /b 1
)

echo Using Python command: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating local Python environment...
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo Failed to create the Python environment.
        goto :fail
    )
)

echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    goto :fail
)

echo Installing ClassRoomSTORM requirements...
if exist "wheels\*.whl" (
    echo Found a local wheels folder - installing offline.
    ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheels -r requirements.txt
) else (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 (
    echo Failed to install requirements.
    goto :fail
)

echo.
echo Setup completed.
echo You can now double-click launch_studio_windows.bat.
pause
exit /b 0

:fail
echo.
echo Setup did not finish.
echo This can happen on restricted networks, proxy networks, or machines
echo where Python cannot verify PyPI SSL certificates.
echo Removing the incomplete local environment so the launcher will not
echo accidentally use it.
rmdir /s /q ".venv" >nul 2>nul
echo.
echo See README_FIRST.txt for troubleshooting notes.
pause
exit /b 1

:find_python
rem Prefer stable python.org installs. Avoid Windows Store / Install Manager
rem preview defaults such as Python 3.14 unless the user explicitly creates
rem their own environment.
set "PYTHON_CMD="
py -3.11 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.11"
    exit /b 0
)
py -3.12 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.12"
    exit /b 0
)
py -3.13 -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3.13"
    exit /b 0
)
python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)
exit /b 0

:offer_python_install
echo No supported Python was found.
echo.
echo ClassRoomSTORM needs Python 3.11, 3.12, or 3.13.
echo If Windows opens the Microsoft Store when you type python, disable the
echo python.exe and python3.exe aliases in:
echo Settings ^> Apps ^> Advanced app settings ^> App execution aliases
echo.
echo Choose an option:
echo   1. Install Python 3.11 using winget
echo   2. Open the official Python download page
echo   3. Exit
echo.
choice /C 123 /N /M "Enter 1, 2, or 3: "
if errorlevel 3 exit /b 1
if errorlevel 2 (
    start "" "https://www.python.org/downloads/windows/"
    echo.
    echo Install Python 3.11, 3.12, or 3.13, then run setup_windows.bat again.
    pause
    exit /b 1
)
where winget >nul 2>nul
if errorlevel 1 (
    echo.
    echo winget was not found. Opening the official Python download page instead.
    start "" "https://www.python.org/downloads/windows/"
    echo Install Python 3.11, 3.12, or 3.13, then run setup_windows.bat again.
    pause
    exit /b 1
)
echo.
echo Starting Python 3.11 installer through winget...
winget install -e --id Python.Python.3.11
echo.
echo After installation finishes, close and reopen this window, then run
echo setup_windows.bat again.
pause
exit /b 1
