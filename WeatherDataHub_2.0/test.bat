@echo off
chcp 65001 > nul
echo ============================================
echo         WeatherDataHub Test Runner
echo ============================================
echo.

:: Check virtual environment
if not exist "venv" (
    echo [ERROR] Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Run tests
echo [INFO] Running tests...
python -m pytest tests/ -v --cov=.

:: Deactivate virtual environment on exit
deactivate
pause