@echo off
chcp 65001 > nul
echo ============================================
echo         WeatherDataHub Launcher
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

:: Run application
echo [INFO] Starting WeatherDataHub...
python main_window.py

:: Deactivate virtual environment on exit
deactivate