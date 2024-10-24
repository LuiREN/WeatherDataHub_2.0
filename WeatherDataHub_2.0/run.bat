@echo off
echo ============================================
echo         WeatherDataHub Launcher
echo ============================================
echo.

:: �������� ������� ������������ ���������
if not exist "venv" (
    echo [ERROR] ����������� ��������� �� �������
    echo ����������, ������� ��������� install.bat
    pause
    exit /b 1
)

:: ��������� ������������ ���������
echo [INFO] ��������� ������������ ���������...
call venv\Scripts\activate.bat

:: ������ ����������
echo [INFO] ������ WeatherDataHub...
python main_window.py

:: ����������� ������������ ��������� ��� ������
deactivate
