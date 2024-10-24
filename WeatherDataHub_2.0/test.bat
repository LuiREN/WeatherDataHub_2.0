@echo off
echo ============================================
echo         WeatherDataHub Test Runner
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

:: ������ ������
echo [INFO] ������ ������...
python -m pytest tests/ -v --cov=.

:: ����������� ������������ ��������� ��� ������
deactivate
pause
