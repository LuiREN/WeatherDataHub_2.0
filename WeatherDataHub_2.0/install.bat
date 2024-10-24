@echo off
echo ============================================
echo         WeatherDataHub Installer
echo ============================================
echo.

:: �������� Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python �� ����������!
    echo ����������, ���������� Python 3.11 ��� ���� � ����� python.org
    pause
    exit /b 1
)

:: �������� ������ Python
python -c "import sys; assert sys.version_info >= (3,11)" > nul 2>&1
if errorlevel 1 (
    echo [ERROR] ��������� Python 3.11 ��� ����
    echo ������� ������:
    python --version
    pause
    exit /b 1
)

:: �������� ������������ ���������
echo [INFO] �������� ������������ ���������...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] ������ ��� �������� ������������ ���������
    pause
    exit /b 1
)

:: ��������� ������������ ���������
echo [INFO] ��������� ������������ ���������...
call venv\Scripts\activate.bat

:: ���������� pip
echo [INFO] ���������� pip...
python -m pip install --upgrade pip

:: ��������� ������������
echo [INFO] ��������� ������������...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] ������ ��� ��������� ������������
    pause
    exit /b 1
)

echo.
echo ============================================
echo        ��������� ������� ���������!
echo ============================================
echo ������ �� ������ ��������� ���������� ����� run.bat
pause
