@echo off
echo ============================================
echo         WeatherDataHub Installer
echo ============================================
echo.

:: Проверка Python
python --version > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python не установлен!
    echo Пожалуйста, установите Python 3.11 или выше с сайта python.org
    pause
    exit /b 1
)

:: Проверка версии Python
python -c "import sys; assert sys.version_info >= (3,11)" > nul 2>&1
if errorlevel 1 (
    echo [ERROR] Требуется Python 3.11 или выше
    echo Текущая версия:
    python --version
    pause
    exit /b 1
)

:: Создание виртуального окружения
echo [INFO] Создание виртуального окружения...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Ошибка при создании виртуального окружения
    pause
    exit /b 1
)

:: Активация виртуального окружения
echo [INFO] Активация виртуального окружения...
call venv\Scripts\activate.bat

:: Обновление pip
echo [INFO] Обновление pip...
python -m pip install --upgrade pip

:: Установка зависимостей
echo [INFO] Установка зависимостей...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Ошибка при установке зависимостей
    pause
    exit /b 1
)

echo.
echo ============================================
echo        Установка успешно завершена!
echo ============================================
echo Теперь вы можете запустить приложение через run.bat
pause
