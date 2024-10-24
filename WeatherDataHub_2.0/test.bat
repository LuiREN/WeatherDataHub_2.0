@echo off
echo ============================================
echo         WeatherDataHub Test Runner
echo ============================================
echo.

:: Проверка наличия виртуального окружения
if not exist "venv" (
    echo [ERROR] Виртуальное окружение не найдено
    echo Пожалуйста, сначала запустите install.bat
    pause
    exit /b 1
)

:: Активация виртуального окружения
echo [INFO] Активация виртуального окружения...
call venv\Scripts\activate.bat

:: Запуск тестов
echo [INFO] Запуск тестов...
python -m pytest tests/ -v --cov=.

:: Деактивация виртуального окружения при выходе
deactivate
pause
