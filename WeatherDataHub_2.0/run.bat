@echo off
echo ============================================
echo         WeatherDataHub Launcher
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

:: Запуск приложения
echo [INFO] Запуск WeatherDataHub...
python main_window.py

:: Деактивация виртуального окружения при выходе
deactivate
