# WeatherDataHub 2.0

Приложение для сбора, анализа и визуализации метеорологических данных с возможностью веб-скрапинга, предобработки и аналитики.

## 📊 Основные возможности

- Сбор метеоданных через веб-скрапинг
- Предобработка и очистка данных
- Визуализация температурных трендов
- Анализ погодных данных по периодам
- Создание аннотаций к данным
- Разделение данных по различным критериям

## 🛠 Технологии

### GUI и визуализация
- PyQt6 - создание графического интерфейса
- matplotlib - построение графиков
- seaborn - улучшенная визуализация данных

### Обработка данных
- pandas - анализ и манипуляция данными
- numpy - математические операции

### Веб-скрапинг
- requests - HTTP запросы
- beautifulsoup4 - парсинг HTML

### Тестирование
- pytest - модульное тестирование
- pytest-qt - тестирование GUI
- pytest-cov - анализ покрытия кода

## ⚙️ Установка

1. Требования:
   - Python 3.11 или выше
   - Windows 10/11

2. Установка:
   ```bash
   # Запустите install.bat
   # Дождитесь завершения установки
   ```

3. Запуск:
   ```bash
   # Используйте run.bat
   ```

## 📦 Структура проекта

```
WeatherDataHub_2.0/
├── main_window.py         # Главное окно приложения
├── data_analysis.py       # Модуль анализа данных
├── scraper.py            # Веб-скрапинг
├── split_csv.py          # Разделение данных
├── data_preprocessing.py  # Предобработка данных
├── annotation.py         # Создание аннотаций
├── optimized_table.py    # Оптимизированная таблица
├── date_widget.py        # Виджет работы с датами
├── tests/               # Модульные тесты
└── requirements.txt     # Зависимости
```

## 📝 Использование

### Сбор данных
1. Нажмите "Создать новый датасет"
2. Укажите период сбора данных
3. Дождитесь завершения сбора

### Анализ данных
1. Загрузите файл с данными
2. Выберите вкладку "Анализ данных"
3. Используйте доступные инструменты анализа:
   - Построение графиков
   - Фильтрация данных
   - Статистический анализ

### Предобработка
1. Загрузите файл
2. Нажмите "Предобработка данных"
3. Сохраните результат

## ⚠️ Ограничения

- Поддерживаются только CSV файлы
- Даты должны быть в формате YYYY-MM-DD
- Первый столбец должен содержать даты
- Необходимо подключение к интернету для сбора данных
- Скрапинг ограничен определенным сайтом

## 🔍 Тестирование

```bash
# Запустите test.bat для выполнения всех тестов
# Будет показано покрытие кода
```
