import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
from typing import Optional, Dict, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QComboBox, QMessageBox, QDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout,
    QFrame
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from optimized_table import OptimizedTableWidget

class StatsDialog(QDialog):
    """Диалоговое окно для отображения статистической информации."""
    def __init__(self, stats_data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Статистическая информация")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.init_ui(stats_data)

    def init_ui(self, stats_data: pd.DataFrame):
        """Инициализация интерфейса диалогового окна."""
        layout = QVBoxLayout()
        
        # Создаем таблицу для отображения статистики
        table = QTableWidget()
        table.setRowCount(len(stats_data.index))
        table.setColumnCount(len(stats_data.columns))
        
        # Устанавливаем заголовки
        table.setHorizontalHeaderLabels(stats_data.columns)
        table.setVerticalHeaderLabels(stats_data.index)
        
        # Заполняем таблицу данными
        for i in range(len(stats_data.index)):
            for j in range(len(stats_data.columns)):
                value = stats_data.iloc[i, j]
                if isinstance(value, float):
                    value = f"{value:.2f}"
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(i, j, item)
        
        # Растягиваем колонки по содержимому
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        layout.addWidget(table)
        self.setLayout(layout)

class DateFilterDialog(QDialog):
    """Диалоговое окно для отображения отфильтрованных по дате данных."""
    def __init__(self, filtered_data: pd.DataFrame, start_date: str, end_date: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Данные за период {start_date} - {end_date}")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self.init_ui(filtered_data)

    def init_ui(self, filtered_data: pd.DataFrame):
        layout = QVBoxLayout()
        
        # Создаем и настраиваем таблицу
        table = QTableWidget()
        table.setRowCount(len(filtered_data))
        table.setColumnCount(len(filtered_data.columns))
        
        # Устанавливаем заголовки
        table.setHorizontalHeaderLabels(filtered_data.columns)
        
        # Заполняем таблицу данными
        for i in range(len(filtered_data)):
            for j in range(len(filtered_data.columns)):
                value = filtered_data.iloc[i, j]
                if isinstance(value, (pd.Timestamp, datetime)):
                    value = value.strftime('%Y-%m-%d')
                item = QTableWidgetItem(str(value))
                table.setItem(i, j, item)
        
        # Настраиваем размеры колонок
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        
        layout.addWidget(table)
        self.setLayout(layout)

class DataAnalysisTab(QWidget):
    """Вкладка анализа данных погоды."""

    def __init__(self, parent: Optional[QWidget] = None):
        """Инициализация вкладки анализа данных."""
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.current_file: Optional[str] = None
        self.init_ui()

    def init_ui(self):
        """Инициализация пользовательского интерфейса."""
        main_layout = QHBoxLayout()

        # Левая панель с кнопками
        left_panel = self.create_left_panel()
        left_panel.setFixedWidth(300)  # Фиксированная ширина левой панели
        main_layout.addWidget(left_panel)

        # Правая панель с результатами
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel)

        # Устанавливаем соотношение размеров панелей (20:80)
        main_layout.setStretch(0, 2)  # Левая панель
        main_layout.setStretch(1, 8)  # Правая панель

        self.setLayout(main_layout)

    def create_left_panel(self) -> QWidget:
        """Создает левую панель с элементами управления."""
        left_panel = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Группа обработки данных
        processing_group = QGroupBox("Обработка данных")
        processing_layout = QVBoxLayout()

        # Кнопка комплексной обработки данных
        self.process_data_btn = QPushButton("Комплексная обработка базы")
        self.process_data_btn.clicked.connect(self.process_data)
        self.process_data_btn.setEnabled(False)
        processing_layout.addWidget(self.process_data_btn)

        # Кнопка для показа статистики
        self.show_stats_btn = QPushButton("Показать статистику температуры")
        self.show_stats_btn.clicked.connect(self.show_temperature_stats)
        self.show_stats_btn.setEnabled(False)
        processing_layout.addWidget(self.show_stats_btn)

        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)

        # Группа фильтрации
        filter_group = QGroupBox("Фильтрация данных")
        filter_layout = QVBoxLayout()

        # Фильтр по температуре
        temp_filter_label = QLabel("Минимальная температура (°C):")
        self.temp_filter = QLineEdit()
        filter_layout.addWidget(temp_filter_label)
        filter_layout.addWidget(self.temp_filter)

        # Фильтры по датам
        date_label = QLabel("Период:")
        filter_layout.addWidget(date_label)
    
        self.start_date = QLineEdit()
        self.start_date.setPlaceholderText("YYYY-MM-DD")
        filter_layout.addWidget(self.start_date)
    
        self.end_date = QLineEdit()
        self.end_date.setPlaceholderText("YYYY-MM-DD")
        filter_layout.addWidget(self.end_date)

        # Кнопки фильтрации
        self.apply_filter_btn = QPushButton("Применить фильтры")
        self.apply_filter_btn.clicked.connect(self.apply_filters)
        self.apply_filter_btn.setEnabled(False)
        filter_layout.addWidget(self.apply_filter_btn)

        self.show_date_filter_btn = QPushButton("Показать фильтр по датам")
        self.show_date_filter_btn.clicked.connect(self.show_date_filtered_data)
        self.show_date_filter_btn.setEnabled(False)
        filter_layout.addWidget(self.show_date_filter_btn)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Группа анализа температуры
        temp_analysis_group = QGroupBox("Анализ температуры")
        temp_analysis_layout = QVBoxLayout()

        # Кнопка для группировки по месяцам
        self.group_by_month_btn = QPushButton("Средняя температура по месяцам")
        self.group_by_month_btn.clicked.connect(self.show_monthly_averages)
        self.group_by_month_btn.setEnabled(False)
        self.group_by_month_btn.setMinimumHeight(40)
        temp_analysis_layout.addWidget(self.group_by_month_btn)

        # Кнопка для графика температуры за весь период
        self.plot_temp_btn = QPushButton("График температуры за весь период")
        self.plot_temp_btn.clicked.connect(self.plot_temperature_changes)
        self.plot_temp_btn.setEnabled(False)
        self.plot_temp_btn.setMinimumHeight(40)
        temp_analysis_layout.addWidget(self.plot_temp_btn)

        # Элементы для графика за конкретный месяц
        month_year_layout = QGridLayout()
        self.month_combo = QComboBox()
        self.month_combo.addItems([str(i) for i in range(1, 13)])
        self.year_combo = QComboBox()
    
        month_year_layout.addWidget(QLabel("Месяц:"), 0, 0)
        month_year_layout.addWidget(self.month_combo, 0, 1)
        month_year_layout.addWidget(QLabel("Год:"), 0, 2)
        month_year_layout.addWidget(self.year_combo, 0, 3)
    
        temp_analysis_layout.addLayout(month_year_layout)

        self.plot_month_temp_btn = QPushButton("График температуры за месяц")
        self.plot_month_temp_btn.clicked.connect(self.plot_monthly_temperature)
        self.plot_month_temp_btn.setEnabled(False)
        self.plot_month_temp_btn.setMinimumHeight(40)
        temp_analysis_layout.addWidget(self.plot_month_temp_btn)

        temp_analysis_group.setLayout(temp_analysis_layout)
        layout.addWidget(temp_analysis_group)

        # Добавляем растягивающийся спейсер в конец
        layout.addStretch()

        left_panel.setLayout(layout)
        left_panel.setFixedWidth(300)
        return left_panel

    def create_right_panel(self) -> QWidget:
        """Создает правую панель для отображения результатов."""
        right_panel = QWidget()
        layout = QVBoxLayout()

        # Информационная метка
        self.info_label = QLabel("Загрузите данные для начала анализа")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        # Таблица для отображения данных
        self.data_preview = OptimizedTableWidget()
        layout.addWidget(self.data_preview)

        right_panel.setLayout(layout)
        return right_panel

    def translate_columns(self) -> Dict[str, str]:
        """Словарь для перевода названий колонок."""
        return {
            'дата': 'date',
            'температура_(день)': 'temperature_day',
            'давление_(день)': 'pressure_day',
            'температура_(вечер)': 'temperature_evening',
            'давление_(вечер)': 'pressure_evening',
            'облачность_(день)-ясно': 'cloudiness_day_clear',
            'облачность_(день)-малооблачно': 'cloudiness_day_partly_cloudy',
            'облачность_(день)-переменная_облачность': 'cloudiness_day_variable',
            'облачность_(день)-пасмурно': 'cloudiness_day_overcast',
            'облачность_(вечер)-ясно': 'cloudiness_evening_clear',
            'облачность_(вечер)-малооблачно': 'cloudiness_evening_partly_cloudy',
            'облачность_(вечер)-переменная_облачность': 'cloudiness_evening_variable',
            'облачность_(вечер)-пасмурно': 'cloudiness_evening_overcast',
            'ветер_(день)_(м/с)': 'wind_speed_day',
            'ветер_(день)-с': 'wind_direction_day_n',
            'ветер_(день)-св': 'wind_direction_day_ne',
            'ветер_(день)-в': 'wind_direction_day_e',
            'ветер_(день)-юв': 'wind_direction_day_se',
            'ветер_(день)-ю': 'wind_direction_day_s',
            'ветер_(день)-юз': 'wind_direction_day_sw',
            'ветер_(день)-з': 'wind_direction_day_w',
            'ветер_(день)-сз': 'wind_direction_day_nw',
            'ветер_(вечер)_(м/с)': 'wind_speed_evening',
            'ветер_(вечер)-с': 'wind_direction_evening_n',
            'ветер_(вечер)-св': 'wind_direction_evening_ne',
            'ветер_(вечер)-в': 'wind_direction_evening_e',
            'ветер_(вечер)-юв': 'wind_direction_evening_se',
            'ветер_(вечер)-ю': 'wind_direction_evening_s',
            'ветер_(вечер)-юз': 'wind_direction_evening_sw',
            'ветер_(вечер)-з': 'wind_direction_evening_w',
            'ветер_(вечер)-сз': 'wind_direction_evening_nw'
        }

    def process_data(self):
        """Комплексная обработка данных."""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Нет загруженных данных")
            return

        # Создаем копию DataFrame
        processed_df = self.df.copy()

        # 1. Переименование колонок
        processed_df.columns = [col.lower().replace(' ', '_') for col in processed_df.columns]
        translation_dict = self.translate_columns()
        processed_df.rename(columns=translation_dict, inplace=True)

        # Преобразуем столбец с датой в datetime и удаляем время
        processed_df['date'] = pd.to_datetime(processed_df['date']).dt.date

        # 2. Проверка на невалидные значения
        null_stats = processed_df.isnull().sum()
        nan_means = processed_df.isna().mean() * 100  # Процент NaN значений

        # Обработка невалидных значений
        # Для числовых колонок заменяем на среднее значение
        numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            mean_value = processed_df[col].mean()
            processed_df[col] = processed_df[col].fillna(mean_value)

        # Для текстовых колонок заменяем на "Нет данных"
        text_columns = processed_df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col != 'date':  # Исключаем столбец с датой
                processed_df[col] = processed_df[col].fillna("Нет данных")

        # 3. Добавление температуры по Фаренгейту
        temp_cols = [col for col in processed_df.columns if 'temperature' in col]
        for col in temp_cols:
            fahrenheit_col = f"{col}_fahrenheit"
            processed_df[fahrenheit_col] = processed_df[col] * 9/5 + 32

        # Создаем директорию для анализа, если её нет
        analysis_dir = 'analysis_data'
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)

        # Формируем имя файла для сохранения
        if self.current_file:
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            output_file = os.path.join(analysis_dir, f"{base_name}_processed.csv")
        else:
            output_file = os.path.join(analysis_dir, "processed_data.csv")

        # Сохраняем обработанные данные
        processed_df.to_csv(output_file, index=False)

        # Обновляем DataFrame и отображение
        self.df = processed_df
        self.data_preview.load_data(processed_df)
    
        # Активируем кнопки
        self.show_stats_btn.setEnabled(True)
        self.apply_filter_btn.setEnabled(True)

        # Подготовка сообщения о невалидных значениях
        invalid_message = "Статистика невалидных значений:\n\n"
    
        # Проверяем наличие невалидных значений
        has_invalid_values = bool(null_stats.sum() > 0 or nan_means.sum() > 0)
    
        if has_invalid_values:
            invalid_message += "Количество пропущенных значений:\n"
            invalid_message += "\n".join([f"{col}: {count}" for col, count in null_stats.items() if count > 0])
            invalid_message += "\n\nПроцент пропущенных значений:\n"
            invalid_message += "\n".join([f"{col}: {percent:.2f}%" for col, percent in nan_means.items() if percent > 0])
        else:
            invalid_message += "Невалидные значения не обнаружены (0 пропущенных значений во всех столбцах)"
    
        # Обновляем информационную метку
        self.info_label.setText(f"Данные обработаны и сохранены в: {output_file}")

        # Показываем сообщение о результатах обработки
        QMessageBox.information(
            self, 
            "Обработка данных", 
            f"Данные успешно обработаны и сохранены!\n\n"
            f"Путь сохранения: {output_file}\n\n"
            f"{invalid_message}"
        )

    def show_temperature_stats(self):
        """Показывает статистику температуры."""
        if self.df is None:
            QMessageBox.warning(self, "Предупреждение", "Нет загруженных данных")
            return

        # Получаем столбцы с температурой
        temp_cols = [col for col in self.df.columns 
                    if ('temperature' in col and 'fahrenheit' not in col)]
        fahrenheit_cols = [col for col in self.df.columns 
                         if ('temperature' in col and 'fahrenheit' in col)]

        # Собираем все температурные данные
        temp_data = pd.concat([
            self.df[temp_cols],      # Цельсий
            self.df[fahrenheit_cols]  # Фаренгейт
        ], axis=1)

        # Вычисляем статистику
        stats = temp_data.describe()

        # Создаем и показываем диалог со статистикой
        dialog = StatsDialog(stats, self)
        dialog.exec()

    def apply_filters(self):
        """Применяет фильтры к данным."""
        if self.df is None:
            return

        filtered_df = self.df.copy()

        # Применяем фильтр температуры
        if self.temp_filter.text():
            try:
                min_temp = float(self.temp_filter.text())
                if 'temperature_day' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['temperature_day'] >= min_temp]
                
                    # Убеждаемся, что даты отображаются без времени
                    filtered_df['date'] = pd.to_datetime(filtered_df['date']).dt.date
                
                    # Создаем директорию для отфильтрованных данных
                    filtered_dir = 'filtered_data'
                    if not os.path.exists(filtered_dir):
                        os.makedirs(filtered_dir)
                
                    # Сохраняем отфильтрованные данные
                    base_name = os.path.splitext(os.path.basename(self.current_file))[0] if self.current_file else "data"
                    output_file = os.path.join(filtered_dir, f"{base_name}_temp_{min_temp}.csv")
                
                    # Преобразуем даты в строки перед сохранением
                    save_df = filtered_df.copy()
                    save_df['date'] = save_df['date'].astype(str)
                    save_df.to_csv(output_file, index=False)
                
                    QMessageBox.information(
                        self,
                        "Фильтрация по температуре",
                        f"Отфильтрованные данные сохранены в:\n{output_file}"
                    )
            
                elif 'температура_(день)' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['температура_(день)'] >= min_temp]
                    filtered_df['дата'] = pd.to_datetime(filtered_df['дата']).dt.date
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Некорректное значение температуры")
                return

        # Применяем фильтр дат
        if self.start_date.text() and self.end_date.text():
            try:
                start = pd.to_datetime(self.start_date.text())
                end = pd.to_datetime(self.end_date.text())
            
                # Преобразуем столбец даты, если он еще не в формате datetime
                date_col = 'date' if 'date' in filtered_df.columns else 'дата'
                if filtered_df[date_col].dtype != 'datetime64[ns]':
                    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col])
            
                # Применяем фильтр и преобразуем даты
                filtered_df = filtered_df[
                    (filtered_df[date_col].dt.date >= start.date()) & 
                    (filtered_df[date_col].dt.date <= end.date())
                ]
                filtered_df[date_col] = filtered_df[date_col].dt.date
            
                # Активируем кнопку показа отфильтрованных по дате данных
                self.show_date_filter_btn.setEnabled(True)
            
            except ValueError:
                QMessageBox.warning(self, "Ошибка", "Некорректный формат даты. Используйте YYYY-MM-DD")
                return
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Ошибка при фильтрации дат: {str(e)}")
                return

        # Обновляем отображение
        self.data_preview.load_data(filtered_df)
        self.info_label.setText(f"Отфильтрованные данные: {len(filtered_df)} записей")

    def show_date_filtered_data(self):
        """Показывает отфильтрованные по дате данные в отдельном окне."""
        if not (self.start_date.text() and self.end_date.text()):
            QMessageBox.warning(self, "Ошибка", "Укажите начальную и конечную даты")
            return
        
        try:
            start = pd.to_datetime(self.start_date.text())
            end = pd.to_datetime(self.end_date.text())
        
            # Получаем отфильтрованные данные
            date_col = 'date' if 'date' in self.df.columns else 'дата'
            df_dates = pd.to_datetime(self.df[date_col])
            filtered_df = self.df[
                (df_dates.dt.date >= start.date()) & 
                (df_dates.dt.date <= end.date())
            ].copy()
        
            # Преобразуем даты в формат без времени
            filtered_df[date_col] = pd.to_datetime(filtered_df[date_col]).dt.date
        
            # Показываем диалог с данными
            dialog = DateFilterDialog(filtered_df, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'), self)
            dialog.exec()
        
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Некорректный формат даты")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при фильтрации: {str(e)}")

    def show_monthly_averages(self):
        """Показывает и сохраняет средние температуры по месяцам."""
        if self.df is None:
            return

        try:
            # Преобразуем дату в datetime если нужно
            date_col = 'date' if 'date' in self.df.columns else 'дата'
            df_temp = self.df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])

            # Создаем столбец для месяца-года
            df_temp['month_year'] = df_temp[date_col].dt.strftime('%Y-%m')

            # Получаем столбцы температур
            temp_cols = [col for col in df_temp.columns if 'temperature' in col]
        
            # Группируем по месяцам и вычисляем среднее
            monthly_avg = df_temp.groupby('month_year')[temp_cols].mean().round(2)

            # Создаем директорию для сохранения результатов
            avg_temp_dir = 'average_temperature'
            if not os.path.exists(avg_temp_dir):
                os.makedirs(avg_temp_dir)

            # Формируем имя файла
            start_date = df_temp[date_col].min().strftime('%Y%m')
            end_date = df_temp[date_col].max().strftime('%Y%m')
            output_file = os.path.join(avg_temp_dir, f'monthly_avg_{start_date}_to_{end_date}.csv')

            # Сохраняем результаты
            monthly_avg.to_csv(output_file)

            # Показываем результаты в диалоговом окне
            dialog = StatsDialog(monthly_avg, self)
            dialog.setWindowTitle("Средняя температура по месяцам")
            dialog.exec()

            QMessageBox.information(
                self,
                "Анализ температуры",
                f"Результаты сохранены в:\n{output_file}"
            )

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при анализе: {str(e)}")

    def plot_temperature_changes(self):
        """Создает график изменения температуры за весь период."""
        if self.df is None:
            return

        try:
            # Закрываем все открытые фигуры
            plt.close('all')
        
            # Создаем фигуру
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # Получаем данные
            date_col = 'date' if 'date' in self.df.columns else 'дата'
            dates = pd.to_datetime(self.df[date_col])
        
            # График для температуры по Цельсию
            temp_c = self.df['temperature_day']
            ax1.plot(dates, temp_c, label='Температура (°C)')
            ax1.set_title('Изменение температуры (°C)')
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Температура (°C)')
            ax1.grid(True)
            ax1.legend()

            # График для температуры по Фаренгейту
            temp_f = self.df['temperature_day_fahrenheit']
            ax2.plot(dates, temp_f, label='Температура (°F)', color='red')
            ax2.set_title('Изменение температуры (°F)')
            ax2.set_xlabel('Дата')
            ax2.set_ylabel('Температура (°F)')
            ax2.grid(True)
            ax2.legend()

            plt.tight_layout()
        
            # Сохраняем график
            plots_dir = 'temperature_plots'
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            start_date = dates.min().strftime('%Y%m')
            end_date = dates.max().strftime('%Y%m')
            plot_file = os.path.join(plots_dir, f'temperature_changes_{start_date}_to_{end_date}.png')
            plt.savefig(plot_file)
        
            # Показываем график
            plt.show()

            QMessageBox.information(
                self,
                "График температуры",
                f"График сохранен в:\n{plot_file}"
            )

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при построении графика: {str(e)}")

    def plot_monthly_temperature(self):
        """Создает графики температуры за выбранный месяц."""
        if self.df is None:
            return

        try:
            month = int(self.month_combo.currentText())
            year = int(self.year_combo.currentText())

            # Фильтруем данные за выбранный месяц
            date_col = 'date' if 'date' in self.df.columns else 'дата'
            df_temp = self.df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            mask = (df_temp[date_col].dt.year == year) & (df_temp[date_col].dt.month == month)
            monthly_data = df_temp[mask]

            if monthly_data.empty:
                QMessageBox.warning(self, "Предупреждение", "Нет данных за выбранный период")
                return

            # Закрываем все открытые фигуры
            plt.close('all')
        
            # Создаем фигуру с двумя графиками
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
            dates = monthly_data[date_col]

            # График температуры в Цельсиях
            temp_c = monthly_data['temperature_day']
            ax1.plot(dates, temp_c, 'b-', label='Температура (°C)')
        
            # Добавляем среднее и медиану для Цельсия
            mean_temp_c = temp_c.mean()
            median_temp_c = temp_c.median()
            ax1.axhline(y=mean_temp_c, color='r', linestyle='--', 
                       label=f'Среднее: {mean_temp_c:.1f}°C')
            ax1.axhline(y=median_temp_c, color='g', linestyle='--', 
                       label=f'Медиана: {median_temp_c:.1f}°C')
        
            ax1.set_title(f'Температура (°C) за {month}.{year}')
            ax1.set_ylabel('Температура (°C)')
            ax1.grid(True)
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)

            # График температуры в Фаренгейтах
            temp_f = monthly_data['temperature_day_fahrenheit']
            ax2.plot(dates, temp_f, 'r-', label='Температура (°F)')
        
            # Добавляем среднее и медиану для Фаренгейта
            mean_temp_f = temp_f.mean()
            median_temp_f = temp_f.median()
            ax2.axhline(y=mean_temp_f, color='b', linestyle='--', 
                       label=f'Среднее: {mean_temp_f:.1f}°F')
            ax2.axhline(y=median_temp_f, color='g', linestyle='--', 
                       label=f'Медиана: {median_temp_f:.1f}°F')
        
            ax2.set_title(f'Температура (°F) за {month}.{year}')
            ax2.set_ylabel('Температура (°F)')
            ax2.grid(True)
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # Сохраняем график
            plots_dir = 'temperature_plots'
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            plot_file = os.path.join(plots_dir, f'temperature_{year}_{month:02d}.png')
            plt.savefig(plot_file)
        
            # Показываем график
            plt.show()

            QMessageBox.information(
                self,
                "График температуры",
                f"График сохранен в:\n{plot_file}"
            )

        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Выберите корректный месяц и год")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка при построении графика: {str(e)}")

    def load_data(self, df: pd.DataFrame, file_path: Optional[str] = None):
        """Загрузка данных для анализа."""
        if df is None:
            return
    
        self.df = df.copy()
        self.current_file = file_path
        self.data_preview.load_data(df)
        self.process_data_btn.setEnabled(True)
    
        # Активируем кнопки анализа температуры
        self.group_by_month_btn.setEnabled(True)
        self.plot_temp_btn.setEnabled(True)
        self.plot_month_temp_btn.setEnabled(True)
    
        # Обновляем список годов в комбобоксе
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        else:
            dates = pd.to_datetime(df['дата'])
        years = sorted(dates.dt.year.unique())
        self.year_combo.clear()
        self.year_combo.addItems([str(year) for year in years])
    
        self.info_label.setText("Данные загружены. Выполните обработку данных для начала анализа.")

if __name__ == "__main__":
    print("Этот модуль является частью приложения WeatherDataHub")