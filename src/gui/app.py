from PIL import Image
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QLabel,
                            QPushButton, QComboBox, QHBoxLayout, QVBoxLayout,
                            QWidget, QProgressBar, QMessageBox, QSpinBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from multiprocessing import cpu_count
from numba import set_num_threads
from pathlib import Path
import json
import os
import sys
import subprocess

from ..logic.methods_manager import MethodsManager
from ..logic.image_converter import ImageConverter
from .styles.DARK_STYLE import DARK_STYLE


__all__ = ('MainWindow')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.PARENT_PATH = Path(sys.argv[0]).parent
        self.output_dir = self.PARENT_PATH / 'output'
        self.methods_manager = MethodsManager(self.PARENT_PATH / 'src' / 'logic' / 'methods')
        self.palettes = {}
        self.conversion_queue = []
        self.is_processing_queue = False
        self.current_palette = None
        self.current_image_path = None
        self.max_threads = cpu_count() - 1 or 1
        self.output_dir.mkdir(exist_ok=True)
        set_num_threads(self.max_threads)

        self.setup_ui()
        self.setStyleSheet(DARK_STYLE)
        self.load_palettes_on_startup()
        self.load_methods_on_startup()

    def setup_ui(self):
        self.setWindowTitle("Mapart Helper")
        self.setGeometry(100, 100, 1000, 650)

        # Создаем главный виджет и основной layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Layout для изображений (горизонтальный)
        images_layout = QHBoxLayout()
        images_layout.setSpacing(15)

        # Исходное изображение
        self.image_label = QLabel("Перетащите изображение сюда", self)
        self.image_label.setMinimumSize(450, 400)
        self.image_label.setToolTip("Область для отображения исходного изображения")
        images_layout.addWidget(self.image_label)

        # Результат
        self.result_label = QLabel("Результат конвертации", self)
        self.result_label.setMinimumSize(450, 400)
        self.result_label.setToolTip("Область для отображения результата конвертации")
        images_layout.addWidget(self.result_label)

        main_layout.addLayout(images_layout)

        # Панель управления
        control_panel = QHBoxLayout()
        control_panel.setSpacing(10)

        self.open_image_btn = QPushButton("Открыть изображение", self)
        self.open_image_btn.setToolTip("Открыть одно или несколько изображений для конвертации")

        self.open_palette_btn = QPushButton("Открыть палитру", self)
        self.open_palette_btn.setToolTip("Загрузить новую палитру из файла (TXT или JSON)")

        self.convert_btn = QPushButton("Конвертировать", self)
        self.convert_btn.setToolTip("Начать конвертацию текущего изображения или очереди")
        self.convert_btn.setEnabled(False)

        self.open_output_btn = QPushButton("Открыть результаты", self)
        self.open_output_btn.setToolTip("Открыть папку с результатами конвертации")
        self.open_output_btn.clicked.connect(self.open_output_folder)

        control_panel.addWidget(self.open_image_btn)
        control_panel.addWidget(self.open_palette_btn)
        control_panel.addWidget(self.convert_btn)
        control_panel.addWidget(self.open_output_btn)

        main_layout.addLayout(control_panel)

        # Выбор палитры и метода
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(10)

        settings_layout.addWidget(QLabel("Палитра:"))

        self.palette_combo = QComboBox()
        self.palette_combo.setMinimumWidth(200)
        self.palette_combo.setToolTip("Выберите палитру для конвертации изображения")
        settings_layout.addWidget(self.palette_combo)

        settings_layout.addWidget(QLabel("Метод:"))

        self.method_combo = QComboBox()
        settings_layout.addWidget(self.method_combo)

        # Настройки потоков
        settings_layout.addWidget(QLabel("Потоки:"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, self.max_threads)
        self.threads_spin.setValue(self.max_threads)
        self.threads_spin.setToolTip(f"Количество потоков для обработки (рекомендуется: {self.max_threads})")
        settings_layout.addWidget(self.threads_spin)

        main_layout.addLayout(settings_layout)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setToolTip("Прогресс выполнения текущей операции")
        main_layout.addWidget(self.progress_bar)

        # Подключаем сигналы
        self.open_image_btn.clicked.connect(self.open_image)
        self.open_palette_btn.clicked.connect(self.open_palette)
        self.convert_btn.clicked.connect(self.convert_current_image)
        self.palette_combo.currentTextChanged.connect(self.select_palette)

    def open_output_folder(self):
        """Открывает папку с результатами в проводнике системы"""
        output_path = os.path.abspath(self.output_dir)
        try:
            if os.name == 'nt':  # Для Windows
                os.startfile(output_path)
            elif os.name == 'posix':  # Для Linux/Mac
                subprocess.Popen(['xdg-open', output_path])
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть папку:\n{str(e)}")

    def load_palettes_on_startup(self):
        """Загружает все сохранённые палитры при старте"""
        palettes_dir = self.PARENT_PATH / 'palettes'
        if not palettes_dir.exists():
            palettes_dir.mkdir(exist_ok=True)
            return

        for file in palettes_dir.iterdir():
            filename = file.name
            if filename.endswith(".json"):
                try:
                    with open(file, 'r') as f:
                        self.palettes[filename] = json.load(f)
                except Exception as e:
                    print(f"Ошибка загрузки палитры {filename}: {e}")

        self.update_palette_combo()

    def load_methods_on_startup(self):
        """Загружает все методы обработки при старте"""
        methods_dir = self.PARENT_PATH / 'src' / 'logic' / 'methods'
        if not methods_dir.exists():
            methods_dir.mkdir(exist_ok=True)
            QMessageBox.warning(self,
                                "Методы не найдены!",
                                "Папка с методами обработки не найдена и была создана заново! " \
                                "Установите методы с репозитория или создайте собственные (для опытных)")
            return
        self.update_methods_combo()


    def update_palette_combo(self):
        self.palette_combo.clear()
        self.palette_combo.addItem("-- Выберите палитру --")
        self.palette_combo.addItems(sorted(self.palettes.keys()))

    def update_methods_combo(self):
        self.method_combo.clear()
        self.methods_manager.update()
        self.method_combo.addItems(self.methods_manager.keys())

    def open_image(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения", "",
            "Images (*.png *.jpg *.jpeg *.bmp)")

        if paths:
            if len(paths) == 1:
                self.load_image(paths[0])
            else:
                self.add_files_to_queue(paths)
                self.convert_btn.setEnabled(self.current_palette is not None)

    def load_image(self, path):
        try:
            self.current_image_path = path
            pixmap = QPixmap(path)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.convert_btn.setEnabled(self.current_palette is not None)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение:\n{str(e)}")

    def open_palette(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл палитры", "",
            "Palette files (*.txt *.json)")
        if path:
            self.load_palette_file(path)

    def load_palette_file(self, path):
        try:
            if path.lower().endswith('.txt'):
                with open(path, 'r') as f:
                    lines = filter(lambda line: not line.startswith(';'), f.readlines())
                    colors = [line.strip() for line in lines if line.strip()]
                    palette_name = os.path.splitext(os.path.basename(path))[0]
                    self.palettes[palette_name] = colors

                    # Сохраняем в JSON
                    json_path = os.path.join("palettes", f"{palette_name}.json")
                    with open(json_path, 'w') as json_file:
                        json.dump(colors, json_file)
            elif path.lower().endswith('.json'):
                with open(path, 'r') as f:
                    colors = json.load(f)
                    palette_name = os.path.splitext(os.path.basename(path))[0]
                    self.palettes[palette_name] = colors

            self.update_palette_combo()
            QMessageBox.information(self, "Успех", f"Палитра '{palette_name}' загружена!")

            # Активируем кнопки, если есть изображения для конвертации
            if self.current_image_path or self.conversion_queue:
                self.convert_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить палитру:\n{str(e)}")

    def select_palette(self, name):
        if name in self.palettes:
            self.current_palette = self.palettes[name]
            if self.current_image_path or self.conversion_queue:
                self.convert_btn.setEnabled(True)

    def add_files_to_queue(self, file_paths):
        """Добавляет файлы в очередь обработки"""
        for file_path in file_paths:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.conversion_queue.append(file_path)

        if len(file_paths) > 1:
            QMessageBox.information(
                self,
                "Добавлено в очередь",
                f"Добавлено {len(file_paths)} изображений в очередь обработки. "
                f"Всего в очереди: {len(self.conversion_queue)}"
            )

    def convert_current_image(self):
        """Конвертирует текущее изображение или начинает обработку очереди"""
        if not self.current_palette:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите палитру")
            return

        if not self.current_image_path and not self.conversion_queue:
            QMessageBox.warning(self, "Ошибка", "Нет изображений для конвертации")
            return

        # Если есть очередь, начинаем обработку
        if self.conversion_queue and not self.is_processing_queue:
            self.process_next_in_queue()
        # Иначе конвертируем текущее изображение
        elif self.current_image_path:
            self.convert_image(self.current_image_path)

    def process_next_in_queue(self):
        """Обрабатывает следующий элемент в очереди"""
        if not self.conversion_queue:
            self.is_processing_queue = False
            QMessageBox.information(self, "Готово", "Все изображения обработаны")
            self.open_output_folder()  # Открываем папку после обработки всей очереди
            return

        self.is_processing_queue = True
        image_path = self.conversion_queue.pop(0)
        self.load_image(image_path)
        self.convert_image(image_path)

    def convert_image(self, image_path):
        """Запускает конвертацию изображения"""
        self.progress_bar.setValue(0)
        self.convert_btn.setEnabled(False)

        method = self.methods_manager.get(self.method_combo.currentText())
        threads = self.threads_spin.value()

        self.converter = ImageConverter(
            image_path,
            self.current_palette,
            threads,
            method=method,
            block_size=1000
        )
        self.converter.progress_updated.connect(self.progress_bar.setValue)
        self.converter.result_ready.connect(self.show_result)
        self.converter.file_finished.connect(self.on_file_converted)
        self.converter.finished.connect(self.on_conversion_finished)
        self.converter.start()

    def show_result(self, img_array):
        """Отображение результата конвертации"""
        try:
            height, width, _ = img_array.shape
            self.result_image = Image.fromarray(img_array, 'RGB')

            # Конвертируем в QImage
            bytes_per_line = 3 * width
            qimage = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Отображаем результат
            self.result_label.setPixmap(QPixmap.fromImage(qimage).scaled(
                self.result_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        except Exception as e:
            print(f"Error showing result: {e}")

    def on_file_converted(self, output_path):
        """Вызывается при завершении конвертации одного файла"""
        print(f"Файл сохранен: {output_path}")

    def on_conversion_finished(self):
        """Вызывается при завершении конвертации"""
        self.convert_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Если есть очередь, обрабатываем следующий файл
        if self.conversion_queue and self.is_processing_queue:
            self.process_next_in_queue()
        else:
            self.is_processing_queue = False
            # Открываем папку output после завершения всех операций
            self.open_output_folder()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        file_paths = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_paths.append(file_path)
            elif file_path.lower().endswith(('.txt', '.json')):
                self.load_palette_file(file_path)

        if file_paths:
            if len(file_paths) == 1:
                self.load_image(file_paths[0])
            else:
                self.add_files_to_queue(file_paths)
                self.convert_btn.setEnabled(self.current_palette is not None)
