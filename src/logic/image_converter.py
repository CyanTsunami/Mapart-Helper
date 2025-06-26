from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from numba import set_num_threads
from scipy.spatial import cKDTree
import numpy as np
import concurrent.futures
import os
from datetime import datetime

from .transformers import (
    WEIGHTED_EUCLIDEAN_WEIGHTS,
    weighted_euclidean_batch,
    rgb_to_lab_numba,
    ciede2000_numba_batch,
    bt2124_batch
)


class ImageConverter(QThread):
    """Конвертер изображений на основе QThread, который сопоставляет цвета изображений с заданной палитрой, используя различные методы.
    Поддерживает несколько алгоритмов сопоставления цветов и параллельную обработку.

    Signals:
        progress_updated(int): Выдает процент выполнения (0-100)
        result_ready(np.ndarray): Выдает преобразованное изображение в виде массива numpy
        finished(): Выдает сообщение о завершении преобразования
        file_finished(str): Выдает путь к сохраненному выходному файлу

    Args:
        image_path (str): Путь к входному изображению
        palette_colors (list): Список шестнадцатеричных строк цветов (формат: "#RRGGBB")
        method (str): Метод сопоставления цветов ('euclidean', 'weighted_euclidean',
 'ciede2000_optimized', или 'bt2124')
        block_size (int): Number of pixels to process in each block
        threads (int): Количество потоков для использования (None для автоматического)
    """

    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    file_finished = pyqtSignal(str)

    def __init__(
    self,
    image_path,
    palette_colors,
    method='euclidean',
    block_size=1000,
    threads=None):
        super().__init__()
        self.image_path = image_path
        self.palette_colors = palette_colors
        self.method = method
        self.block_size = block_size
        self.threads = threads
        self._is_running = True
        set_num_threads(threads or self.parent().max_threads)

    def run(self):
        try:
            # Загрузка изображения
            img = Image.open(self.image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.uint8)
            height, width = img_array.shape[:2]

            # Преобразуем палитру в массив numpy
            palette_array = np.array([(int(c[2:4],16), int(c[4:6],16), int(c[6:8],16)) for c in self.palette_colors],
                                   dtype=np.uint8)

            if self.method == 'euclidean':
                # Многопоточная реализация евклидова расстояния
                pixels = img_array.reshape(-1, 3)
                result = np.zeros((height * width, 3), dtype=np.uint8)

                # Разделяем работу на блоки
                total_pixels = len(pixels)
                chunk_size = max(1, total_pixels // self.threads)

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i in range(0, total_pixels, chunk_size):
                        chunk_end = min(i + chunk_size, total_pixels)
                        futures.append(
                            executor.submit(
                                self.process_euclidean_chunk,
                                pixels[i:chunk_end],
                                palette_array,
                                i,
                                chunk_end
                            )
                        )

                    for future in concurrent.futures.as_completed(futures):
                        if not self._is_running:
                            return

                        chunk_result, start_idx, end_idx = future.result()
                        result[start_idx:end_idx] = chunk_result

                        # Обновляем прогресс
                        progress = int((end_idx / total_pixels) * 100)
                        self.progress_updated.emit(progress)

                # Формируем итоговое изображение
                result = result.reshape((height, width, 3))
                self.result_ready.emit(result)

            elif self.method == 'ciede2000_optimized':
                pixels = img_array.reshape(-1, 3)
                palette_lab = np.array([rgb_to_lab_numba(c) for c in palette_array])

                result = np.zeros((height, width, 3), dtype=np.uint8)
                total_pixels = pixels.shape[0]
                processed_pixels = 0

                # Обрабатываем изображение блоками с использованием ThreadPool
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i in range(0, total_pixels, self.block_size):
                        if not self._is_running:
                            return

                        block_end = min(i + self.block_size, total_pixels)
                        block = pixels[i:block_end]

                        futures.append(executor.submit(
                            self.process_block,
                            block, palette_lab, palette_array, i, width
                        ))

                    for future in concurrent.futures.as_completed(futures):
                        if not self._is_running:
                            return

                        block_result, block_start, block_processed = future.result()

                        # Заполняем результат
                        for j in range(block_result.shape[0]):
                            pixel_idx = block_start + j
                            y = pixel_idx // width
                            x = pixel_idx % width
                            result[y, x] = block_result[j]

                        # Обновляем прогресс
                        processed_pixels += block_processed
                        progress = int((processed_pixels / total_pixels) * 100)
                        self.progress_updated.emit(progress)

                self.result_ready.emit(result)

            elif self.method == 'weighted_euclidean':
                # Реализация взвешенного евклидова расстояния
                pixels = img_array.reshape(-1, 3)
                result = np.zeros((height * width, 3), dtype=np.uint8)

                # Разделяем работу на блоки
                total_pixels = len(pixels)
                chunk_size = max(1, total_pixels // self.threads)

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i in range(0, total_pixels, chunk_size):
                        chunk_end = min(i + chunk_size, total_pixels)
                        futures.append(
                            executor.submit(
                                self.process_weighted_euclidean_chunk,
                                pixels[i:chunk_end],
                                palette_array,
                                WEIGHTED_EUCLIDEAN_WEIGHTS,
                                i,
                                chunk_end
                            )
                        )

                    for future in concurrent.futures.as_completed(futures):
                        if not self._is_running:
                            return

                        chunk_result, start_idx, end_idx = future.result()
                        result[start_idx:end_idx] = chunk_result

                        # Обновляем прогресс
                        progress = int((end_idx / total_pixels) * 100)
                        self.progress_updated.emit(progress)

                # Формируем итоговое изображение
                result = result.reshape((height, width, 3))
                self.result_ready.emit(result)

            elif self.method == 'bt2124':
                # Реализация Rec. ITU-R BT.2124 (2019)
                pixels = img_array.reshape(-1, 3)
                result = np.zeros((height * width, 3), dtype=np.uint8)

                # Разделяем работу на блоки
                total_pixels = len(pixels)
                chunk_size = max(1, total_pixels // self.threads)

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.threads) as executor:
                    futures = []
                    for i in range(0, total_pixels, chunk_size):
                        chunk_end = min(i + chunk_size, total_pixels)
                        futures.append(
                            executor.submit(
                                self.process_bt2124_chunk,
                                pixels[i:chunk_end],
                                palette_array,
                                i,
                                chunk_end
                            )
                        )

                    for future in concurrent.futures.as_completed(futures):
                        if not self._is_running:
                            return

                        chunk_result, start_idx, end_idx = future.result()
                        result[start_idx:end_idx] = chunk_result

                        # Обновляем прогресс
                        progress = int((end_idx / total_pixels) * 100)
                        self.progress_updated.emit(progress)

                # Формируем итоговое изображение
                result = result.reshape((height, width, 3))
                self.result_ready.emit(result)

            # Сохраняем результат
            output_dir = "output"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_{timestamp}.png")
            Image.fromarray(result, 'RGB').save(output_path)

            self.file_finished.emit(output_path)

        except Exception as e:
            print(f"Error in ImageConverter: {e}")
        finally:
            self.finished.emit()

    def process_euclidean_chunk(self, pixels_chunk, palette_array, start_idx, end_idx):
        """Обработка блока пикселей с использованием метода евклидова расстояния

        Args:
            pixels_chunk: массив RGB-пикселей для обработки
            palette_array: массив цветов палитры
            start_idx: начальный индекс этого чанка
            end_idx: конечный индекс этого чанка

        Returns:
            tuple: (result_chunk, start_idx, end_idx)
        """
        # Создаём дерево для быстрого поиска ближайших цветов
        tree = cKDTree(palette_array)

        # Находим ближайшие цвета для всех пикселей в блоке
        distances, indices = tree.query(pixels_chunk, workers=1)

        # Получаем цвета из палитры по найденным индексам
        chunk_result = palette_array[indices]

        return chunk_result, start_idx, end_idx

    def process_weighted_euclidean_chunk(self, pixels_chunk, palette_array, weights, start_idx, end_idx):
        """Обработка блока пикселей с использованием метода взвешенного евклидова расстояния

        Args:
            pixels_chunk: массив RGB-пикселей для обработки
            palette_array: массив цветов палитры
            weights: значения веса для каждого цветового канала
            start_idx: начальный индекс этого чанка
            end_idx: конечный индекс этого чанка

        Returns:
            tuple: (result_chunk, start_idx, end_idx)
        """
        # Вычисляем расстояния
        distances = weighted_euclidean_batch(pixels_chunk, palette_array, weights)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        chunk_result = palette_array[best_indices]

        return chunk_result, start_idx, end_idx

    def process_bt2124_chunk(self, pixels_chunk, palette_array, start_idx, end_idx):
        """Обработка блока пикселей с использованием метода Rec. ITU-R BT.2124

        Args:
            pixels_chunk: массив RGB-пикселей для обработки
            palette_array: массив цветов палитры
            start_idx: начальный индекс этого чанка
            end_idx: конечный индекс этого чанка

            Returns:
                tuple: (result_chunk, start_idx, end_idx)
    """
        # Вычисляем расстояния
        distances = bt2124_batch(pixels_chunk, palette_array)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        chunk_result = palette_array[best_indices]

        return chunk_result, start_idx, end_idx

    def process_block(self, block, palette_lab, palette_array, block_start, image_width):
        """Обработка блока пикселей для метода CIEDE2000

                Args:
                    block: массив RGB пикселей для обработки
                    palette_lab: цвета палитры в пространстве Lab
                    palette_array: исходные цвета палитры
                    block_start: Начальный индекс блока
                    image_width: Ширина исходного изображения

                Returns:
                    tuple: (result_block, block_start, num_processed)
                """
        # Преобразуем блок в Lab
        block_lab = np.array([rgb_to_lab_numba(p) for p in block])

        # Находим ближайшие цвета в палитре
        distances = ciede2000_numba_batch(block_lab, palette_lab)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        block_result = palette_array[best_indices]

        return block_result, block_start, block.shape[0]

    def stop(self):
        """Завершение процесса преобразования."""
        self._is_running = False
        self.wait()
