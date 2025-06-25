import numpy as np
import concurrent.futures

from ..image_converter import ImageConverter
from ..transformers import (WEIGHTED_EUCLIDEAN_WEIGHTS, weighted_euclidean_batch)


def weighted_euclidean_method(converter: ImageConverter, img_array: np.array, palette_array: np.array) -> np.array:
    def process_weighted_euclidean_chunk(pixels_chunk, palette_array, weights, start_idx, end_idx):
        """Обработка блока пикселей с использованием взвешенного евклидова расстояния"""
        # Вычисляем расстояния
        distances = weighted_euclidean_batch(pixels_chunk, palette_array, weights)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        chunk_result = palette_array[best_indices]

        return chunk_result, start_idx, end_idx
    
    # Реализация взвешенного евклидова расстояния
    height, width = img_array.shape[:2]
    pixels = img_array.reshape(-1, 3)
    result = np.zeros((height * width, 3), dtype=np.uint8)

    # Разделяем работу на блоки
    total_pixels = len(pixels)
    chunk_size = max(1, total_pixels // converter.threads)

    with concurrent.futures.ThreadPoolExecutor(max_workers=converter.threads) as executor:
        futures = []
        for i in range(0, total_pixels, chunk_size):
            chunk_end = min(i + chunk_size, total_pixels)
            futures.append(
                executor.submit(
                    process_weighted_euclidean_chunk,
                    pixels[i:chunk_end],
                    palette_array,
                    WEIGHTED_EUCLIDEAN_WEIGHTS,
                    i,
                    chunk_end
                )
            )

        for future in concurrent.futures.as_completed(futures):
            if not converter._is_running:
                return

            chunk_result, start_idx, end_idx = future.result()
            result[start_idx:end_idx] = chunk_result

            # Обновляем прогресс
            progress = int((end_idx / total_pixels) * 100)
            converter.progress_updated.emit(progress)

    # Формируем итоговое изображение
    return result.reshape((height, width, 3))