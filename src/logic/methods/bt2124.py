import numpy as np
import concurrent.futures

from src.logic.image_converter import ImageConverter
from src.logic.transformers import bt2124_batch


def bt2124_method(converter: ImageConverter, img_array: np.array, palette_array: np.array) -> np.array:
    def process_bt2124_chunk(pixels_chunk, palette_array, start_idx, end_idx):
        """Обработка блока пикселей с использованием Rec. ITU-R BT.2124"""
        # Вычисляем расстояния
        distances = bt2124_batch(pixels_chunk, palette_array)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        chunk_result = palette_array[best_indices]

        return chunk_result, start_idx, end_idx
    
    # Реализация Rec. ITU-R BT.2124 (2019)
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
                    process_bt2124_chunk,
                    pixels[i:chunk_end],
                    palette_array,
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

def load():
    return bt2124_method, 'Rec. ITU-R BT.2124 (2019)'
