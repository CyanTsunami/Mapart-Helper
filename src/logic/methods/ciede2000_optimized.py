import numpy as np
import concurrent.futures

from ..image_converter import ImageConverter
from ..transformers import (rgb_to_lab_numba, ciede2000_numba_batch)


def ciede2000_optimized_method(converter: ImageConverter, img_array: np.array, palette_array: np.array) -> np.array:
    def process_block(block, palette_lab, palette_array, block_start, image_width):
        """Обработка блока пикселей в отдельном потоке (для CIEDE2000)"""
        # Преобразуем блок в Lab
        block_lab = np.array([rgb_to_lab_numba(p) for p in block])

        # Находим ближайшие цвета в палитре
        distances = ciede2000_numba_batch(block_lab, palette_lab)
        best_indices = np.argmin(distances, axis=1)

        # Получаем цвета из палитры
        block_result = palette_array[best_indices]

        return block_result, block_start, block.shape[0]
    
    height, width = img_array.shape[:2]
    pixels = img_array.reshape(-1, 3)
    palette_lab = np.array([rgb_to_lab_numba(c) for c in palette_array])

    result = np.zeros((height, width, 3), dtype=np.uint8)
    total_pixels = pixels.shape[0]
    processed_pixels = 0

    # Обрабатываем изображение блоками с использованием ThreadPool
    with concurrent.futures.ThreadPoolExecutor(max_workers=converter.threads) as executor:
        futures = []
        for i in range(0, total_pixels, converter.block_size):
            if not converter._is_running:
                return

            block_end = min(i + converter.block_size, total_pixels)
            block = pixels[i:block_end]

            futures.append(executor.submit(
                process_block,
                block, palette_lab, palette_array, i, width
            ))

        for future in concurrent.futures.as_completed(futures):
            if not converter._is_running:
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
            converter.progress_updated.emit(progress)
    
    return result