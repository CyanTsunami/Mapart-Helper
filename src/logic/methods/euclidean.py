from scipy.spatial import cKDTree
import numpy as np
import concurrent.futures

from src.logic.image_converter import ImageConverter


def euclidean_method(converter: ImageConverter, img_array: np.array, palette_array: np.array) -> np.array:
    def process_euclidean_chunk(pixels_chunk, palette_array, start_idx, end_idx):
        """Обработка блока пикселей с использованием евклидова расстояния"""
        # Создаём дерево для быстрого поиска ближайших цветов
        tree = cKDTree(palette_array)

        # Находим ближайшие цвета для всех пикселей в блоке
        distances, indices = tree.query(pixels_chunk, workers=1)

        # Получаем цвета из палитры по найденным индексам
        chunk_result = palette_array[indices]

        return chunk_result, start_idx, end_idx
    
    # Многопоточная реализация евклидова расстояния
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
                    process_euclidean_chunk,
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
    return euclidean_method, 'Евклидово расстояние (быстро)'
