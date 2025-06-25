from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal
from numba import set_num_threads
import numpy as np
import os


class ImageConverter(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    file_finished = pyqtSignal(str)

    def __init__(self, image_path, palette_colors, threads, method, block_size=1000):
        super().__init__()
        self.image_path = image_path
        self.palette_colors = palette_colors
        self.method = method
        self.block_size = block_size
        self.threads = threads
        self._is_running = True
        set_num_threads(threads)

    def run(self):
        try:
            # Загрузка изображения
            img = Image.open(self.image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_array = np.array(img, dtype=np.uint8)

            # Преобразуем палитру в массив numpy
            palette_array = np.array([(int(c[2:4],16), int(c[4:6],16), int(c[6:8],16)) for c in self.palette_colors],
                                   dtype=np.uint8)

            result = self.method(self, img_array, palette_array)
            self.result_ready.emit(result)

            # Сохраняем результат
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_converted.png")
            Image.fromarray(result, 'RGB').save(output_path)

            self.file_finished.emit(output_path)

        except Exception as e:
            print(f"Error in ImageConverter: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._is_running = False
        self.wait()