from pathlib import Path
import importlib.util


class MethodsManager:
    """Менеджер методов, который динамически подгружает способы обработки
    изображений. Позволяет в программу встраивать свои методы
    """
    def __init__(self, dir: Path):
        self.__methods = {}
        self.__methods_dir = dir
    
    def __load_method(self, file: Path):
        """Загрузка метода обработки"""

        # Метод не особо безопасный, придумать что-то
        spec = importlib.util.spec_from_file_location(file.name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'load') and callable(module.load):
            response = module.load()
            if not isinstance(response, tuple):
                response = (response, file.name)
            return response
        raise AttributeError("В модуле отсутствует функция load")

    def update(self):
        """Обновляет и загружает список методов"""
        for file in self.__methods_dir.iterdir():
            filename = file.name
            if filename.endswith(".py"):
                try:
                    func, name = self.__load_method(file)
                    self.__methods[name] = func
                except Exception as e:
                    print(f"Ошибка загрузки метода обработки {filename}: {e}")
    
    def get(self, key):
        return self.__methods.get(key)
    
    def keys(self):
        return self.__methods.keys()
