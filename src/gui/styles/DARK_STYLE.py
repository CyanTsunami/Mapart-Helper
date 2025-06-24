# Классическая тёмная тема
DARK_STYLE = """
QMainWindow {
    background-color: #2D2D2D;
    color: #E0E0E0;
    font-family: Arial, sans-serif;
    font-size: 12px;
}

QLabel {
    color: #E0E0E0;
    qproperty-alignment: AlignCenter;
    border: 2px dashed #555;
    background-color: #252525;
    padding: 10px;
}

QPushButton {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 100px;
}

QPushButton:hover {
    background-color: #4A4A4A;
}

QPushButton:disabled {
    background-color: #2A2A2A;
    color: #707070;
}

QComboBox {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 5px;
    min-width: 150px;
}

QComboBox QAbstractItemView {
    background-color: #3A3A3A;
    color: #E0E0E0;
    selection-background-color: #505050;
}

QProgressBar {
    border: 1px solid #555;
    border-radius: 4px;
    text-align: center;
    background-color: #252525;
    color: #E0E0E0;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #4CAF50;
    border-radius: 3px;
}

QMessageBox {
    background-color: #2D2D2D;
}

QMessageBox QLabel {
    color: #E0E0E0;
    border: none;
    background-color: transparent;
}

QSpinBox {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #555;
    border-radius: 4px;
    padding: 5px;
    min-width: 60px;
}
"""
