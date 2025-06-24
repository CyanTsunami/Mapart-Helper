from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
import sys

from src.gui.app import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamily("Arial")
    font.setPointSize(9)
    app.setFont(font)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
