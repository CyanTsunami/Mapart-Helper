DARK_STYLE = """
/* Основные настройки */
QMainWindow, QDialog {
    background-color: #1E1E1E;
    color: #E0E0E0;
    font-family: Arial, sans-serif;
    font-size: 13px;
}

/* Общие свойства */
QWidget {
    selection-background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4A6BFF, stop:1 #6A4BFF);
    selection-color: #FFFFFF;
}

QLabel {
    color: #E0E0E0;
    qproperty-alignment: AlignCenter;
    border: 1px solid #3A3A3A;
    border-radius: 6px;
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2E2E2E, stop:1 #262626);
    padding: 8px 12px;
    font-size: 12px;
}

QLabel#title {
    font-size: 16px;
    font-weight: bold;
    color: #4CAF50;
    background: transparent;
    border: none;
}

/* Стильные кнопки*/
QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3A3A3A, stop:1 #2E2E2E);
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    border-radius: 6px;
    padding: 8px 16px;
    min-width: 100px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4A4A4A, stop:1 #3E3E3E);
    border-color: #5A5A5A;
}

QPushButton:pressed {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2A2A2A, stop:1 #222222);
}

QPushButton:disabled {
    background-color: #2A2A2A;
    color: #606060;
    border-color: #353535;
}

QPushButton#primary {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4CAF50, stop:1 #3E9F42);
    border-color: #5CBF60;
}

QPushButton#primary:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5CBF60, stop:1 #4EAF52);
}

QPushButton#primary:pressed {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3C9F40, stop:1 #2E8F32);
}

/* Выпадающие списки */
QComboBox {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3A3A3A, stop:1 #2E2E2E);
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    border-radius: 6px;
    padding: 6px 12px;
    min-width: 150px;
}

QComboBox:hover {
    border-color: #5A5A5A;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 25px;
    border-left-width: 1px;
    border-left-color: #4A4A4A;
    border-left-style: solid;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

QComboBox QAbstractItemView {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    selection-background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4A6BFF, stop:1 #6A4BFF);
    selection-color: #FFFFFF;
    outline: none;
    border-radius: 6px;
}

/* Прогресс-бар с анимацией */
QProgressBar {
    border: 1px solid #3A3A3A;
    border-radius: 6px;
    text-align: center;
    background-color: #2E2E2E;
    color: #E0E0E0;
    height: 22px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4CAF50, stop:1 #6A4BFF);
    border-radius: 5px;
}

/* Спинбоксы */
QSpinBox, QDoubleSpinBox {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3A3A3A, stop:1 #2E2E2E);
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    border-radius: 6px;
    padding: 6px;
    min-width: 70px;
}

QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #5A5A5A;
}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #4A4A4A;
    border-top-right-radius: 6px;
    border-bottom-right-radius: 6px;
}

/* Вкладки с анимацией */
QTabWidget::pane {
    border: 1px solid #3A3A3A;
    border-radius: 8px;
    margin-top: 5px;
    background-color: #2E2E2E;
}

QTabBar::tab {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3A3A3A, stop:1 #2E2E2E);
    color: #E0E0E0;
    border: 1px solid #3A3A3A;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 8px 20px;
}

QTabBar::tab:selected {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4A4A4A, stop:1 #3E3E3E);
    border-color: #4CAF50;
    border-bottom-color: #4A4A4A;
    color: #FFFFFF;
}

QTabBar::tab:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4A4A4A, stop:1 #3E3E3E);
    color: #FFFFFF;
}

/* Сообщения */
QMessageBox {
    background-color: #1E1E1E;
    border: 1px solid #3A3A3A;
    border-radius: 8px;
}

QMessageBox QLabel {
    color: #E0E0E0;
    border: none;
    background-color: transparent;
    padding: 0;
}

/* Разделители */
QFrame[frameShape="4"], /* HLine */
QFrame[frameShape="5"] { /* VLine */
    background-color: #3A3A3A;
    border: none;
    height: 1px;
}

/* Скроллбары? */
QScrollBar:vertical {
    border: none;
    background: #2E2E2E;
    width: 12px;
    margin: 0;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4A6BFF, stop:1 #6A4BFF);
    min-height: 30px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5A7BFF, stop:1 #7A5BFF);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
    background: none;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

/* Анимации */

@keyframes progressAnimation {
    0% { opacity: 0.8; }
    50% { opacity: 1; }
    100% { opacity: 0.8; }
}

/* Меню */
QMenuBar {
    background-color: #2E2E2E;
    color: #E0E0E0;
    border-bottom: 1px solid #3A3A3A;
}

QMenuBar::item {
    padding: 5px 10px;
    background-color: transparent;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #3A3A3A;
}

QMenu {
    background-color: #2E2E2E;
    border: 1px solid #3A3A3A;
    border-radius: 6px;
    padding: 5px;
}

QMenu::item {
    padding: 8px 25px 8px 20px;
    background-color: transparent;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4A6BFF, stop:1 #6A4BFF);
}

QMenu::separator {
    height: 1px;
    background-color: #3A3A3A;
    margin: 5px 0;
}

/* ToolTip */
QToolTip {
    background-color: #3A3A3A;
    color: #E0E0E0;
    border: 1px solid #4A4A4A;
    border-radius: 6px;
    padding: 5px 10px;
    opacity: 240;
}
"""
