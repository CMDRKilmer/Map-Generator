"""
UI 样式系统 — 暗色游戏主题
"""

from PySide6.QtGui import QColor, QFont, QPalette


# ========== 色彩系统 ==========
class ThemeColors:
    """主题色彩"""

    # 主背景
    BG_DARKEST = QColor(18, 18, 24)  # 最深背景
    BG_DARK = QColor(28, 28, 36)  # 主背景
    BG_CARD = QColor(38, 38, 48)  # 卡片背景
    BG_HOVER = QColor(48, 48, 60)  # 悬停背景
    BG_ACTIVE = QColor(55, 55, 70)  # 激活背景

    # 边框
    BORDER = QColor(55, 55, 70)  # 普通边框
    BORDER_HOVER = QColor(80, 80, 100)  # 悬停边框
    BORDER_ACTIVE = QColor(100, 130, 180)  # 激活边框

    # 文字
    TEXT_PRIMARY = QColor(230, 230, 240)  # 主文字
    TEXT_SECONDARY = QColor(160, 160, 180)  # 次要文字
    TEXT_MUTED = QColor(100, 100, 120)  # 弱化文字
    TEXT_ACCENT = QColor(100, 180, 255)  # 强调文字

    # 强调色
    ACCENT_BLUE = QColor(80, 140, 220)  # 主强调色
    ACCENT_BLUE_LIGHT = QColor(120, 180, 255)
    ACCENT_BLUE_DARK = QColor(50, 100, 180)

    ACCENT_GOLD = QColor(220, 180, 60)  # 金色（用于重要操作）
    ACCENT_GREEN = QColor(80, 180, 100)  # 绿色（成功）
    ACCENT_RED = QColor(200, 80, 80)  # 红色（危险/删除）
    ACCENT_PURPLE = QColor(150, 100, 200)  # 紫色（特殊）

    # 地图相关
    MAP_BG = QColor(22, 22, 30)
    GRID_LINE = QColor(45, 45, 60)
    GRID_LINE_WATER = QColor(30, 60, 120)


# ========== 字体系统 ==========
class ThemeFonts:
    """主题字体"""

    @staticmethod
    def title() -> QFont:
        font = QFont("Microsoft YaHei", 11, QFont.Bold)
        return font

    @staticmethod
    def subtitle() -> QFont:
        font = QFont("Microsoft YaHei", 10, QFont.Bold)
        return font

    @staticmethod
    def body() -> QFont:
        font = QFont("Microsoft YaHei", 9)
        return font

    @staticmethod
    def caption() -> QFont:
        font = QFont("Microsoft YaHei", 8)
        return font

    @staticmethod
    def mono() -> QFont:
        font = QFont("Consolas", 9)
        return font


# ========== 样式表 ==========
def get_stylesheet() -> str:
    """返回全局样式表"""
    return """
    /* 主窗口 */
    QMainWindow {
        background-color: #1a1a20;
    }

    /* 菜单栏 */
    QMenuBar {
        background-color: #1e1e28;
        color: #e6e6f0;
        border-bottom: 1px solid #373748;
        padding: 4px;
    }
    QMenuBar::item {
        background: transparent;
        padding: 6px 12px;
        border-radius: 4px;
    }
    QMenuBar::item:selected {
        background-color: #373748;
    }
    QMenuBar::item:pressed {
        background-color: #4a4a5c;
    }

    /* 菜单 */
    QMenu {
        background-color: #262630;
        color: #e6e6f0;
        border: 1px solid #373748;
        border-radius: 6px;
        padding: 6px;
    }
    QMenu::item {
        padding: 8px 24px;
        border-radius: 4px;
    }
    QMenu::item:selected {
        background-color: #4a6fa5;
    }
    QMenu::separator {
        height: 1px;
        background-color: #373748;
        margin: 6px 12px;
    }

    /* 分组框 */
    QGroupBox {
        background-color: #262630;
        border: 1px solid #373748;
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 16px;
        padding: 16px;
        font-weight: bold;
        color: #a0a0b4;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 8px;
        color: #78b4ff;
    }

    /* 按钮 */
    QPushButton {
        background-color: #3a3a4a;
        color: #e6e6f0;
        border: 1px solid #4a4a5c;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #4a4a5c;
        border-color: #5a5a6e;
    }
    QPushButton:pressed {
        background-color: #323240;
    }
    QPushButton:disabled {
        background-color: #2a2a35;
        color: #646480;
        border-color: #323240;
    }

    /* 主要操作按钮 */
    QPushButton#primaryButton {
        background-color: #5080c8;
        border-color: #6090d8;
    }
    QPushButton#primaryButton:hover {
        background-color: #6098e0;
        border-color: #70a8f0;
    }
    QPushButton#primaryButton:pressed {
        background-color: #4068a8;
    }

    /* 危险操作按钮 */
    QPushButton#dangerButton {
        background-color: #a04040;
        border-color: #b05050;
    }
    QPushButton#dangerButton:hover {
        background-color: #b85050;
    }

    /* 滑块 */
    QSlider::groove:horizontal {
        height: 6px;
        background-color: #2a2a38;
        border-radius: 3px;
    }
    QSlider::sub-page:horizontal {
        background-color: #5080c8;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        width: 16px;
        height: 16px;
        background-color: #78b4ff;
        border: 2px solid #5080c8;
        border-radius: 8px;
        margin: -5px 0;
    }
    QSlider::handle:horizontal:hover {
        background-color: #a0d0ff;
        border-color: #78b4ff;
    }

    /* 复选框 */
    QCheckBox {
        color: #b0b0c8;
        spacing: 8px;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 2px solid #4a4a5c;
        background-color: #2a2a38;
    }
    QCheckBox::indicator:checked {
        background-color: #5080c8;
        border-color: #6098e0;
    }
    QCheckBox::indicator:hover {
        border-color: #5a5a6e;
    }

    /* 下拉框 */
    QComboBox {
        background-color: #2a2a38;
        color: #e6e6f0;
        border: 1px solid #4a4a5c;
        border-radius: 6px;
        padding: 6px 12px;
        min-width: 80px;
    }
    QComboBox:hover {
        border-color: #5a5a6e;
    }
    QComboBox::drop-down {
        border: none;
        width: 24px;
    }
    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid #a0a0b4;
        width: 0;
        height: 0;
    }
    QComboBox QAbstractItemView {
        background-color: #2a2a38;
        color: #e6e6f0;
        border: 1px solid #4a4a5c;
        border-radius: 6px;
        selection-background-color: #4a6fa5;
    }

    /* 数字输入框 */
    QSpinBox {
        background-color: #2a2a38;
        color: #e6e6f0;
        border: 1px solid #4a4a5c;
        border-radius: 6px;
        padding: 6px;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        background-color: #3a3a4a;
        border: none;
        width: 20px;
    }
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background-color: #4a4a5c;
    }

    /* 标签 */
    QLabel {
        color: #b0b0c8;
    }
    QLabel#titleLabel {
        color: #78b4ff;
        font-weight: bold;
        font-size: 12px;
    }
    QLabel#valueLabel {
        color: #e6e6f0;
        font-weight: bold;
        min-width: 36px;
    }

    /* 状态栏 */
    QStatusBar {
        background-color: #1e1e28;
        color: #808098;
        border-top: 1px solid #373748;
    }
    QStatusBar::item {
        border: none;
    }

    /* 分割器 */
    QSplitter::handle {
        background-color: #373748;
    }
    QSplitter::handle:horizontal {
        width: 2px;
    }
    QSplitter::handle:hover {
        background-color: #5080c8;
    }

    /* 工具提示 */
    QToolTip {
        background-color: #2a2a38;
        color: #e6e6f0;
        border: 1px solid #4a4a5c;
        border-radius: 6px;
        padding: 8px;
    }

    /* 滚动条 */
    QScrollBar:vertical {
        background-color: #1e1e28;
        width: 10px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical {
        background-color: #4a4a5c;
        border-radius: 5px;
        min-height: 30px;
    }
    QScrollBar::handle:vertical:hover {
        background-color: #5a5a6e;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    """


# ========== 调色板 ==========
def get_palette() -> QPalette:
    """返回应用程序调色板"""
    palette = QPalette()
    colors = ThemeColors()

    palette.setColor(QPalette.Window, colors.BG_DARK)
    palette.setColor(QPalette.WindowText, colors.TEXT_PRIMARY)
    palette.setColor(QPalette.Base, colors.BG_CARD)
    palette.setColor(QPalette.AlternateBase, colors.BG_DARKEST)
    palette.setColor(QPalette.ToolTipBase, colors.BG_CARD)
    palette.setColor(QPalette.ToolTipText, colors.TEXT_PRIMARY)
    palette.setColor(QPalette.Text, colors.TEXT_PRIMARY)
    palette.setColor(QPalette.Button, colors.BG_CARD)
    palette.setColor(QPalette.ButtonText, colors.TEXT_PRIMARY)
    palette.setColor(QPalette.BrightText, colors.TEXT_ACCENT)
    palette.setColor(QPalette.Highlight, colors.ACCENT_BLUE)
    palette.setColor(QPalette.HighlightedText, colors.TEXT_PRIMARY)

    return palette
