"""Dark theme. Neutral (untinted) greys so the surround never biases colour judgement, one rose accent."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

ASSETS = Path(__file__).resolve().parents[2] / "assets"

C = dict(
    bg0="#171717",       # window base
    bg1="#1E1E1E",       # panels
    bg2="#272727",       # inputs, raised controls
    bg3="#313131",       # hover
    bg4="#3B3B3B",       # pressed / active
    viewer="#121212",    # image surround
    line="#2B2B2B",      # hairline dividers
    line2="#383838",     # control borders
    text="#E8E8E8",
    text2="#A6A6A6",
    text3="#6F6F6F",
    accent="#F0709A",
    accent_hi="#F68DB1",
    accent_lo="#D45A83",
    accent_soft="#3A2530",
    warn="#E8B04A",
    error="#F0564A",
    ok="#5CC8A0",
    select="#4A4A4A",
)

FONT_FAMILY = "Pretendard"


def qc(name: str, alpha: int | None = None) -> QColor:
    c = QColor(C[name])
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def load_fonts() -> str:
    fam = None
    for f in sorted((ASSETS / "fonts").glob("Pretendard-*.ttf")):
        fid = QFontDatabase.addApplicationFont(str(f))
        if fid >= 0 and fam is None:
            fams = QFontDatabase.applicationFontFamilies(fid)
            if fams:
                fam = fams[0]
    global FONT_FAMILY
    FONT_FAMILY = fam or "Malgun Gothic"
    return FONT_FAMILY


def font(px: int = 12, weight: int = 400, tabular: bool = False) -> QFont:
    f = QFont(FONT_FAMILY)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    f.setHintingPreference(QFont.PreferNoHinting)
    if tabular:
        try:
            f.setFeature(QFont.Tag("tnum"), 1)
        except Exception:
            pass
    return f


def stylesheet() -> str:
    from .icons import icon_file
    c = C
    return f"""
* {{ color: {c['text']}; }}
QMainWindow, QWidget#Root {{ background: {c['bg0']}; }}
QWidget {{ background: transparent; }}
QWidget#Panel, QFrame#Panel {{ background: {c['bg1']}; }}
QWidget#TopBar {{ background: {c['bg0']}; border-bottom: 1px solid {c['line']}; }}
QFrame#HLine {{ background: {c['line']}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#VLine {{ background: {c['line']}; max-width: 1px; min-width: 1px; border: none; }}
QLabel#Muted {{ color: {c['text2']}; }}
QLabel#Faint {{ color: {c['text3']}; }}
QLabel#Section {{ color: {c['text2']}; font-weight: 600; font-size: 11px; padding-top: 6px; }}
QLabel#Title {{ font-size: 15px; font-weight: 600; }}
QLabel#Big {{ font-size: 26px; font-weight: 600; }}
QToolTip {{ background: {c['bg3']}; color: {c['text']}; border: 1px solid {c['line2']}; padding: 5px 7px; border-radius: 4px; }}

QPushButton {{ background: {c['bg2']}; border: 1px solid {c['line2']}; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ background: {c['bg3']}; }}
QPushButton:pressed {{ background: {c['bg4']}; }}
QPushButton:disabled {{ color: {c['text3']}; background: {c['bg1']}; border-color: {c['line']}; }}
QPushButton[kind="primary"] {{ background: {c['accent']}; border: 1px solid {c['accent']}; color: #1A0E13; font-weight: 600; }}
QPushButton[kind="primary"]:hover {{ background: {c['accent_hi']}; border-color: {c['accent_hi']}; }}
QPushButton[kind="primary"]:pressed {{ background: {c['accent_lo']}; }}
QPushButton[kind="primary"]:disabled {{ background: {c['accent_soft']}; border-color: {c['accent_soft']}; color: {c['text3']}; }}
QPushButton[kind="ghost"] {{ background: transparent; border: 1px solid transparent; }}
QPushButton[kind="ghost"]:hover {{ background: {c['bg3']}; }}
QPushButton[kind="danger"]:hover {{ border-color: {c['error']}; color: {c['error']}; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 4px; }}
QToolButton:hover {{ background: {c['bg3']}; }}
QToolButton:checked {{ background: {c['bg4']}; border-color: {c['line2']}; }}
QToolButton:disabled {{ color: {c['text3']}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {c['bg2']}; border: 1px solid {c['line2']}; border-radius: 6px; padding: 5px 8px;
    selection-background-color: {c['accent_lo']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{icon_file('chevron-down', c['text2'])}"); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{ background: {c['bg2']}; border: 1px solid {c['line2']}; selection-background-color: {c['bg4']}; outline: none; padding: 4px; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid {c['line2']}; background: {c['bg2']}; }}
QCheckBox::indicator:checked {{ background: {c['accent']}; border-color: {c['accent']};
    image: url("{icon_file('check', '#1A0E13')}"); }}

QSlider::groove:horizontal {{ height: 3px; background: {c['bg4']}; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 1px; }}
QSlider::handle:horizontal {{ background: {c['text']}; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }}
QSlider::handle:horizontal:hover {{ background: #FFFFFF; }}

QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {c['bg4']}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {c['select']}; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {c['bg4']}; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QMenu {{ background: {c['bg2']}; border: 1px solid {c['line2']}; padding: 5px; border-radius: 8px; }}
QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {c['bg4']}; }}
QMenu::item:disabled {{ color: {c['text3']}; }}
QMenu::separator {{ height: 1px; background: {c['line2']}; margin: 4px 6px; }}

QProgressBar {{ background: {c['bg2']}; border: none; border-radius: 3px; height: 6px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}

QTableWidget, QTreeWidget, QListWidget {{ background: {c['bg1']}; border: 1px solid {c['line']}; border-radius: 8px;
    gridline-color: {c['line']}; outline: none; alternate-background-color: #212121; }}
QTableWidget::item, QTreeWidget::item {{ padding: 6px 8px; border: none; }}
QTableWidget::item:selected, QTreeWidget::item:selected, QListWidget::item:selected {{ background: {c['bg4']}; color: {c['text']}; }}
QHeaderView::section {{ background: {c['bg1']}; color: {c['text2']}; border: none; border-bottom: 1px solid {c['line']};
    padding: 7px 8px; font-weight: 600; font-size: 11px; }}
QTableCornerButton::section {{ background: {c['bg1']}; border: none; }}

QSplitter::handle {{ background: {c['line']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QDialog {{ background: {c['bg1']}; }}
QMessageBox {{ background: {c['bg1']}; }}
QGroupBox {{ border: 1px solid {c['line']}; border-radius: 8px; margin-top: 14px; padding: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {c['text2']}; }}
"""


def apply(app: QApplication) -> None:
    load_fonts()
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, qc("bg0"))
    pal.setColor(QPalette.WindowText, qc("text"))
    pal.setColor(QPalette.Base, qc("bg2"))
    pal.setColor(QPalette.AlternateBase, qc("bg1"))
    pal.setColor(QPalette.Text, qc("text"))
    pal.setColor(QPalette.Button, qc("bg2"))
    pal.setColor(QPalette.ButtonText, qc("text"))
    pal.setColor(QPalette.Highlight, qc("accent_lo"))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ToolTipBase, qc("bg3"))
    pal.setColor(QPalette.ToolTipText, qc("text"))
    pal.setColor(QPalette.PlaceholderText, qc("text3"))
    pal.setColor(QPalette.Disabled, QPalette.Text, qc("text3"))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, qc("text3"))
    app.setPalette(pal)
    app.setFont(font(12))
    app.setStyleSheet(stylesheet())
