"""DIAP: video beauty retouch and body reshaping for post-production (code package: contour)."""
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _log_dir() -> Path:
    from contour.core.settings import settings
    d = settings().data_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _excepthook(etype, value, tb):
    text = "".join(traceback.format_exception(etype, value, tb))
    try:
        with open(_log_dir() / "contour.log", "a", encoding="utf-8") as f:
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{text}")
    except Exception:
        pass
    sys.__stderr__ and sys.__stderr__.write(text)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            m = QMessageBox()
            m.setIcon(QMessageBox.Warning)
            m.setWindowTitle("오류")
            m.setText(f"예상하지 못한 오류가 생겼습니다: {value}")
            m.setInformativeText("작업 내용은 자동 저장됩니다. 같은 문제가 반복되면 로그 파일을 확인하세요.")
            m.setDetailedText(text)
            m.exec()
    except Exception:
        pass


def main():
    sys.excepthook = _excepthook
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QIcon
    from PySide6.QtWidgets import QApplication

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SEWOO.DIAP")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName("DIAP")
    app.setOrganizationName("SEWOO")
    app.setWindowIcon(QIcon(str(HERE / "assets" / "icon.png")))
    from contour.ui import theme
    theme.apply(app)
    from contour.ui import gatedlg
    if not gatedlg.ask():
        sys.exit(0)
    from contour.core.settings import APP_VERSION
    from contour.ui.splash import Splash
    splash = Splash(APP_VERSION)
    splash.show()
    app.processEvents()
    try:
        splash.wait_imports(app)
        splash.set_message("화면 준비 중")
        app.processEvents()
        from contour.ui.main_window import MainWindow
        w = MainWindow()
    except Exception:
        splash.close()
        raise
    w.show()
    splash.finish(w)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
