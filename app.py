"""DIAP: video beauty retouch and body reshaping for post-production (code package: contour)."""
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _excepthook(etype, value, tb):
    text = "".join(traceback.format_exception(etype, value, tb))
    try:
        from contour.core import logs
        logs.write("errors.log", text)
    except Exception:
        pass
    try:
        sys.stderr and sys.stderr.write(text)
    except Exception:
        pass
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance() is not None:
            m = QMessageBox()
            m.setIcon(QMessageBox.Warning)
            m.setWindowTitle("오류")
            m.setText(f"예상하지 못한 오류가 생겼습니다: {value}")
            m.setInformativeText("작업 내용은 자동 저장됩니다. 같은 문제가 반복되면 설정의 '로그 폴더 열기'에서 "
                                 "errors.log를 보내 주세요.")
            m.setDetailedText(text)
            m.exec()
    except Exception:
        pass


def _worker(kind: str, args) -> int:
    """A child job of the app (an analysis of the batch runner) started as `<app> --worker batch ...`: runs it and
    ends. Nothing of the window is loaded, so a packed app started this way does not open a second DIAP."""
    if kind == "batch":
        from contour import batch
        return int(batch.main(list(args)) or 0)
    if kind == "smoke":
        # the short run every system makes (tools/ci_smoke.py), from inside this install: project, proxy, the
        # models that come with the app, one cut exported
        import runpy
        f = HERE / "tools" / "ci_smoke.py"
        if not f.is_file():
            sys.stderr.write("this install has no tools/ci_smoke.py\n")
            return 2
        sys.argv = [str(f)] + list(args)
        try:
            runpy.run_path(str(f), run_name="__main__")
        except SystemExit as e:
            return int(e.code or 0)
        return 0
    sys.stderr.write(f"unknown worker: {kind}\n")
    return 2


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        sys.exit(_worker(sys.argv[2], sys.argv[3:]))
    try:
        from contour.core import logs
        from contour.core.settings import APP_VERSION as _V
        logs.setup(_V)
    except Exception:
        pass
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
        try:
            # model packages: what an install that never finished left behind is cleared (what was in use stays)
            from contour.track import modelpkg
            left = modelpkg.store().recover()
            if left:
                sys.stderr.write(f"model packages: cleared {left}\n")
        except Exception:
            pass
        splash.set_message("화면 준비 중")
        app.processEvents()
        from contour.ui.main_window import MainWindow
        w = MainWindow()
    except Exception:
        splash.close()
        raise
    w.show()
    splash.finish(w)
    try:
        from contour.core import logs
        logs.start_watchdog(w)
    except Exception:
        pass
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
