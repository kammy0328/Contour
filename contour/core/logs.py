"""Log files: what DIAP prints, its errors, crashes and stalls of the screen, in <data folder>\\logs.

DIAP.exe is a windowed Python: without this, everything the app prints (handled errors with their traceback, model
fallbacks, warnings) goes nowhere and a problem on the user's PC leaves no trace to look at.

- output.log   everything printed (stdout and stderr), also the handled errors
- errors.log   unhandled errors (the error box), from any thread
- crash.log    a hard crash of Python itself (native code: GPU driver, onnxruntime, Qt): the stack of every thread
- freeze.log   the screen did not answer for FREEZE_S seconds: the stack of every thread at that moment, and how long
               the stall lasted once the screen answers again

Each file is started again (the old one kept as .1) once it is over MAX_BYTES."""
from __future__ import annotations

import faulthandler
import io
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

MAX_BYTES = 8 * 1024 * 1024
FREEZE_S = 5.0
_files: dict = {}
_started = False


def folder() -> Path:
    try:
        from .settings import settings
        d = settings().data_path / "logs"
    except Exception:
        d = Path.home() / "DIAP_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rotate(p: Path) -> None:
    try:
        if p.exists() and p.stat().st_size > MAX_BYTES:
            old = p.with_name(p.stem + ".1" + p.suffix)
            try:
                old.unlink()
            except OSError:
                pass
            p.rename(old)
    except OSError:
        pass


def _open(name: str):
    f = _files.get(name)
    if f is None:
        p = folder() / name
        _rotate(p)
        f = open(p, "a", encoding="utf-8", errors="replace", buffering=1)
        _files[name] = f
    return f


def path(name: str) -> Path:
    return folder() / name


def write(name: str, text: str) -> None:
    try:
        f = _open(name)
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n{text.rstrip()}\n")
        f.flush()
    except Exception:
        pass


class _Tee(io.TextIOBase):
    """stdout / stderr into output.log, and to the console as well when there is one (run_debug.bat)."""

    def __init__(self, console, f):
        self.console = console
        self.f = f
        self._lock = threading.Lock()

    def write(self, s):
        if not s:
            return 0
        with self._lock:
            try:
                self.f.write(s)
            except Exception:
                pass
            if self.console is not None:
                try:
                    self.console.write(s)
                except Exception:
                    pass
        return len(s)

    def flush(self):
        for x in (self.f, self.console):
            try:
                if x is not None:
                    x.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        return self.f.fileno()

    @property
    def buffer(self):
        return getattr(self.console, "buffer", None) or self.f.buffer

    @property
    def encoding(self):
        return "utf-8"


def _thread_hook(args) -> None:
    if args.exc_type is SystemExit:
        return
    text = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    name = getattr(args.thread, "name", "?")
    write("errors.log", f"thread {name}\n{text}")
    try:
        sys.stderr.write(f"[thread {name}] {text}")
    except Exception:
        pass


def setup(version: str = "") -> None:
    """First thing at start: output into output.log, crash and thread error hooks."""
    global _started
    if _started:
        return
    _started = True
    try:
        out = _open("output.log")
        out.write(f"\n===== DIAP {version} start {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} "
                  f"python {sys.version.split()[0]} =====\n")
        sys.stdout = _Tee(sys.stdout, out)
        sys.stderr = _Tee(sys.stderr, out)
    except Exception:
        pass
    try:
        crash = _open("crash.log")
        faulthandler.enable(file=crash, all_threads=True)
    except Exception:
        pass
    try:
        threading.excepthook = _thread_hook
    except Exception:
        pass


class Watchdog(threading.Thread):
    """Notes in freeze.log where every thread is when the screen stops answering (the UI thread misses its beat)."""

    def __init__(self, limit: float = FREEZE_S):
        super().__init__(name="diap-watchdog", daemon=True)
        self.limit = float(limit)
        self.last = time.monotonic()
        self.stalled_since: Optional[float] = None
        self.dumps = 0
        self._halt = threading.Event()

    def beat(self) -> None:
        now = time.monotonic()
        if self.stalled_since is not None:
            took = now - self.stalled_since
            write("freeze.log", f"화면 응답 복귀: {took:.1f}초 멈춤")
            try:
                sys.stderr.write(f"FREEZE {took:.1f}s (stacks in freeze.log)\n")
            except Exception:
                pass
            self.stalled_since = None
            self.dumps = 0
        self.last = now

    def _dump(self, gap: float) -> None:
        try:
            f = _open("freeze.log")
            f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} === 화면 응답 없음 {gap:.1f}초, 모든 스레드:\n")
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)
            f.flush()
        except Exception:
            pass

    def run(self) -> None:
        while not self._halt.wait(1.0):
            gap = time.monotonic() - self.last
            if gap < self.limit:
                continue
            if self.stalled_since is None:
                self.stalled_since = self.last
                self._dump(gap)
                self.dumps = 1
            elif self.dumps < 3 and gap > self.limit * (1 + 3 * self.dumps):
                # still stuck: once more later (the same place again means it is stuck there, not just slow)
                self._dump(gap)
                self.dumps += 1

    def stop(self) -> None:
        self._halt.set()


_watchdog: Optional[Watchdog] = None


def start_watchdog(parent=None) -> Optional[Watchdog]:
    """Starts the beat (a Qt timer on the UI thread) and the watching thread."""
    global _watchdog
    if _watchdog is not None:
        return _watchdog
    try:
        from PySide6.QtCore import QTimer
        wd = Watchdog()
        t = QTimer(parent)
        t.setInterval(500)
        t.timeout.connect(wd.beat)
        t.start()
        wd._timer = t
        wd.start()
        _watchdog = wd
        return wd
    except Exception:
        return None
