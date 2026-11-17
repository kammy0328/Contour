"""Runs `python -m contour.batch analyze` for a list of cuts, one process at a time, and hands each result
back to the main window (which owns the project)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

APP_DIR = Path(__file__).resolve().parents[2]
CANCELLED = "__cancelled__"     # finishedCut error text of a cut stopped by kill()


def _tree(pid: int) -> List[int]:
    """The process and every process under it (the analysis may run ffmpeg)."""
    if sys.platform != "win32":
        try:
            import subprocess
            out = subprocess.run(["ps", "-e", "-o", "pid=,ppid="], capture_output=True, text=True).stdout
            kids = {}
            for ln in out.splitlines():
                a, b = ln.split()
                kids.setdefault(int(b), []).append(int(a))
        except Exception:
            return [pid]
    else:
        import ctypes
        from ctypes import wintypes

        class PE(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                        ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                        ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                        ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]
        k32 = ctypes.windll.kernel32
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)
        kids = {}
        try:
            e = PE()
            e.dwSize = ctypes.sizeof(PE)
            ok = k32.Process32First(snap, ctypes.byref(e))
            while ok:
                kids.setdefault(int(e.th32ParentProcessID), []).append(int(e.th32ProcessID))
                ok = k32.Process32Next(snap, ctypes.byref(e))
        finally:
            k32.CloseHandle(snap)
    out, todo = [], [pid]
    while todo:
        q = todo.pop()
        if q in out:
            continue
        out.append(q)
        todo.extend(kids.get(q, []))
    return out


def _suspend(pid: int, on: bool) -> bool:
    """Stops (on) or lets go on (off) a process and its children where they are."""
    ok = False
    for q in _tree(pid):
        try:
            if sys.platform == "win32":
                import ctypes
                h = ctypes.windll.kernel32.OpenProcess(0x0800, False, q)       # PROCESS_SUSPEND_RESUME
                if h:
                    f = ctypes.windll.ntdll.NtSuspendProcess if on else ctypes.windll.ntdll.NtResumeProcess
                    ok = (f(h) == 0) or ok
                    ctypes.windll.kernel32.CloseHandle(h)
            else:
                import os
                import signal
                os.kill(q, signal.SIGSTOP if on else signal.SIGCONT)
                ok = True
        except Exception:
            pass
    return ok


class BatchRunner(QObject):
    progressed = Signal(str, float, str)       # cut id, 0..1, message
    finishedCut = Signal(str, bool, str)       # cut id, ok, error text
    finishedAll = Signal(int, int)             # done, failed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.queue: List[str] = []
        self.proc: Optional[QProcess] = None
        self.current: Optional[str] = None
        self.project_path = ""
        self.opts = {}
        self._buf = b""
        self._err = []
        self.done = 0
        self.failed = 0
        self._stop = False
        self.paused = False
        self.before_start: Optional[Callable[[str], bool]] = None   # may veto / prepare a cut (returns False: skip)

    @property
    def busy(self) -> bool:
        return self.proc is not None or bool(self.queue)

    def start(self, project_path: str, cut_ids: List[str], accuracy: str = "high", all_targets: bool = True):
        self.project_path = project_path
        self.opts = dict(accuracy=accuracy, all_targets=all_targets)
        for c in cut_ids:
            if c not in self.queue and c != self.current:
                self.queue.append(c)
        if self.proc is None:
            self.done = self.failed = 0
            self._stop = False
            self._killed = False
            self._next()

    def stop(self):
        """No new cut is started; the running one finishes (stopping a half-written analysis loses it)."""
        self._stop = True
        self.queue.clear()

    def kill(self):
        """Stops now: the running cut's half-done analysis is dropped (the cut stays as it was)."""
        self.queue.clear()
        self._stop = True
        if self.proc is None:
            self.finishedAll.emit(self.done, self.failed)
            return
        if self.proc is not None:
            self._killed = True
            if self.paused:
                self.resume()
            self.proc.kill()

    def pause(self) -> bool:
        """Stops the running analysis where it is (the process keeps its memory); resume() goes on."""
        if self.proc is None or self.paused:
            return False
        pid = int(self.proc.processId() or 0)
        if pid and _suspend(pid, True):
            self.paused = True
        return self.paused

    def resume(self) -> None:
        if self.proc is not None and self.paused:
            pid = int(self.proc.processId() or 0)
            if pid:
                _suspend(pid, False)
        self.paused = False

    def _next(self):
        while self.queue and not self._stop:
            cid = self.queue.pop(0)
            if self.before_start is not None and not self.before_start(cid):
                continue
            self.current = cid
            self._buf = b""
            self._err = []
            self._errline = ""
            p = QProcess(self)
            p.setWorkingDirectory(str(APP_DIR))
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONIOENCODING", "utf-8")
            env.insert("PYTHONUNBUFFERED", "1")
            env.insert("PYTHONWARNINGS", "ignore")
            import os as _os
            nthr = str(max(1, (_os.cpu_count() or 4) - 2))
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
                env.insert(k, nthr)
            p.setProcessEnvironment(env)
            # messages and warnings go to a file: a full stderr pipe (while the app is busy and not reading it)
            # used to stop the analysis process in the middle of a cut
            from ..core.settings import settings as _settings
            logd = _settings().data_path / "logs"
            logd.mkdir(parents=True, exist_ok=True)
            self._errfile = logd / f"batch_{cid}.log"
            p.setStandardErrorFile(str(self._errfile))
            p.readyReadStandardOutput.connect(self._read)
            p.finished.connect(self._finished)
            exe = sys.executable
            if not Path(exe).name.lower().startswith("python.exe"):
                # pythonw.exe or the DIAP.exe launcher (a renamed pythonw): the console python next to it
                cand = Path(exe).with_name("python.exe")
                if cand.exists():
                    exe = str(cand)
            p.setProgram(exe)
            p.setArguments(["-m", "contour.batch", "analyze", self.project_path, cid,
                            "--accuracy", self.opts.get("accuracy", "high"),
                            "--all-targets", "1" if self.opts.get("all_targets", True) else "0"])
            if sys.platform == "win32":
                try:
                    # no window, below-normal priority: the computer stays responsive while it analyses
                    p.setCreateProcessArgumentsModifier(lambda a: setattr(a, "flags", a.flags | 0x08000000 | 0x00004000))
                except Exception:
                    pass
            self.proc = p
            self.progressed.emit(cid, 0.0, "시작")
            p.start()
            return
        self.proc = None
        self.current = None
        self.finishedAll.emit(self.done, self.failed)

    def _read(self):
        if self.proc is None:
            return
        self._buf += bytes(self.proc.readAllStandardOutput())
        *lines, self._buf = self._buf.split(b"\n")
        for ln in lines:
            t = ln.decode("utf-8", "replace").strip()
            if t.startswith("PROGRESS "):
                parts = t.split(" ", 2)
                try:
                    f = float(parts[1])
                except (IndexError, ValueError):
                    continue
                self.progressed.emit(self.current or "", f, parts[2] if len(parts) > 2 else "")
            elif t.startswith("ERROR "):
                self._errline = t[6:]

    def _finished(self, code, _status):
        self.paused = False
        cid = self.current
        err = ""
        try:
            ef = getattr(self, "_errfile", None)
            if ef is not None and ef.exists():
                lines = [l for l in ef.read_text(encoding="utf-8", errors="replace").splitlines()
                         if l.strip() and ("Error" in l or "error" in l or "Traceback" in l)]
                err = "\n".join(lines[-3:])
                if code == 0:
                    ef.unlink()
        except OSError:
            pass
        ok = code == 0
        killed = getattr(self, "_killed", False)
        self._killed = False
        if killed:
            ok, err, self._errline = False, "", CANCELLED
        elif ok:
            self.done += 1
        else:
            self.failed += 1
        if self.proc is not None:
            self.proc.deleteLater()
        self.proc = None
        self.current = None
        last = self._errline or (err.splitlines()[-1] if err else "")
        self.finishedCut.emit(cid or "", ok, "" if ok else last)
        self._next()
