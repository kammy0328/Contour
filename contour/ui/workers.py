"""Background jobs (analysis, proxy, QC, training, render queue) on QThreads."""
from __future__ import annotations

import queue
import threading
import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal


class Job(QThread):
    progressed = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[..., Any], parent=None):
        super().__init__(parent)
        self.fn = fn
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def cancelled(self) -> bool:
        return self._cancel

    def run(self):
        try:
            res = self.fn(lambda f, m="": self.progressed.emit(float(f), str(m)), self.cancelled)
            self.succeeded.emit(res)
        except Exception as e:
            from ..track.analyze import Cancelled
            if isinstance(e, Cancelled) or self._cancel:
                self.failed.emit("__cancelled__")
            else:
                self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class SerialWorker(QThread):
    """Runs small edit-time tasks one after another (joint re-solve etc.)."""
    done = Signal(object, object)       # tag, result
    error = Signal(str)
    busy = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.q: "queue.Queue" = queue.Queue()
        self._stop = False

    def submit(self, tag, fn: Callable[[], Any]):
        self.q.put((tag, fn))
        if not self.isRunning():
            self.start()

    def pending(self) -> int:
        return self.q.qsize()

    def stop(self):
        self._stop = True
        self.q.put(None)

    def run(self):
        while not self._stop:
            item = self.q.get()
            if item is None:
                break
            tag, fn = item
            self.busy.emit(True)
            try:
                res = fn()
                self.done.emit(tag, res)
            except Exception as e:
                self.error.emit(f"{e}\n{traceback.format_exc()}")
            if self.q.empty():
                self.busy.emit(False)


class PreviewWorker(QThread):
    """Computes preview frames off the UI thread. Only the latest request matters: older ones are
    dropped, so dragging a slider never queues up work."""
    ready = Signal(object, object)       # key, image (np.ndarray)

    def __init__(self, compute: Callable[[Any], Any], parent=None):
        super().__init__(parent)
        import threading
        self.compute = compute
        self._cv = threading.Condition()
        self._want = None
        self._stop = False

    def request(self, key) -> None:
        with self._cv:
            self._want = key
            self._cv.notify()
        if not self.isRunning():
            self.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def run(self):
        while True:
            with self._cv:
                while self._want is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                key = self._want
                self._want = None
            try:
                img = self.compute(key)
            except Exception:
                traceback.print_exc()
                img = None
            if img is not None:
                self.ready.emit(key, img)


class NextCutPrep:
    """Playback going on into the next cut: that cut is opened in a session of its own on a background thread, its
    first frames decoded and as many results as time allows rendered into that session's playback cache. When the
    playhead reaches the end of the current cut, the main session adopts it (no proxy to open, nothing to wait for)."""

    def __init__(self):
        self.cut_id: Optional[str] = None
        self.session = None
        self.ready = False
        self.failed = False
        self._stop = threading.Event()

    def start(self, project, cut, head: int = 24) -> None:
        if self.cut_id == cut.id and (self.ready or not self.failed) and not self._stop.is_set():
            return
        self.cancel()
        self.cut_id, self.session, self.ready, self.failed = cut.id, None, False, False
        stop = threading.Event()
        self._stop = stop
        from .session import EditSession

        def run():
            try:
                sh = EditSession()
                sh.project = project
                sh.open_cut(cut)
                if sh.proxy is None or not sh.proxy_ready():
                    # no proxy yet (a cut never opened): made here, in the background, like opening it would
                    from ..core.colour import manager
                    if sh.proxy is not None:
                        sh.proxy.close()
                    sh.proxy, sh.tracker = None, None
                    vs = sh.view_settings(cut)
                    if cut.tl and (project.timeline or {}).get("baked"):
                        from ..core.tlimport import build_bake_proxy
                        ok = build_bake_proxy(project, manager(), vs, None, stop.is_set)
                    else:
                        ok = sh.make_proxy(cut).build(manager(), vs, None, stop.is_set)
                    if not ok or stop.is_set():
                        self.failed = not stop.is_set()
                        return
                    sh.open_cut(cut)
                    if sh.proxy is None or not sh.proxy_ready():
                        self.failed = True
                        return
                a, b = cut.work_range
                for t in range(a, min(b, a + head)):          # the originals of the first frames, decoded
                    if stop.is_set():
                        return
                    sh.frame(t)
                if stop.is_set():
                    return
                self.session = sh
                self.ready = True
                for t in range(a, b):                          # then results, as far as time allows
                    if stop.is_set():
                        return
                    key = sh.result_key(t, True)
                    img = sh.result(t, final=True)
                    if img is not None and not stop.is_set():
                        sh.play_put(t, key, img, near=a)
            except Exception:
                traceback.print_exc()
                self.failed = True

        threading.Thread(target=run, name="diap-nextcut", daemon=True).start()

    def take(self, cut_id: str):
        """The prepared session for cut_id (the background work stops), or None if it is not ready yet."""
        if self.cut_id != cut_id or not self.ready or self.session is None:
            return None
        self._stop.set()
        sh = self.session
        self.cut_id, self.session, self.ready = None, None, False
        return sh

    def cancel(self) -> None:
        self._stop.set()                 # the thread ends after the frame it is on; its session is then dropped
        self.cut_id, self.session, self.ready, self.failed = None, None, False, False


class CacheWorker(QThread):
    """Renders finished results ahead of time while nobody is editing, so playback runs at full speed.
    Starts from the frame you are on and goes forward through the work range."""
    progressed = Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        import threading
        self.s = session
        self._cv = threading.Condition()
        self._start = 0
        self._kick = 0
        self._stop = False
        self.enabled = True

    def kick(self, frame: int) -> None:
        with self._cv:
            self._start = int(frame)
            self._kick += 1
            self._cv.notify()
        if not self.isRunning():
            self.start()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    def _busy(self) -> bool:
        import time
        s = self.s
        now = time.time()
        return (s.interactive or now - s.last_touch < 0.8 or now - getattr(s, "last_seek", 0.0) < 0.8
                or getattr(s, "preview_wait", 0) > 0)

    def run(self):
        import time
        while True:
            with self._cv:
                self._cv.wait(timeout=0.5)
                if self._stop:
                    return
                start, kick = self._start, self._kick
            s = self.s
            if not self.enabled or s.cut is None or s.proxy is None or s.tracker is None or s.tracker.store is None \
                    or not s.cut.analyzed or self._busy():
                continue
            a, b = s.cut.work_range
            if b - a < 1:
                continue
            start = min(max(start, a), b - 1)
            order = list(range(start, b)) + list(range(a, start))
            n_new = 0
            for t in order:
                if self._stop or self._kick != kick or self._busy() or not self.enabled:
                    break
                if s.play_has(t):
                    continue
                key = s.result_key(t, True)
                try:
                    img = s.result(t, final=True)
                except Exception:
                    traceback.print_exc()
                    img = None
                if img is None or key != s.result_key(t, True):
                    continue
                s.play_put(t, key, img, near=start)
                n_new += 1
                if n_new % 4 == 0:
                    self.progressed.emit()
                time.sleep(0.002)
            if n_new:
                self.progressed.emit()
