"""One thumbnail per cut, shared by the cut list and the timeline strip (1.22.1).

Opening a project used to read every cut's thumbnail (a proxy check and a JPG or a frame per cut) before the
project was shown, and the timeline strip read the same JPGs again on the UI thread. Now the project opens with
empty thumbnail boxes and one background worker fills them:
- the saved JPG (timeline/thumbs/<cut id>.jpg) is read once and shared by both views;
- a cut with no JPG (or one made for another original / colour view) gets one made from its proxy or original,
  saved for next time, but only once it is on screen (or open): never every cut of a big project at once;
- every request belongs to one project opening (gen): what an earlier opening's work brings back is dropped.
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

THUMB_W = 192
INDEX = "index.json"
TIMELINE = "timeline"            # sig of a JPG made by the timeline import (a still of the baked edit)


def thumb_frame(cut) -> int:
    return max(0, int(cut.info.frames) // 2)


def cut_sig(project, cut, vs=None) -> str:
    """What a made thumbnail depends on: the original file, the colour view and the frame shown."""
    from ..core.frames import _sig
    if vs is None:
        from ..warp.render import _vs_for
        vs = _vs_for(cut)
    return f"{_sig(cut.info)}|{vs.key()}|{thumb_frame(cut)}"


def read_jpg(project, cut_id: str) -> Optional[np.ndarray]:
    from ..core.tlimport import load_thumb
    return load_thumb(project, cut_id)


def make_thumb(project, cut, vs=None) -> Optional[np.ndarray]:
    """The cut's middle frame from its proxy (in this view) or its original; None when neither is there."""
    from ..core.frames import frames_for
    fr = frames_for(project, cut, vs)
    if fr is None:
        return None
    try:
        return fr.thumbnail(thumb_frame(cut), THUMB_W)
    finally:
        try:
            fr.close()
        except Exception:
            pass


def _dir(project) -> Path:
    from ..core.tlimport import thumbs_dir
    return thumbs_dir(project)


def _write_index(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


class ThumbCache(QObject):
    """images: cut id -> QImage, the one dict both views paint from. ready(cut id) after one arrived."""
    ready = Signal(str)
    _arrived = Signal(int, str, object, str)      # gen, cut id, QImage, sig (from the worker)

    def __init__(self, parent=None, reader: Callable = None, maker: Callable = None, sig: Callable = None):
        super().__init__(parent)
        self.reader = reader or read_jpg
        self.maker = maker or make_thumb
        self.sig = sig or cut_sig
        self.images: Dict[str, object] = {}
        self.gen = 0
        self.project = None
        self.vs_for: Callable = None             # cut -> ViewSettings (the session's); None: the cut's own
        self._cv = threading.Condition()
        self._queue: "OrderedDict[str, str]" = OrderedDict()   # cut id -> 'read' | 'make'
        self._done: Dict[str, str] = {}          # cut id -> what was done this opening ('read' | 'make')
        self._have: set = set()                  # cut ids with a picture this opening (set by the worker)
        self._working = False
        self._index: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None
        self._quit = False
        self.reads = 0                           # JPG reads (tests and timing)
        self.made = 0
        self._arrived.connect(self._on_arrived)

    # ------------------------------------------------------------------ UI thread
    def reset(self, project) -> int:
        """A project was opened (or closed: None): a new opening, nothing of the old one is kept or shown."""
        with self._cv:
            self.gen += 1
            self.project = project
            self._queue.clear()
            self._done.clear()
            self._have = set()
            self._index = None
            self.images.clear()
            self._cv.notify_all()
        return self.gen

    def want(self, ids: Iterable[str], make: bool = False, front: bool = False) -> None:
        """Thumbnails for these cuts: read the saved JPG; make: also make one when there is none (or it is stale).
        front: before everything queued (the cuts on screen)."""
        ids = [i for i in ids if i]
        if not ids or self.project is None:
            return
        with self._cv:
            for cid in (reversed(ids) if front else ids):
                had = self._done.get(cid)
                job = "make" if make else "read"
                if had == "make" or (had == "read" and job == "read"):
                    continue
                old = self._queue.get(cid)
                if old is not None and not front:
                    if job == "make":
                        self._queue[cid] = "make"        # keeps its place in line
                    continue
                self._queue[cid] = "make" if old == "make" else job
                if front:
                    self._queue.move_to_end(cid, last=False)
            self._ensure_thread()
            self._cv.notify_all()

    def refresh(self, cut_id: str, front: bool = True) -> None:
        """This cut's thumbnail may be stale (original linked again, colour view changed): checked and remade if so."""
        with self._cv:
            self._done.pop(cut_id, None)
        self.want([cut_id], make=True, front=front)

    def drop(self, cut_ids: Iterable[str] = None) -> None:
        """Forget thumbnails (a timeline imported again wrote new JPGs): read again when next wanted."""
        with self._cv:
            ids = list(self.images) if cut_ids is None else list(cut_ids)
            for cid in ids:
                self.images.pop(cid, None)
                self._done.pop(cid, None)
                self._have.discard(cid)
            if cut_ids is None:
                self._done.clear()
                self._index = None

    def pending(self) -> int:
        with self._cv:
            return len(self._queue)

    def idle(self) -> bool:
        with self._cv:
            return not self._queue and not self._working

    def stop(self) -> None:
        with self._cv:
            self._quit = True
            self._queue.clear()
            self._cv.notify_all()
        th = self._thread
        if th is not None:
            th.join(timeout=2.0)

    def _on_arrived(self, gen: int, cid: str, img, sig: str) -> None:
        if gen != self.gen:
            return                               # an earlier opening's (or a closed project's) thumbnail
        if img is None:
            self.images.pop(cid, None)
        else:
            self.images[cid] = img
        self.ready.emit(cid)

    # ------------------------------------------------------------------ worker
    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._quit = False
            self._thread = threading.Thread(target=self._run, name="diap-thumbs", daemon=True)
            self._thread.start()

    def _load_index(self, project) -> dict:
        if self._index is None:
            try:
                self._index = json.loads((_dir(project) / INDEX).read_text(encoding="utf-8"))
            except Exception:
                self._index = {}
        return self._index

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._quit:
                    self._cv.wait(timeout=5.0)
                    if not self._queue and not self._quit:
                        self._thread = None
                        return
                if self._quit:
                    self._thread = None
                    return
                cid, job = self._queue.popitem(last=False)
                gen, project, vs_for, have_set = self.gen, self.project, self.vs_for, self._have
                self._done[cid] = job
                self._working = True
                index = self._load_index(project) if project is not None else {}
            try:
                if project is not None:
                    self._one(gen, project, cid, job, index, vs_for, have_set)
            except Exception:
                traceback.print_exc()
            finally:
                with self._cv:
                    self._working = False

    def _one(self, gen, project, cid, job, index, vs_for, have_set) -> None:
        from .viewer import np_to_qimage
        cut = project.cut(cid)
        if cut is None:
            return
        have = cid in have_set                   # read (or made) already: never decoded twice
        rgb = None
        if not have:
            self.reads += 1
            rgb = self.reader(project, cid)
            if rgb is not None:
                have_set.add(cid)
                self._emit(gen, cid, rgb, np_to_qimage)
        if job != "make":
            return
        saved = index.get(cid)
        if saved is None and cut.tl and (have or rgb is not None):
            return                               # the timeline import's still: kept until imported again
        if saved == TIMELINE:
            return
        vs = vs_for(cut) if vs_for else None
        want = self.sig(project, cut, vs)
        if (have or rgb is not None) and saved == want:
            return
        if gen != self.gen:
            return
        made = self.maker(project, cut, vs)
        if made is None:
            return
        self.made += 1
        if gen != self.gen:
            return                               # made for a project closed meanwhile: not saved into it either
        try:
            from ..core.tlimport import save_thumb
            save_thumb(project, cid, made)
            with self._cv:
                if gen == self.gen:
                    index[cid] = want
                    _write_index(_dir(project) / INDEX, index)
        except Exception:
            traceback.print_exc()
        have_set.add(cid)
        self._emit(gen, cid, made, np_to_qimage)

    def _emit(self, gen, cid, rgb, np_to_qimage) -> None:
        if gen != self.gen:
            return
        img = np_to_qimage(np.ascontiguousarray(rgb)).copy()
        self._arrived.emit(gen, cid, img, "")


def visible_ids(lw, ahead: float = 1.0) -> tuple:
    """(ids on screen, plus about one screen ahead on both sides; every listed id) of a QListWidget of cuts."""
    from PySide6.QtCore import Qt
    vp = lw.viewport().rect()
    dx, dy = int(vp.width() * ahead), int(vp.height() * ahead)
    near = vp.adjusted(-dx, -dy, dx, dy)
    on, close, every = [], [], []
    for i in range(lw.count()):
        it = lw.item(i)
        c = it.data(Qt.UserRole)
        cid = getattr(c, "id", None)
        if cid is None or it.isHidden():
            continue
        every.append(cid)
        r = lw.visualItemRect(it)
        if r.intersects(vp):
            on.append(cid)
        elif r.intersects(near):
            close.append(cid)
    return on + close, every


class VisibleWatch(QObject):
    """Calls back (wanted ids, every id) a moment after a list of cuts was filled, scrolled, resized or shown."""

    def __init__(self, lw, callback: Callable, delay_ms: int = 40):
        super().__init__(lw)
        from PySide6.QtCore import QTimer
        self.lw, self.callback = lw, callback
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(delay_ms)
        self.timer.timeout.connect(self._fire)
        lw.verticalScrollBar().valueChanged.connect(self.poke)
        lw.horizontalScrollBar().valueChanged.connect(self.poke)
        lw.viewport().installEventFilter(self)

    def poke(self, *_):
        self.timer.start()

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if ev.type() in (QEvent.Resize, QEvent.Show):
            self.poke()
        return False

    def _fire(self):
        try:
            on, every = visible_ids(self.lw)
        except RuntimeError:
            return                               # the list was deleted
        if every:
            self.callback(on, every)
