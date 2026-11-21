"""Keeps the open cut's person matte in step with the edits (instruction 2026-11-19 §4).

An edit that changes what the matte is prompted with (a corrected joint once it is let go, people merged, a joint
hidden or shown) marks the frames whose prompts it changed (track/matte_refresh.changed: a swap or a split of two
track ids moves nobody and marks nothing). A moment after the last edit the marked frames of the cut are matted
again in the background, while the preview keeps the old matte; the new alpha, owners and outlines are written
together, on the edit worker (after any re-solve queued before). A result that an edit or a job started after it
has outdated (or a cut left meanwhile) is dropped: the marks stay and the next run takes them. The marks are kept
with the cut (status 'matte_dirty'), so a cut closed before its refresh catches up when it is opened, and a render
of it asks first. Moving a slider never marks anything."""
from __future__ import annotations

import threading
import traceback
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from ..track import matte_refresh as MR

DELAY_MS = 1500
TRACK_TAGS = ("resolve", "resolve_h")      # edit worker tasks that change the tracking


def pending(cut) -> List[List[int]]:
    """The frames of a cut still to be matted again ([[a, b), ...])."""
    return [list(r) for r in ((getattr(cut, "status", None) or {}).get("matte_dirty") or [])] if cut is not None \
        else []


def clear(cut, done=None) -> None:
    """Marks taken off (all of them, or the frames in done)."""
    st = dict(cut.status or {})
    left = MR.subtract(pending(cut), done) if done is not None else []
    if left:
        st["matte_dirty"] = left
    else:
        st.pop("matte_dirty", None)
    cut.status = st


class MatteRefresher(QObject):
    changed = Signal(str)                      # cut id: its refresh state changed (for the inspector line)
    _computed = Signal(object)                 # from the worker thread

    def __init__(self, win, delay_ms: int = DELAY_MS):
        super().__init__(win)
        self.win = win
        self.gen: Dict[str, int] = {}
        self.models: dict = {}                 # SAM kept loaded between runs
        self.running: Optional[dict] = None    # {cut, gen, cancel, tracker, ranges}
        self.committing = 0                    # commits queued on the edit worker
        self._lock = threading.Lock()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(delay_ms)
        self.timer.timeout.connect(self._go)
        self._computed.connect(self._done)
        self.enabled = True
        self.last_error = ""

    # ------------------------------------------------------------------ marks
    pending = staticmethod(pending)

    def mark(self, cut, a: int, b: int) -> None:
        self.mark_ranges(cut, [[a, b]])

    def mark_ranges(self, cut, ranges) -> None:
        rs = [[int(a), int(b)] for a, b in (ranges or []) if int(b) > int(a)]
        if cut is None or not rs:
            return
        cut.status = dict(cut.status or {}, matte_dirty=MR.merge(pending(cut) + rs))
        self.hold(cut)
        if self.win.session.cut is cut:
            self.win.session.dirty = True
        self.changed.emit(cut.id)
        self.timer.start()

    def hold(self, cut=None) -> None:
        """Something is about to change the tracking of the cut (an edit queued, a job started): a refresh made
        before it is not used."""
        cut = cut if cut is not None else self.win.session.cut
        if cut is None:
            return
        self.gen[cut.id] = self.gen.get(cut.id, 0) + 1
        r = self.running
        if r is not None and r["cut"] == cut.id:
            r["cancel"] = True

    def state(self, cut) -> str:
        """'' (up to date) / 'waiting' / 'running'."""
        if cut is None:
            return ""
        if self.running is not None and self.running["cut"] == cut.id and not self.running["cancel"]:
            return "running"
        return "waiting" if pending(cut) else ""

    def kick(self) -> None:
        """The open cut has marks (opened again with some): refresh them soon. What no mark says (an edit whose
        marks were never saved, a refresh that was taken back when the app ended, an edit of a kind that marks
        nothing itself) is told from what is recorded with the matte (track/matte_rev: the joints each frame's
        matte was made from), on the edit worker: frames that no longer match are marked like any edit's."""
        s = self.win.session
        cut, tr = s.cut, s.tracker
        if cut is None:
            return
        if pending(cut):
            self.timer.start()
        if not self.enabled or tr is None or tr.store is None or not getattr(cut, "analyzed", False):
            return

        def check():
            from ..track import matte_rev as MRV
            try:
                if not tr.has_seg():
                    return None
                MRV.ensure_prompts(tr)
                return dict(cut=cut.id, marks=MR.stale(tr), tracker=tr)
            except Exception:
                traceback.print_exc()
                return None
        self.win.edit_worker.submit(("matte_check", cut.id), check)

    def checked(self, res) -> None:
        """The edit worker compared the tracking with what the matte was made from."""
        if not res or not res.get("marks"):
            return
        s = self.win.session
        if s.cut is None or s.cut.id != res["cut"] or s.tracker is not res["tracker"]:
            return
        self.mark_ranges(s.cut, res["marks"])

    def cancel(self) -> None:
        """Leaving the cut (or the project): a running refresh is dropped, its marks stay with the cut."""
        self.timer.stop()
        r = self.running
        if r is not None:
            r["cancel"] = r["dropped"] = True
            self.gen[r["cut"]] = self.gen.get(r["cut"], 0) + 1

    def before_job(self) -> None:
        """A job is starting (merge, rebuild, render...): nothing of a refresh may be written while it runs."""
        self.cancel()
        self.hold()
        if self.committing:
            self.win.edit_worker.drain(15.0)

    # ------------------------------------------------------------------ run
    def _busy(self) -> bool:
        w = self.win
        return (getattr(w, "job", None) is not None or w.batch.busy or w.edit_worker.open_tasks() > 0
                or getattr(w, "_opening", None) is not None)

    def _go(self) -> None:
        if not self.enabled:
            return
        s = self.win.session
        cut, tr = s.cut, s.tracker
        if self.running is not None:
            self.timer.start()                  # the dropped run is still finishing its frame
            return
        if cut is None or tr is None or tr.store is None:
            return
        rs = pending(cut)
        if not rs:
            return
        if self._busy():
            self.timer.start()
            return
        if not tr.has_seg():
            # an RVM matte does not follow the tracking: nothing to refresh
            clear(cut)
            self.changed.emit(cut.id)
            return
        try:
            from .modelsdlg import needed_text
            need = needed_text(cut)
        except Exception:
            need = ""
        if need:
            # analysed with model package versions that are not on this computer: the marks stay, the matte in
            # use stays, and nothing is made with another version (the cut's note says what to install or do)
            if self.last_error != need:
                self.last_error = need
                self.changed.emit(cut.id)
            return
        job = dict(cut=cut.id, gen=self.gen.get(cut.id, 0), cancel=False, tracker=tr, ranges=rs)
        self.running = job
        self.changed.emit(cut.id)

        def work():
            res = err = None
            try:
                res = MR.compute(tr, rs, cancel=lambda: job["cancel"], models=self.models)
            except Exception as e:
                if type(e).__name__ != "Cancelled":
                    err = f"{e}"
                    traceback.print_exc()
            self._computed.emit((job, res, err))
        threading.Thread(target=work, name="diap-matte-refresh", daemon=True).start()

    def _stale(self, job) -> bool:
        s = self.win.session
        return (job["cancel"] or s.cut is None or s.cut.id != job["cut"] or s.tracker is not job["tracker"]
                or self.gen.get(job["cut"], 0) != job["gen"] or getattr(self.win, "job", None) is not None)

    def _done(self, payload) -> None:
        job, res, err = payload
        self.running = None
        w = self.win
        stale = self._stale(job)
        if err and not stale:
            self.last_error = err
            w.toast(f"인물 매트를 다시 만들지 못해 이전 매트를 그대로 씁니다: {err[:120]}", kind="warn")
        if stale or err or res is None:
            self.changed.emit(job["cut"])
            if stale and not job.get("dropped") and w.session.cut is not None and w.session.cut.id == job["cut"]:
                self.timer.start()              # an edit came meanwhile: its marks include these frames
            return
        self.last_error = ""
        tr, gen, cid, marks = job["tracker"], job["gen"], job["cut"], job["ranges"]
        s = w.session

        def commit():
            # on the edit worker, after the re-solves queued before it; checked once more there
            try:
                if self.gen.get(cid, 0) != gen or s.tracker is not tr:
                    return None
                try:
                    wrote = MR.commit(tr, res)
                except Exception as e:
                    # the revision before is in use again (matte_rev.activate put it back): the marks stay
                    traceback.print_exc()
                    return dict(cut=cid, marks=[], wrote=[], error=f"{e}"[:200])
                cl = (res.get("stage") or {}).get("cloth") or {}
                return dict(cut=cid, marks=marks, wrote=wrote, cloth_stale=len(cl.get("stale") or []),
                            cloth_error=cl.get("error", ""))
            finally:
                with self._lock:
                    self.committing -= 1
        with self._lock:
            self.committing += 1
        w.edit_worker.submit(("matte", cid), commit)

    def committed(self, res) -> None:
        """The edit worker wrote a refresh: its frames are no longer marked (marks made since stay)."""
        self.changed.emit(res["cut"] if res else "")
        if not res:
            cut = self.win.session.cut
            if cut is not None and pending(cut):
                self.timer.start()
            return
        if res.get("error"):
            # not tried again by itself (it would fail the same way every second and a half): the marks stay, the
            # next edit, opening or render takes them
            self.last_error = res["error"]
            self.win.toast(f"인물 매트를 적용하지 못해 이전 매트를 그대로 씁니다: {res['error'][:120]}", kind="warn")
            return
        prj = self.win.session.project
        cut = prj.cut(res["cut"]) if prj is not None else None
        if cut is None:
            return
        if res.get("cloth_stale"):
            self.win.toast(f"인물 매트가 바뀐 {res['cloth_stale']}프레임의 옷 영역을 다시 만들지 못했습니다. "
                           "그 구간은 다시 만들 때까지 옷 그대로가 적용되지 않습니다"
                           + (f" ({res['cloth_error'][:80]})" if res.get("cloth_error") else ""), kind="warn")
        clear(cut, res["marks"])
        self.win.session.dirty = True
        self.changed.emit(cut.id)
        if pending(cut):
            self.timer.start()
