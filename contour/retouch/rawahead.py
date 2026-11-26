"""The masks of the next frames made ahead in threads of this process, for the render (1.27).

The render's serial stage is the preparation of a frame (warp.render: one frame at a time, the others wait on it), and four
fifths of it is the frame's own masks (engine._prepare_raw: nothing but the frame itself goes into them: its pictures, the tracks,
the values). Here the masks of the frames the render will ask for next are made by one or two threads while it steadies and
hands on the frame it has, and engine._raw_cached takes them from here before it would make them itself. The same function on
the same inputs: LNGSHOT 024 (4 faces, 4K LogC4), 36 frames, 8.67 -> 7.91 s a frame, the serial stage 282 -> 126 s, every frame
of the file the same (j2886).

It costs memory (a frame's masks in the making are a gigabyte or so) and cores, so core/perf.py asks for it only on a computer
with both; CONTOUR_RAW_AHEAD ("frames" or "frames:threads", 0: off) says otherwise, and 설정 render_parallel off switches it
off with the rest. And only while the computer has memory to spare at that moment (perf.RAW_AHEAD_MIN_FREE_GB free: the other
programs' share is not known beforehand); a frame not made ahead is made by the render as before. Not used beside the helper
processes of retouch.rawpool (CONTOUR_RAW_PROCS), nor by the preview.

A frame that could not be made here (any error) is made by the render as before, which then meets the error itself."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional


class RawAhead:
    def __init__(self, cut, tracker, lo: int, hi: int, ahead: int, threads: int):
        self.cut, self.tracker = cut, tracker
        self.lo, self.hi = int(lo), int(hi)             # frames lo..hi-1 are the ones a render of this range asks for
        self.ahead = max(1, int(ahead))
        self.threads = max(1, int(threads))
        self._ex = ThreadPoolExecutor(max_workers=self.threads, thread_name_prefix="raw-ahead")
        self._fut: Dict[int, object] = {}
        self._asked: set = set()
        self._lock = threading.Lock()
        self._closed = False
        self.hit = 0
        self.miss = 0
        self.failed = 0
        self.held = 0                       # times nothing was started ahead because memory was short

    def _make(self, s: int):
        from . import engine as E
        rf = E._prepare_raw(self.cut, self.tracker, s, None)
        if rf is not None:
            for j in rf.jobs:               # kept as engine._raw_cached keeps a frame it made itself (a waiting frame is small)
                j.masks = {k: E._raw_keep(v) for k, v in j.masks.items()}
        return rf

    @staticmethod
    def _room() -> bool:
        try:
            from ..core import perf
            need = float(perf.RAW_AHEAD_MIN_FREE_GB)
            if need <= 0:
                return True
            free = perf.free_gb()
            return free < 0 or free >= need          # (cannot be read: as the installed memory said)
        except Exception:
            return True

    def take(self, s: int):
        """(made, the frame's masks): made False when frame s was not made ahead (the caller makes it). Asks for the frames
        after it. Called by the render's serial stage, one call at a time."""
        s = int(s)
        with self._lock:
            if self._closed:
                return False, None
            f = self._fut.pop(s, None)
            self._asked.add(s)
            # only while the computer has memory to spare: a frame in the making is a gigabyte or so, and what the other
            # programs leave is not what is installed (a frame not made ahead is made by the render, as before 1.27)
            room = self._room()
            if not room:
                self.held += 1
            for s2 in range(s + 1, s + 1 + self.ahead):
                if room and self.lo <= s2 < self.hi and s2 not in self._asked:
                    self._asked.add(s2)
                    self._fut[s2] = self._ex.submit(self._make, s2)
        if f is None:
            self.miss += 1
            return False, None
        try:
            rf = f.result()
        except Exception:
            self.failed += 1
            return False, None
        self.hit += 1
        return True, rf

    def close(self) -> dict:
        with self._lock:
            self._closed = True
            futs, self._fut = list(self._fut.values()), {}
        for f in futs:
            f.cancel()
        self._ex.shutdown(wait=True)
        return dict(hit=self.hit, miss=self.miss, failed=self.failed, held=self.held, ahead=self.ahead, threads=self.threads)


def start_for_render(cut, tracker, a: int, b: int) -> Optional[RawAhead]:
    """Set up for a render of frames a..b-1 when this computer has the memory and the cores (core.perf.raw_ahead) and the cut has
    tracks; None otherwise. engine._raw_cached finds it on the tracker; stop_for_render takes it off again."""
    try:
        from ..core import perf
        from . import engine as E
        st = getattr(tracker, "store", None)
        if st is None:
            return None
        people = sum(1 for p in cut.targets() if st.has(p.id))
        n, k = perf.raw_ahead(people)
        if n <= 0 or b - a < 3:
            return None
        N = int(st.final.shape[1])
        ra = RawAhead(cut, tracker, max(0, int(a) - int(E.STEADY_R)), min(N, int(b) + int(E.STEADY_R)), n, k)
        tracker.__dict__["_raw_ahead"] = ra
        return ra
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def stop_for_render(tracker) -> Optional[dict]:
    ra = tracker.__dict__.pop("_raw_ahead", None)
    if ra is None:
        return None
    try:
        return ra.close()
    except Exception:
        return None
