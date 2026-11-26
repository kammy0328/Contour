"""The frames of a pass over a cut read ahead of the work on them (1.27).

A pass of the analysis asked for a frame, waited for it to be decoded (the original read by ffmpeg, turned into the display
view: 0.2 to 0.25 s for a 4K log frame), worked on it, and asked for the next: the GPU stood while the frame was decoded and
the decoding stood while the GPU worked. Here a thread of its own reads the frames in the pass's order, a few ahead, and with
`extra` also does the first thing the pass does with every frame (the SAM pass: the whole frame's encoding), so the two go
side by side. What the pass gets is what it would have read and made itself: the same frame, the same function on it.

    with FramesAhead(proxy, a, b, extra=fn) as fa:
        for t in range(a, b):
            rgb, x = fa.get(t)

- frames are asked in rising order; a frame that was not read ahead (asked out of order, the reader ended) is read on the spot;
- an error while reading or in `extra` is raised by get() at that frame, as the pass would have met it;
- close() (the end of the `with`, also on an error or a cancel) ends the thread; a frame it was working on is finished
  first, so nothing of it runs on after the pass.
N (settings of this module): frames ahead; 0 reads every frame on the spot, as before 1.27."""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional, Tuple

import numpy as np

from .exposure import analysis_view

N = 2                   # frames read ahead of the one a pass works on (0: none; DIAP_AHEAD)


def frames_ahead() -> int:
    import os
    v = os.environ.get("DIAP_AHEAD", "")
    try:
        return max(0, int(v)) if v != "" else int(N)
    except ValueError:
        return int(N)


_ANALYSIS = object()


class FramesAhead:
    def __init__(self, proxy, a: int, b: int, n: Optional[int] = None, extra: Optional[Callable] = None, view=_ANALYSIS):
        """view: what is done to every frame read (the analysis's view of dark and colour-lit frames unless told otherwise;
        None: the frame as it is). extra(t, frame as viewed): made in the reader's thread too, given with the frame."""
        self.proxy, self.a, self.b = proxy, int(a), int(b)
        self.n = frames_ahead() if n is None else max(0, int(n))
        self.extra, self.view = extra, (analysis_view if view is _ANALYSIS else view)
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, self.n))
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._at = self.a                    # the next frame the reader's queue will give
        self.hits = 0
        self.direct = 0
        if self.n > 0 and self.b - self.a > 1:
            self._th = threading.Thread(target=self._run, name="frames-ahead", daemon=True)
            self._th.start()

    # ------------------------------------------------------------------ the reader's thread
    def _one(self, t: int) -> Tuple[np.ndarray, object]:
        rgb = np.asarray(self.proxy.frame(t))
        if self.view is not None:
            rgb = self.view(rgb)
        x = self.extra(t, rgb) if self.extra is not None else None
        return rgb, x

    def _run(self) -> None:
        for t in range(self.a, self.b):
            if self._stop.is_set():
                return
            try:
                item = (t, self._one(t), None)
            except BaseException as e:           # raised by get() at this frame; nothing is read after it
                item = (t, None, e)
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue
            if item[2] is not None:
                return

    # ------------------------------------------------------------------ the pass
    def get(self, t: int) -> Tuple[np.ndarray, object]:
        """(the frame as the pass sees it, extra(t, frame) or None)."""
        t = int(t)
        th = self._th
        while th is not None and self._at <= t:
            try:
                ft, val, err = self._q.get(timeout=0.25)
            except queue.Empty:
                if not th.is_alive() and self._q.empty():
                    break                        # the reader ended (its last frame, an error already given, close())
                continue
            self._at = ft + 1
            if ft < t:
                continue                         # a frame the pass skipped
            if ft > t:
                break                            # asked out of order: read on the spot (the reader's frame is dropped)
            if err is not None:
                raise err
            self.hits += 1
            return val
        self.direct += 1
        return self._one(t)

    def close(self) -> None:
        self._stop.set()
        th, self._th = self._th, None
        if th is not None:
            try:
                while True:
                    self._q.get_nowait()         # room for a put that is waiting, so the thread sees the stop at once
            except queue.Empty:
                pass
            th.join(timeout=30.0)

    def __enter__(self) -> "FramesAhead":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
