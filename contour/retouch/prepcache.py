"""The render's analysis of each frame kept on disk, so a cut whose grade is replaced is rendered again with the retouch stage alone (11/28).

A render does two things to every frame: it prepares it (engine.prepare: the masks of every person, the flaws found, the values; made from the
analysis picture, the tracks and the values, and held steady over the frames round it) and then retouches the full picture with what was prepared
(engine.apply_planes) before the warp. Here what prepare gives is written to a folder, frame by frame (save_cut: the preparing alone, no picture
read, retouched or written), and a later render of the same cut with the same values takes it from there (Reader) instead of preparing the frame
again: only the retouch, the warp and the writing are left.

What is kept is the prepared frame as the render would have used it, every mask in its own type (float32 or float16) and to the bit: a box round its
non-zero part and zlib (_pack). So a render from the folder gives the file a render with prepare gives, as long as the analysis picture is the one
it was saved from. With a new grade the masks are those of the grade before; check_grade says how far the two grades' analysis pictures are apart
on the skin, for the caller to decide a cut must be prepared again.

A folder holds one range of one cut with one set of values: meta.json (the cut, the range, the values' signature, the code's tag) and t00000.pz ... A render takes from it
only when the cut, the range and the values are the same (Reader.open); anything else is prepared as ever. CONTOUR_PREP_CACHE (the root folder)
with CONTOUR_PREP_CACHE_MODE "read" switches the render's use on (warp.render); "write" has a render save what it prepares."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import threading
import zlib
from typing import Optional

import numpy as np

FORMAT = 1
LEVEL = 1                      # zlib level: the masks are smooth floats; level 1 is within a few percent of 6 and five times faster


class _Box:
    """A 2-D array kept as the box of its non-zero part, in its own dtype (to the bit)."""
    __slots__ = ("shape", "dtype", "y0", "x0", "a")

    def __init__(self, shape, dtype, y0, x0, a):
        self.shape, self.dtype, self.y0, self.x0, self.a = shape, dtype, y0, x0, a

    def __getstate__(self):
        return (self.shape, self.dtype, self.y0, self.x0, self.a)

    def __setstate__(self, st):
        self.shape, self.dtype, self.y0, self.x0, self.a = st


def _box(v):
    from . import engine as E
    if isinstance(v, E._PackedMask):
        return v                                   # already a box (float16, as the raw frames are kept): pickled as it is
    if not isinstance(v, np.ndarray) or v.ndim != 2:
        return v
    rows = np.flatnonzero(v.any(1))
    if rows.size == 0:
        return _Box(v.shape, v.dtype.str, 0, 0, np.zeros((0, 0), v.dtype))
    cols = np.flatnonzero(v.any(0))
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return _Box(v.shape, v.dtype.str, y0, x0, np.ascontiguousarray(v[y0:y1, x0:x1]))


def _unbox(v):
    if isinstance(v, _Box):
        out = np.zeros(v.shape, np.dtype(v.dtype))
        if v.a.size:
            out[v.y0:v.y0 + v.a.shape[0], v.x0:v.x0 + v.a.shape[1]] = v.a
        return out
    return v


def _pickled(rf) -> bytes:
    """A prepared frame pickled now, its jobs' masks boxed (the frame itself is not changed; what is done to it afterwards is not in the bytes)."""
    import copy
    if rf is None:
        return pickle.dumps(None, protocol=5)
    r2 = copy.copy(rf)
    r2.jobs = []
    for j in rf.jobs:
        jj = copy.copy(j)
        jj.masks = {k: _box(v) for k, v in j.masks.items()}
        r2.jobs.append(jj)
    return pickle.dumps(r2, protocol=5)


def _pack(rf) -> bytes:
    return zlib.compress(_pickled(rf), LEVEL)


def _unpack(b: bytes):
    rf = pickle.loads(zlib.decompress(b))
    if rf is not None:
        for j in rf.jobs:
            j.masks = {k: _unbox(v) for k, v in j.masks.items()}
    return rf


def values_sig(cut) -> str:
    """What the prepared frames depend on besides the pictures and the tracks: every target's values (and the analysis' version)."""
    d = {str(p.id): cut.params_for(p.id).to_dict() for p in cut.targets()}
    d["__analysis__"] = getattr(cut, "analysis_version", 0)
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:16]


def folder(root: str, cut, a: int, b: int) -> str:
    """A range of a cut with its values (two versions of a cut with other values are two folders)."""
    return os.path.join(root, "%s_%d_%d_%s" % (str(cut.name).replace(os.sep, "_").replace("/", "_"), int(a), int(b), values_sig(cut)[:8]))


def _fpath(d: str, t: int) -> str:
    return os.path.join(d, "t%05d.pz" % int(t))


def _meta(cut, a, b, tag) -> dict:
    return dict(format=FORMAT, cut=str(cut.name), a=int(a), b=int(b), values=values_sig(cut), tag=str(tag or ""))


class Writer:
    """Frames written by a thread of their own (packing and the disk off the caller's way); close() waits for them and writes meta.json last."""

    def __init__(self, d: str, meta: dict, queue: int = 4):
        import queue as Q
        os.makedirs(d, exist_ok=True)
        try:
            os.remove(os.path.join(d, "meta.json"))       # a folder being written is not one to read from
        except OSError:
            pass
        self.d, self.meta = d, meta
        self.q = Q.Queue(maxsize=max(1, queue))
        self.err = None
        self.lock = threading.Lock()
        self.seen = set()
        self.bytes = 0
        self.n = 0
        self.th = threading.Thread(target=self._run, daemon=True, name="prep-cache")
        self.th.start()

    def _run(self):
        while True:
            it = self.q.get()
            if it is None:
                return
            t, raw = it
            try:
                b = zlib.compress(raw, LEVEL)
                p = _fpath(self.d, t)
                with open(p + ".tmp", "wb") as f:
                    f.write(b)
                os.replace(p + ".tmp", p)
                self.bytes += len(b)
                self.n += 1
            except Exception as e:          # noqa: BLE001
                self.err = e

    def put(self, t: int, rf) -> None:
        """The frame as it is now (pickled here: the render goes on to retouch with it), written by the thread."""
        if self.err is not None:
            raise self.err
        with self.lock:
            if int(t) in self.seen:
                return
            self.seen.add(int(t))
        self.q.put((int(t), _pickled(rf)))

    def close(self, ok: bool = True) -> dict:
        self.q.put(None)
        self.th.join()
        if self.err is not None:
            raise self.err
        want = self.meta.get("b", 0) - self.meta.get("a", 0)
        if ok and self.n == want:
            m = dict(self.meta, frames=self.n, bytes=self.bytes)
            tmp = os.path.join(self.d, "meta.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=1)
            os.replace(tmp, os.path.join(self.d, "meta.json"))
        return dict(frames=self.n, bytes=self.bytes)


def save_cut(cut, tracker, a: int, b: int, root: str, tag: str = "", progress=None, cancel=None) -> dict:
    """Prepare frames a..b-1 of an open cut as its render would (engine.prepare in order, the next frames' raw masks made ahead where the computer
    allows: retouch.rawahead) and write them to the cut's folder under root. Returns {frames, bytes, seconds, folder}."""
    import time
    from . import engine as E
    from . import rawahead as RA
    t0 = time.time()
    d = folder(root, cut, a, b)
    w = Writer(d, _meta(cut, a, b, tag))
    ra = RA.start_for_render(cut, tracker, a, b)
    ok = False
    try:
        for t in range(int(a), int(b)):
            if cancel and cancel():
                break
            rf = E.prepare(cut, tracker, t)
            w.put(t, rf)
            if progress:
                progress(t - a + 1, b - a)
        else:
            ok = True
    finally:
        if ra is not None:
            RA.stop_for_render(tracker)
        r = w.close(ok)
    r.update(seconds=round(time.time() - t0, 1), folder=d, complete=ok)
    return r


class Reader:
    """The prepared frames of a cut's range, read from its folder; open() gives None when the folder is not for this cut, range and values."""

    def __init__(self, d: str, meta: dict):
        self.d, self.meta = d, meta
        self.hit = 0
        self.miss = 0

    @classmethod
    def open(cls, root: str, cut, a: int, b: int, tag: Optional[str] = None) -> "tuple[Optional[Reader], str]":
        d = folder(root, cut, a, b)
        try:
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            return None, "no saved analysis for this range"
        want = _meta(cut, a, b, tag if tag is not None else m.get("tag", ""))
        for k in ("format", "cut", "a", "b", "values", "tag"):
            if m.get(k) != want.get(k):
                return None, "saved analysis is not for this (%s)" % k
        if int(m.get("frames", -1)) != int(b) - int(a):
            return None, "saved analysis is not complete"
        return cls(d, m), ""

    def get(self, t: int):
        """(True, frame) from the folder, or (False, None) when this frame is not there."""
        try:
            with open(_fpath(self.d, t), "rb") as f:
                b = f.read()
        except OSError:
            self.miss += 1
            return False, None
        self.hit += 1
        return True, _unpack(b)


def mode() -> str:
    return (os.environ.get("CONTOUR_PREP_CACHE_MODE") or "").strip().lower() if os.environ.get("CONTOUR_PREP_CACHE") else ""


def reader_for_render(cut, a: int, b: int, warnings: Optional[list] = None) -> Optional[Reader]:
    """The render's reader when the environment asks for it (CONTOUR_PREP_CACHE, CONTOUR_PREP_CACHE_MODE read) and the folder fits."""
    if mode() != "read":
        return None
    r, why = Reader.open(os.environ["CONTOUR_PREP_CACHE"], cut, a, b, os.environ.get("CONTOUR_PREP_CACHE_TAG"))
    if r is None and warnings is not None:
        warnings.append("저장된 분석을 쓰지 않음: " + why)
    return r


def writer_for_render(cut, a: int, b: int, warnings: Optional[list] = None) -> Optional[Writer]:
    """(CONTOUR_PREP_CACHE_MODE write) the render saves each frame's analysis as it prepares it, so the cut can be rendered again on a new grade
    from it (a render that is cancelled or fails leaves a folder that is never read)."""
    if mode() != "write":
        return None
    try:
        root = os.environ["CONTOUR_PREP_CACHE"]
        if not _room(root):
            if warnings is not None:
                warnings.append("분석 저장 안 함: 디스크 공간 부족")
            return None
        return Writer(folder(root, cut, a, b), _meta(cut, a, b, os.environ.get("CONTOUR_PREP_CACHE_TAG", "")))
    except Exception as e:      # noqa: BLE001
        if warnings is not None:
            warnings.append("분석 저장 안 함: " + str(e)[:120])
        return None


MIN_FREE_GB = 20.0


def _room(root: str) -> bool:
    import shutil
    try:
        os.makedirs(root, exist_ok=True)
        return shutil.disk_usage(root).free / 2 ** 30 >= MIN_FREE_GB
    except OSError:
        return False


def check_grade(old_rgb: np.ndarray, new_rgb: np.ndarray, mask: Optional[np.ndarray]) -> dict:
    """How far two grades' analysis pictures (uint8 RGB, the same size) are apart where the retouch works (mask > 0.3; None: everywhere):
    mean and 99th percentile of the absolute difference per channel, in 8-bit levels."""
    a = old_rgb.astype(np.int16)
    b = new_rgb.astype(np.int16)
    d = np.abs(a - b).max(-1)
    if mask is not None:
        m = mask > 0.3
        if m.shape != d.shape:
            import cv2
            m = cv2.resize(mask.astype(np.float32), (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.3
        v = d[m]
    else:
        v = d.ravel()
    if v.size == 0:
        return dict(n=0, mean=0.0, p99=0.0)
    return dict(n=int(v.size), mean=float(v.mean()), p99=float(np.percentile(v, 99)))
