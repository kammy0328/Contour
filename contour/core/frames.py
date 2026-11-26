"""Frames for viewing and analysis, straight from the original (1.18).

Before 1.18 every cut was first decoded whole into a proxy file (display-referred 8-bit RGB, about 11 GB per minute
of 1080p) and nothing could be seen or analysed until that file was complete. Now a cut opens on its original:

- SourceFrames gives the same frames a proxy would hold (same size, same colour conversion, same frame numbers),
  decoded from the original with ffmpeg as they are asked for.
- Decoding is sequential: a reader stays open and keeps going forward (playback, analysis passes). A jump starts a
  new reader at that frame; a step backwards starts it a block earlier so the frames before are decoded with it
  (analysis passes that run backwards decode one block per BLOCK frames, not one seek per frame). Two readers are
  kept so playback and an analysis pass do not keep restarting each other.
- Decoded frames go into one memory pool shared by every SourceFrames of the app, capped in bytes; the least
  recently used frames leave first. Nothing is written to disk.
- A proxy file is still used when there is one, and can still be made (in the background) when the original is
  too slow to decode: frames_for() picks.

The analysis size of a cut never changes once it has been analysed: a proxy keeps the size it was made at, and the
person mattes an analysis wrote fix it as well (frame_size), so a cut analysed on a proxy opens on its original with
the same coordinates and the same mattes.

Frames handed out are always fresh copies (never a view of the pool or of a reader's buffer)."""
from __future__ import annotations

import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .media import FrameReader, MediaInfo, planes_to_rgb_float
from .proxy import Proxy, proxy_size

BLOCK = 16              # frames decoded with one backwards step
AHEAD = 48              # a request this far past a reader's position reads on instead of seeking
READERS = 2             # open readers per source (playback and an analysis pass at once)
DEFAULT_POOL_MB = 1536
SCRUB_SEEKS = 10        # reader starts on one cut (the user jumping around it) after which 'auto' makes a proxy:
                        # a jump in a long-GOP original takes ~0.2 s (measured, 1080p H.264), in a proxy ~2 ms


class FrameError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- the memory pool
class FramePool:
    """Decoded frames of every source, least recently used out first, at most `limit` bytes."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self._d: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.bytes = 0
        self.peak = 0
        self._lock = threading.Lock()

    def get(self, key) -> Optional[np.ndarray]:
        with self._lock:
            a = self._d.get(key)
            if a is not None:
                self._d.move_to_end(key)
            return a

    def put(self, key, a: np.ndarray) -> None:
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.bytes -= old.nbytes
            self._d[key] = a
            self.bytes += a.nbytes
            while self.bytes > self.limit and len(self._d) > 1:
                _, x = self._d.popitem(last=False)
                self.bytes -= x.nbytes
            self.peak = max(self.peak, self.bytes)

    def drop(self, owner) -> None:
        with self._lock:
            for k in [k for k in self._d if k[0] == owner]:
                self.bytes -= self._d.pop(k).nbytes

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.bytes = 0
            self.peak = 0

    def set_limit(self, limit: int) -> None:
        with self._lock:
            self.limit = int(limit)
            while self.bytes > self.limit and self._d:
                _, x = self._d.popitem(last=False)
                self.bytes -= x.nbytes


_pool: Optional[FramePool] = None
_pool_lock = threading.Lock()


def pool() -> FramePool:
    global _pool
    with _pool_lock:
        if _pool is None:
            mb = DEFAULT_POOL_MB
            try:
                from .settings import settings
                mb = int(getattr(settings(), "frame_cache_mb", DEFAULT_POOL_MB) or DEFAULT_POOL_MB)
            except Exception:
                pass
            mb = int(os.environ.get("DIAP_FRAME_CACHE_MB", mb))
            _pool = FramePool(max(64, mb) * 1024 * 1024)
        return _pool


# ---------------------------------------------------------------------------------------------- source frames
def source_present(info: MediaInfo) -> bool:
    try:
        if info.kind == "sequence":
            return Path(info.seq_pattern % info.seq_start).exists()
        return bool(info.path) and Path(info.path).exists()
    except Exception:
        return False


def _sig(info: MediaInfo) -> str:
    try:
        st = os.stat(info.path if info.kind == "video" else info.seq_pattern % info.seq_start)
        return f"{st.st_size}:{st.st_mtime_ns}"
    except (OSError, TypeError):
        return ""


def _decode_key(info: MediaInfo) -> tuple:
    """The fields of the file's description the decode command and the conversion read (frame rate for seeking,
    pixel layout, matrix and range for the RGB conversion)."""
    return tuple(str(getattr(info, k, "")) for k in ("fps_num", "fps_den", "width", "height", "pix_fmt", "chroma",
                                                       "color_range", "color_matrix", "color_space", "is_float",
                                                       "bit_depth", "fps_fraction"))


def _owner(info: MediaInfo, n: int, w: int, h: int, view_key: str) -> tuple:
    return (info.kind, str(info.path if info.kind == "video" else info.seq_pattern),
            int(getattr(info, "seq_start", 0) or 0), _sig(info), int(getattr(info, "offset", 0) or 0),
            int(n), int(w), int(h), view_key, _decode_key(info))


def origin(px, info: MediaInfo, vs) -> tuple:
    """What the pixels of a cut's frames come from, whoever reads them (the keys of results kept in memory are made
    with it): the file and its version, the stretch, the size and the colour view. The same for the original and
    for a proxy made from that version of it in this view (same pixels, so results made on one stay good when the
    other takes over). A proxy shown because there is nothing else (the original missing, or made in another
    view) is told by its own file."""
    if px is None:
        return ()
    if getattr(px, "kind", "") == "source":
        return px.owner
    vk = vs.key()
    try:
        ok = source_present(info) and px.is_valid(vk)
    except Exception:
        ok = False
    if ok:
        return _owner(info, px.n, px.w, px.h, vk)
    m = px.meta()
    try:
        st = os.stat(px.meta_path)
        made = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        made = ""
    return ("proxy", str(px.path), int(px.offset), int(px.n), int(px.w), int(px.h), str(m.get("view")),
            str(m.get("sig")), made)


class _Cursor:
    def __init__(self, reader: FrameReader, start: int, end: int):
        self.reader = reader
        self.next = start          # the frame the reader yields next
        self.end = end
        self.used = time.monotonic()

    def close(self):
        try:
            self.reader.close()
        except Exception:
            pass


class SourceFrames:
    """The frames of a cut decoded from its original at the analysis size, converted to the display view exactly
    as a proxy is made. Same interface as Proxy for everything that reads frames (n, w, h, sx, sy, frame, gray,
    to_proxy, to_source, thumbnail, has_frames, is_valid, close)."""
    kind = "source"
    shared = False
    offset = 0

    def __init__(self, cache_dir: Path, info: MediaInfo, size: Tuple[int, int], cm, vs):
        self.dir = Path(cache_dir)
        self.info = info
        self.w, self.h = int(size[0]), int(size[1])
        self.n = int(info.frames)
        self.n_file = self.n
        self.sx = self.w / info.width
        self.sy = self.h / info.height
        self.cm, self.vs = cm, vs
        self.view_key = vs.key()
        # what the decoded pixels depend on: the file (and its version), the stretch, the size and the view
        self.owner = _owner(info, self.n, self.w, self.h, self.view_key)
        self._lock = threading.RLock()      # the readers
        self._cursors: list = []
        self._last = -1
        self._gray: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._glock = threading.Lock()
        self.decoded = 0
        self.decode_s = 0.0
        self.seeks = 0
        self.error = ""

    # ------------------------------------------------------------------ state
    def has_frames(self) -> bool:
        """The original is there (it is decoded when frames are asked for)."""
        return source_present(self.info)

    def is_valid(self, view_key: str) -> bool:
        return view_key == self.view_key and self.has_frames()

    def open(self) -> None:
        pass

    def close(self) -> None:
        """Readers and their ffmpeg processes end; decoded frames stay in the pool for a reopening."""
        with self._lock:
            for c in self._cursors:
                c.close()
            self._cursors = []
        with self._glock:
            self._gray.clear()
        sc = self.__dict__.get("_src_crops")
        if sc is not None:
            sc.close()

    def decode_fps(self) -> float:
        return self.decoded / self.decode_s if self.decode_s > 0 else 0.0

    # ------------------------------------------------------------------ access
    def frame(self, i: int) -> np.ndarray:
        i = int(np.clip(int(i), 0, self.n - 1))
        key = (self.owner, i)
        p = pool()
        a = p.get(key)
        if a is None:
            with self._lock:
                a = p.get(key)
                if a is None:
                    a = self._decode_to(i)
        return np.array(a)                  # a copy: the pool's frame is never handed out

    def peek(self, i: int) -> Optional[np.ndarray]:
        """Frame i if it is already decoded (a copy), else None: never decodes (for the UI thread)."""
        a = pool().get((self.owner, int(np.clip(int(i), 0, self.n - 1))))
        return None if a is None else np.array(a)

    def gray(self, i: int) -> np.ndarray:
        with self._glock:
            g = self._gray.get(i)
        if g is None:
            g = cv2.cvtColor(self.frame(i), cv2.COLOR_RGB2GRAY)
            with self._glock:
                self._gray[i] = g
                while len(self._gray) > 64:
                    self._gray.popitem(last=False)
        return g

    def _open_cursor(self, start: int) -> _Cursor:
        pf = "gbrpf32le" if self.info.is_float else "rgb48le"
        r = FrameReader(self.info, start, self.n - start, pf, size=(self.w, self.h), prefetch=4)
        c = _Cursor(r, start, self.n)
        self._cursors.append(c)
        self.seeks += 1
        while len(self._cursors) > READERS:
            old = min(self._cursors, key=lambda x: x.used)
            self._cursors.remove(old)
            old.close()
        return c

    def _decode_to(self, i: int) -> np.ndarray:
        """Called with the reader lock held. Reads on with the reader that is just before i, else starts a reader at
        i (or a block before i when going backwards), keeping every frame read on the way."""
        back = 0 <= i < self._last
        self._last = i
        cur = None
        for c in self._cursors:
            if c.next <= i <= c.next + AHEAD and (cur is None or c.next > cur.next):
                cur = c
        if cur is None:
            start = max(0, i - BLOCK + 1) if back else i
            cur = self._open_cursor(start)
        cur.used = time.monotonic()
        p = pool()
        got = None
        while cur.next <= i:
            t0 = time.perf_counter()
            planes = cur.reader.read()
            if planes is None:
                err = cur.reader.error_text.strip()
                self._cursors.remove(cur)
                cur.close()
                self.error = f"원본에서 프레임 {i}을(를) 읽지 못했습니다 ({cur.next}에서 끝남). {err[:300]}"
                raise FrameError(self.error)
            rgb = planes_to_rgb_float(planes, cur.reader.layout)
            u8 = self.cm.to_display_u8(rgb, self.vs)
            self.decode_s += time.perf_counter() - t0
            self.decoded += 1
            p.put((self.owner, cur.next), u8)
            if cur.next == i:
                got = u8
            cur.next += 1
        if cur.next >= cur.end:
            self._cursors.remove(cur)
            cur.close()
        return got

    # ------------------------------------------------------------------ coordinates, thumbnails
    def to_proxy(self, xy: np.ndarray) -> np.ndarray:
        out = np.array(xy, dtype=np.float32, copy=True)
        out[..., 0] *= self.sx
        out[..., 1] *= self.sy
        return out

    def to_source(self, xy: np.ndarray) -> np.ndarray:
        out = np.array(xy, dtype=np.float32, copy=True)
        out[..., 0] /= self.sx
        out[..., 1] /= self.sy
        return out

    def one(self, i: int) -> np.ndarray:
        """Frame i with a reader of its own that ends at once (a thumbnail, a look at one frame from another
        thread): no reader is left open by it, whoever closes this source and when."""
        i = int(np.clip(int(i), 0, self.n - 1))
        a = self.peek(i)
        if a is not None:
            return a
        pf = "gbrpf32le" if self.info.is_float else "rgb48le"
        with FrameReader(self.info, i, 1, pf, size=(self.w, self.h), prefetch=1) as r:
            planes = r.read()
            if planes is None:
                raise FrameError(f"원본에서 프레임 {i}을(를) 읽지 못했습니다. {r.error_text.strip()[:300]}")
            u8 = self.cm.to_display_u8(planes_to_rgb_float(planes, r.layout), self.vs)
        pool().put((self.owner, i), u8)
        return np.array(u8)

    def thumbnail(self, i: int, width: int = 160) -> np.ndarray:
        f = self.one(i)
        h = max(1, int(round(f.shape[0] * width / f.shape[1])))
        return cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)

    def meta(self) -> dict:
        return {}


# ---------------------------------------------------------------------------------------------- pieces of the original
CROP_READERS = 8            # piece readers kept open at once (faces and, since 1.23, bodies of one cut)
SHARED_DECODE = True    # (1.26.3) every piece of a frame cut from one decode of the whole frame (one reader, the frames kept in the pool), not a reader
                        # of its own a region that decoded the whole frame again (4 faces and their bodies: 5 to 8 decodes of a 4K frame, the
                        # retouch waiting on them: LNGSHOT j2849). A piece made smaller is resized here (area), as ffmpeg's area scaler did.
CROP_GRID = 96          # the piece decoded is the piece asked for, widened to this grid: a face moving a little
                        # keeps its piece (one reader going on) instead of a new reader every frame


class SourceCrops:
    """Pieces of the original at (up to) its own resolution, in the display view: for looking at fine skin detail
    (1.19: the analysis picture of a 4K original is a third to a half of its size, and a pore, a thin neck line or a
    small spot on a small face is gone from it). Only the piece is converted and kept; the reader goes on through
    the frames as long as the piece asked for stays inside the piece it decodes.

    piece(i, box, scale): frame i, box (x0, y0, x1, y1) in source pixels, scale <= 1 (pixels handed out per source
    pixel). The same frame numbers, the same colour conversion as the frames of the cut."""

    def __init__(self, info: MediaInfo, cm, vs):
        self.info, self.cm, self.vs = info, cm, vs
        self.n = int(info.frames)
        self.base = (info.kind, str(info.path if info.kind == "video" else info.seq_pattern),
                     int(getattr(info, "seq_start", 0) or 0), _sig(info), int(getattr(info, "offset", 0) or 0),
                     self.n, vs.key(), _decode_key(info), "crop")
        self._lock = threading.RLock()              # the table of readers
        # one reader per region (several faces of one frame), the least recently used closed beyond CROP_READERS:
        # with one reader for all of them it restarted at every face (review 6c3dcb3: 12 restarts for 6 frames x 2)
        self._readers: "OrderedDict[tuple, list]" = OrderedDict()      # key -> [reader, next frame]
        # a region is read by one thread at a time, different regions side by side (the people of a frame: with one
        # lock for all of them the second and third person waited a third of a second each: j2799)
        self._rlocks: dict = {}
        self.decoded = 0
        self.seeks = 0
        self._full_lock = threading.Lock()          # (SHARED_DECODE) the reader of the whole frame
        self._full = None                           # [reader, next frame]

    def _whole(self, i: int) -> np.ndarray:
        """(SHARED_DECODE) Frame i of the original, whole, as the reader gives it ((H, W, 3) uint16 rgb48, or float32 RGB for float
        sources), kept in the pool for the other pieces of the frame."""
        p = pool()
        k = (self.base, "whole", i)
        a = p.get(k)
        if a is not None:
            return a
        with self._full_lock:
            a = p.get(k)
            if a is not None:
                return a
            r = self._full
            if r is not None and not (r[1] <= i <= r[1] + 8):
                try:
                    r[0].close()
                except Exception:
                    pass
                r = self._full = None
            if r is None:
                pf = "gbrpf32le" if self.info.is_float else "rgb48le"
                r = self._full = [FrameReader(self.info, i, self.n - i, pf, prefetch=2), i]
                self.seeks += 1
            got = None
            while r[1] <= i:
                planes = r[0].read()
                if planes is None:
                    err = r[0].error_text.strip()
                    at = r[1]
                    try:
                        r[0].close()
                    except Exception:
                        pass
                    self._full = None
                    raise FrameError(f"원본에서 프레임 {i}을(를) 읽지 못했습니다 ({at}에서 끝남). {err[:300]}")
                if r[0].layout.pix_fmt == "gbrpf32le":
                    g_, b_, r_ = planes[0], planes[1], planes[2]
                    w_ = np.dstack([r_, g_, b_]).astype(np.float32)
                else:
                    w_ = planes[0]
                p.put((self.base, "whole", r[1]), w_)
                if r[1] == i:
                    got = w_
                r[1] += 1
            return got

    def _from_whole(self, i: int, gb, osz) -> np.ndarray:
        w_ = self._whole(i)
        x0, y0, x1, y1 = gb
        c = w_[y0:y1, x0:x1]
        rgb = c.astype(np.float32) if c.dtype == np.float32 else c.astype(np.float32) / 65535.0
        if (rgb.shape[1], rgb.shape[0]) != tuple(osz):
            rgb = cv2.resize(rgb, (int(osz[0]), int(osz[1])), interpolation=cv2.INTER_AREA)
        return self.cm.to_display_u8(np.ascontiguousarray(rgb), self.vs)

    def _region_lock(self, key) -> "threading.Lock":
        with self._lock:
            lk = self._rlocks.get(key)
            if lk is None:
                lk = self._rlocks[key] = threading.Lock()
            return lk

    def _drop(self, key) -> None:
        r = self._readers.pop(key, None)
        if r is not None:
            try:
                r[0].close()
            except Exception:
                pass

    def close(self) -> None:
        with self._full_lock:
            if self._full is not None:
                try:
                    self._full[0].close()
                except Exception:
                    pass
                self._full = None
        with self._lock:
            keys = list(self._readers)
        for k in keys:
            with self._region_lock(k):              # (a read under way finishes first)
                with self._lock:
                    self._drop(k)
        with self._lock:
            for k in list(self._readers):
                self._drop(k)
            self._rlocks.clear()

    def _grid(self, box) -> Tuple[int, int, int, int]:
        W, H = int(self.info.width), int(self.info.height)
        g = CROP_GRID
        x0 = max(0, (int(box[0]) - g // 3) // g * g)
        y0 = max(0, (int(box[1]) - g // 3) // g * g)
        x1 = min(W, -(-(int(box[2]) + g // 3) // g) * g)
        y1 = min(H, -(-(int(box[3]) + g // 3) // g) * g)
        x1 -= (x1 - x0) % 2
        y1 -= (y1 - y0) % 2
        return x0, y0, x1, y1

    def piece(self, i: int, box, scale: float = 1.0) -> np.ndarray:
        i = int(np.clip(int(i), 0, self.n - 1))
        scale = float(min(1.0, max(0.05, scale)))
        gb = self._grid(box)
        gw, gh = gb[2] - gb[0], gb[3] - gb[1]
        ow, oh = max(2, int(round(gw * scale))), max(2, int(round(gh * scale)))
        key = (self.base, gb, (ow, oh))
        p = pool()
        a = p.get((key, i))
        if a is None:
            with self._region_lock(key):
                a = p.get((key, i))
                if a is None:
                    if SHARED_DECODE:
                        a = self._from_whole(i, gb, (ow, oh))
                        self.decoded += 1
                        p.put((key, i), a)
                    else:
                        a = self._decode(i, gb, (ow, oh), key)
        # the piece asked for inside the piece decoded
        fx, fy = ow / gw, oh / gh
        xa, ya = int(round((box[0] - gb[0]) * fx)), int(round((box[1] - gb[1]) * fy))
        xb, yb = int(round((box[2] - gb[0]) * fx)), int(round((box[3] - gb[1]) * fy))
        return np.array(a[max(0, ya):max(ya + 1, yb), max(0, xa):max(xa + 1, xb)])

    def _decode(self, i: int, gb, osz, key) -> np.ndarray:
        """Frame i of a region, with the region's lock held (piece)."""
        with self._lock:
            r = self._readers.get(key)
            if r is not None and not (r[1] <= i <= r[1] + 8):
                self._drop(key)
                r = None
        if r is None:
            pf = "gbrpf32le" if self.info.is_float else "rgb48le"
            gw, gh = gb[2] - gb[0], gb[3] - gb[1]
            rd = FrameReader(self.info, i, self.n - i, pf, size=osz if osz != (gw, gh) else None, prefetch=2,
                             crop=(gb[0], gb[1], gw, gh))
            r = [rd, i]
            with self._lock:
                self._readers[key] = r
                self.seeks += 1
                for k in list(self._readers):               # the least recently used first; one being read stays
                    if len(self._readers) <= CROP_READERS:
                        break
                    lk = self._rlocks.get(k)
                    if k != key and not (lk is not None and lk.locked()):
                        self._drop(k)
        with self._lock:
            if key in self._readers:
                self._readers.move_to_end(key)
        got = None
        p = pool()
        while r[1] <= i:
            planes = r[0].read()
            if planes is None:
                err = r[0].error_text.strip()
                at = r[1]
                with self._lock:
                    if self._readers.get(key) is r:
                        self._drop(key)
                    else:
                        try:
                            r[0].close()
                        except Exception:
                            pass
                raise FrameError(f"원본에서 프레임 {i}의 일부를 읽지 못했습니다 ({at}에서 끝남). {err[:300]}")
            u8 = self.cm.to_display_u8(planes_to_rgb_float(planes, r[0].layout), self.vs)
            self.decoded += 1
            p.put((key, r[1]), u8)
            if r[1] == i:
                got = u8
            r[1] += 1
        return got


def source_crops(px, cm=None, vs=None) -> Optional[SourceCrops]:
    """The piece reader of a cut's frames (a proxy or the original), made on first use and closed with them. None
    when the original is not there."""
    sc = px.__dict__.get("_src_crops")
    if sc is None:
        if not source_present(px.info):
            return None
        if cm is None:
            from .colour import manager
            cm = manager()
        vs = vs or getattr(px, "vs", None)
        if vs is None:
            return None
        sc = px.__dict__["_src_crops"] = SourceCrops(px.info, cm, vs)
    return sc


# ---------------------------------------------------------------------------------------------- choosing
_MATTE = re.compile(r"^matte_(\d+)x(\d+)\.u8$")


def frame_size(project, cut, long_side: Optional[int] = None) -> Tuple[int, int]:
    """The cut's analysis size: a complete proxy keeps its size; else the size of the mattes an analysis made;
    else the size the setting gives."""
    from .settings import settings
    long_side = int(long_side or settings().proxy_long_side)
    px = Proxy(project.proxy_dir(cut), cut.info, long_side)
    if px.has_frames():
        return px.w, px.h
    n = int(cut.info.frames)
    best = None
    try:
        for f in Path(project.cache_dir(cut)).iterdir():
            m = _MATTE.match(f.name)
            if not m:
                continue
            mw, mh = int(m.group(1)), int(m.group(2))
            st = f.stat()
            if st.st_size == n * mw * mh and (best is None or st.st_mtime > best[0]):
                best = (st.st_mtime, mw * 2, mh * 2)
    except OSError:
        pass
    if best is not None:
        return best[1], best[2]
    return proxy_size(cut.info, long_side)


def proxy_for(project, cut, long_side: Optional[int] = None) -> Proxy:
    """The cut's proxy file (made or not) at the cut's analysis size."""
    from .settings import settings
    long_side = int(long_side or settings().proxy_long_side)
    px = Proxy(project.proxy_dir(cut), cut.info, long_side)
    if px.has_frames() or px.shared:
        return px
    return Proxy(project.proxy_dir(cut), cut.info, long_side, size=frame_size(project, cut, long_side))


def frames_for(project, cut, vs=None, cm=None, long_side: Optional[int] = None):
    """What a cut's frames are read from, in this order:
    1. its proxy (or the bake's shared one) made in this view: fastest
    2. its original, decoded as frames are asked for
    3. a proxy made in another view, when the original is missing (it can be looked at, not analysed in the right
       view; callers that need the original check source_present)
    None when there is neither."""
    from .colour import manager
    from .settings import settings
    if vs is None:
        from ..warp.render import _vs_for
        vs = _vs_for(cut)
    cm = cm or manager()
    long_side = int(long_side or settings().proxy_long_side)
    px = proxy_for(project, cut, long_side)
    if px.has_frames() and px.is_valid(vs.key()):
        return px
    if source_present(cut.info):
        return SourceFrames(project.proxy_dir(cut), cut.info, frame_size(project, cut, long_side), cm, vs)
    if px.has_frames():
        return px
    return None


def proxy_mode() -> str:
    """'auto' (a proxy is made in the background when the original decodes slower than it plays), 'always' (made in
    the background for every opened cut) or 'off'."""
    try:
        from .settings import settings
        m = str(getattr(settings(), "proxy_mode", "auto") or "auto")
    except Exception:
        m = "auto"
    return m if m in ("auto", "always", "off") else "auto"


def wants_proxy(fr, fps: float) -> bool:
    """Whether a proxy should be made for frames read from this source (by the setting and the measured speed)."""
    if fr is None or getattr(fr, "kind", "") != "source":
        return False
    m = proxy_mode()
    if m == "off":
        return False
    if m == "always":
        return True
    if fr.seeks >= SCRUB_SEEKS:
        return True
    return fr.decoded >= 24 and fr.decode_fps() < 0.9 * max(1.0, float(fps or 24.0))
