"""Edit session: the open project / cut, preview rendering and edit operations (UI-agnostic)."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.colour import PASSTHROUGH, ViewSettings, manager
from ..core.project import Cut, Project
from ..core.proxy import Proxy
from ..core.settings import settings
from ..stats.tracker import log_event
from ..track import skeleton as S
from ..track.analyze import CutTracker
from ..warp import field as F
from ..warp.separate import fill_background


_EPOCHS = __import__("itertools").count(1)


class EditSession:
    """epoch: changes whenever another cut is opened (unique over every session of the app). The cache keys of a
    frame's intermediate results (proxy frame, retouch masks, retouched picture, warp field) carry it, so a frame of
    the cut that was open when its computation started can never be found as the same frame of the cut open now."""

    def __init__(self):
        self.project: Optional[Project] = None
        self.cut: Optional[Cut] = None
        self.proxy: Optional[Proxy] = None
        self.tracker: Optional[CutTracker] = None
        self.version = 0                     # bump to invalidate previews (any change)
        self.epoch = next(_EPOCHS)
        self.vers = {"geo": 0, "ret": 0, "trk": 0}   # what changed: shape params / retouch params / tracking
        self.interactive = False             # a slider is being dragged: coarser preview field
        self._cache: "OrderedDict[tuple, object]" = OrderedDict()
        self._frames: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._mask_cache: dict = {}
        self._lock = threading.RLock()
        self._compute = threading.RLock()    # one heavy frame computation at a time (preview / playback cache)
        self._play: Dict[int, Tuple[tuple, bytes]] = {}   # playback cache: frame -> (result key, jpeg)
        self._play_bytes = 0
        self.play_limit = 1200 * 1024 * 1024
        self.last_touch = 0.0
        self.last_seek = 0.0          # the user moved to another frame (the playback cache waits a moment)
        self.preview_wait = 0         # preview frames waiting: the playback cache gives way to them
        self.dirty = False

    # ------------------------------------------------------------------ project / cut
    def view_settings(self, cut: Optional[Cut] = None) -> ViewSettings:
        cut = cut or self.cut
        s = settings()
        cs = (cut.input_cs if cut else "") or PASSTHROUGH
        return ViewSettings(input_cs=cs, display=s.display, view=s.view, lut_path=cut.lut_path if cut else "")

    def make_proxy(self, cut: Cut) -> Proxy:
        """The cut's proxy file at its analysis size (made or not): only to build one."""
        from ..core.frames import proxy_for
        return proxy_for(self.project, cut)

    def make_frames(self, cut: Cut):
        """What the cut's frames are read from: its proxy when there is one in this view, else its original."""
        from ..core.frames import frames_for
        return frames_for(self.project, cut, self.view_settings(cut))

    def open_cut(self, cut: Optional[Cut]) -> None:
        with self._lock:
            if self.proxy is not None:
                self.proxy.close()
            self.cut = cut
            self.proxy = None
            self.tracker = None
            self._cache.clear()
            self._frames.clear()
            self._mask_cache = {}
            self._play = {}
            self._play_bytes = 0
            self.version += 1
            self._new_epoch()
            if cut is None or self.project is None:
                return
            px = self.make_frames(cut)
            if px is not None:
                self.proxy = px
                self.tracker = CutTracker(self.project, cut, px)

    def adopt(self, other: "EditSession") -> None:
        """Takes over a cut another session opened and pre-rendered in the background (playback going on into the
        next cut without a pause): its proxy, tracker, caches and the versions their keys were made with."""
        with self._lock:
            if self.proxy is not None and self.proxy is not other.proxy:
                self.proxy.close()
            with other._lock:
                self.cut, self.proxy, self.tracker = other.cut, other.proxy, other.tracker
                self._cache = OrderedDict(other._cache)
                self._frames = OrderedDict(other._frames)
                self._mask_cache = dict(other._mask_cache)
                self._play = dict(other._play)
                self._play_bytes = other._play_bytes
                self.vers = dict(other.vers)
                self.epoch = other.epoch          # the other session's entries were made for this cut
            self.version += 1

    def _new_epoch(self) -> None:
        with self._lock:
            self.epoch = next(_EPOCHS)

    def proxy_ready(self) -> bool:
        """Frames can be read in the cut's current view (from a proxy or the original)."""
        return self.proxy is not None and self.proxy.is_valid(self.view_settings().key())

    def from_source(self) -> bool:
        return getattr(self.proxy, "kind", "") == "source"

    def swap_frames(self, cut_id: str, new) -> bool:
        """A proxy finished in the background for the open cut: read from it from now on. Same size, same frame
        numbers and the same pixels as the original's frames, so nothing else changes (tracks and mattes stay)."""
        with self._lock:
            if self.cut is None or self.cut.id != cut_id or new is None or self.proxy is None:
                return False
            if (new.w, new.h, new.n) != (self.proxy.w, self.proxy.h, self.proxy.n):
                return False
            old, self.proxy = self.proxy, new
            if self.tracker is not None:
                self.tracker.proxy = new
            self._frames.clear()
        try:
            old.close()
        except Exception:
            pass
        return True

    def save(self) -> None:
        if self.project:
            self.project.save()
        if self.tracker:
            self.tracker.save()
        self.dirty = False

    def touch(self, kind: str = "all") -> None:
        """Something changed. kind: 'geo' (shape sliders, protection), 'ret' (retouch sliders),
        'trk' (tracking / joints) or 'all'. Only caches that depend on it are dropped."""
        self.last_touch = time.time()
        with self._lock:
            self.version += 1
            kinds = ("geo", "ret", "trk") if kind == "all" else (kind,)
            for k in kinds:
                self.vers[k] += 1
            if "trk" in kinds:
                self._mask_cache = {}
            # entries keyed by stale versions simply stop being hit; drop the obviously stale ones
            for key in [k for k in self._cache if not self._valid(k)]:
                self._cache.pop(key, None)
        self.dirty = True

    def _valid(self, key) -> bool:
        kind = key[0]
        v = self.vers
        if kind == "field":
            return key[2:4] == (v["geo"], v["trk"]) and key[-1] == self.epoch
        if kind in ("retouch", "ret_img"):
            return key[2:4] == (v["ret"], v["trk"]) and key[-1] == self.epoch
        if kind == "res":
            return key[2:5] == (v["geo"], v["ret"], v["trk"]) and key[-1] == self.epoch
        return False

    # ------------------------------------------------------------------ preview
    def frame(self, t: int) -> Optional[np.ndarray]:
        with self._lock:
            px, ep = self.proxy, self.epoch
            if px is None:
                return None
            f = self._frames.get(t)
            if f is not None:
                self._frames.move_to_end(t)
                return f
        f = np.asarray(px.frame(t))
        with self._lock:
            if self.epoch == ep:                  # still the same cut: else this frame is not the open cut's
                self._frames[t] = f
                while len(self._frames) > 8:
                    self._frames.popitem(last=False)
        return f

    def field(self, t: int, coarse: Optional[bool] = None) -> Optional[F.FieldResult]:
        if self.tracker is None or self.tracker.store is None or self.cut is None:
            return None
        coarse = self.interactive if coarse is None else coarse
        ep = self.epoch
        key = ("field", t, self.vers["geo"], self.vers["trk"], coarse, ep)
        fine = ("field", t, self.vers["geo"], self.vers["trk"], False, ep)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if coarse and fine in self._cache:
                return self._cache[fine]
        fi = F.build_input(self.cut, self.tracker, t)
        px = self.proxy
        if px is None:                     # the cut was closed while this was being computed
            return None
        div = 240 if coarse else 480
        fr = F.compute_field(fi, px.w, px.h, max(2, int(round(px.w / div))))
        self._put(key, fr)
        self._last_field = (t, fr)
        return fr

    def peek_field(self, t: int) -> Optional[F.FieldResult]:
        """The newest field already computed for frame t (current settings first, else the last one), without
        computing anything: for drawing overlays where the picture is."""
        with self._lock:
            best = None
            for key, val in reversed(self._cache.items()):
                if key[0] != "field" or key[1] != t or key[-1] != self.epoch:
                    continue
                if key[2:4] == (self.vers["geo"], self.vers["trk"]):
                    return val
                if best is None:
                    best = val
            if best is None:
                lf = getattr(self, "_last_field", None)
                if lf is not None and lf[0] == t:
                    best = lf[1]
            return best

    # ------------------------------------------------------------------ playback cache
    def play_get(self, t: int) -> Optional[np.ndarray]:
        """Finished result of frame t from the playback cache (current settings only)."""
        e = self._play.get(t)
        if e is None or self.cut is None or e[0] != self.result_key(t, True):
            return None
        a = cv2.imdecode(np.frombuffer(e[1], np.uint8), cv2.IMREAD_COLOR)
        return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    def play_has(self, t: int) -> bool:
        e = self._play.get(t)
        return e is not None and self.cut is not None and e[0] == self.result_key(t, True)

    def play_put(self, t: int, key: tuple, img: np.ndarray, near: int = 0) -> None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return
        data = buf.tobytes()
        with self._lock:
            old = self._play.pop(t, None)
            if old is not None:
                self._play_bytes -= len(old[1])
            self._play[t] = (key, data)
            self._play_bytes += len(data)
            # over budget: drop stale entries first, then the frames furthest from where we are
            if self._play_bytes > self.play_limit:
                cur = {f for f in self._play if self._play[f][0] == self.result_key(f, True)}
                order = sorted(self._play, key=lambda f: (f in cur, -abs(f - near)))
                for f in order:
                    if self._play_bytes <= self.play_limit * 0.9:
                        break
                    if f == t:
                        continue
                    self._play_bytes -= len(self._play.pop(f)[1])

    def play_frames(self) -> List[int]:
        if self.cut is None:
            return []
        return [f for f, e in list(self._play.items()) if e[0] == self.result_key(f, True)]

    def _put(self, key, val):
        with self._lock:
            self._cache[key] = val
            while len(self._cache) > 40:
                self._cache.popitem(last=False)

    def result_key(self, t: int, final: bool = False) -> tuple:
        # the cut is part of the key: a result computed for the previous cut must never show on this one
        # and the epoch: the same cut opened again (or its proxy made again) while an old computation was running
        # must not get that computation's picture
        return ("res", t, self.vers["geo"], self.vers["ret"], self.vers["trk"], False if final else self.interactive,
                self.cut.mode if self.cut else "", self.cut.id if self.cut else "", self.epoch)

    def cached_result(self, t: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._cache.get(self.result_key(t))

    def retouched(self, t: int) -> Optional[np.ndarray]:
        ep = self.epoch                     # before the frame is read: the key is for the cut this frame is of
        img = self.frame(t)
        if img is None:
            return None
        live = bool(self.interactive)
        key = ("ret_img", t, self.vers["ret"], self.vers["trk"], live, ep)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            full = ("ret_img", t, self.vers["ret"], self.vers["trk"], False, ep)
            if live and full in self._cache:
                return self._cache[full]
        rf = self.retouch(t)
        if rf is not None:
            from ..retouch.engine import apply_rgb
            if live:
                # while a slider is dragged: the retouch at half size, its change added back onto the full
                # picture (about 4x faster, the slider follows the hand); full size again on release
                import cv2 as _cv2
                h, w = img.shape[:2]
                small = _cv2.resize(img, (w // 2, h // 2), interpolation=_cv2.INTER_AREA)
                ds = apply_rgb(small, rf, self.proxy.sx * 0.5).astype(np.int16) - small.astype(np.int16)
                d = _cv2.resize(ds, (w, h), interpolation=_cv2.INTER_LINEAR)
                img = np.clip(img.astype(np.int16) + d, 0, 255).astype(np.uint8)
            else:
                img = apply_rgb(img, rf, self.proxy.sx)
        self._put(key, img)
        return img

    def result(self, t: int, final: bool = False) -> Optional[np.ndarray]:
        if self.frame(t) is None:
            return None
        key = self.result_key(t, final)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        with self._compute:
            return self._result(t, key, final)

    def _result(self, t: int, key: tuple, final: bool) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        img = self.retouched(t)
        fr = self.field(t, False if final else None)
        if fr is None or fr.empty():
            out = img
        elif self.cut.mode == "separate":
            out = self._separate_preview(t, img, fr)
        else:
            out = F.warp_image(img, fr)
        self._put(key, out)
        return out

    def retouch(self, t: int):
        """Retouch masks and values for frame t (None when nothing is active)."""
        if self.tracker is None or self.tracker.store is None or self.cut is None:
            return None
        key = ("retouch", t, self.vers["ret"], self.vers["trk"], self.epoch)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        from ..retouch.engine import prepare
        try:
            rf = prepare(self.cut, self.tracker, t, self._mask_cache)
        except Exception:
            rf = None
        self._put(key, rf)
        return rf

    def skin_masks(self, t: int):
        rf = self.retouch(t)
        return [] if rf is None else rf

    def _people_alpha(self, t: int) -> Optional[np.ndarray]:
        a = self.tracker.matte.get(t)
        if a is None:
            return None
        return cv2.resize(np.asarray(a), (self.proxy.w, self.proxy.h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0

    def target_alpha(self, t: int) -> Optional[np.ndarray]:
        al = self._people_alpha(t)
        lab = self.tracker.labels(t)
        st = self.tracker.store
        if al is None or lab is None:
            return None
        idx = [st.index(p.id) for p in self.cut.targets() if st.has(p.id)]
        m = cv2.resize(np.isin(lab, idx).astype(np.uint8), (self.proxy.w, self.proxy.h), interpolation=cv2.INTER_NEAREST)
        return np.clip(al * cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0), 0, 1)

    def _separate_preview(self, t: int, img: np.ndarray, fr: F.FieldResult) -> np.ndarray:
        from ..warp.separate import composite
        ta = self.target_alpha(t)
        if ta is None:
            return F.warp_image(img, fr)
        st = self.tracker.store

        def neighbour(f):
            if not (0 <= f < self.proxy.n):
                return None
            pa = self._people_alpha(f)
            if pa is None:
                return None
            return [np.asarray(self.proxy.frame(f))], cv2.dilate(pa, np.ones((5, 5), np.uint8))

        out, _, _ = composite([img], [(1, 1)], fr, ta, t, neighbour, st.homography,
                              (self.proxy.sx, self.proxy.sy), 255)
        return out[0]

    # ------------------------------------------------------------------ joints
    def joint_hit(self, t: int, x_src: float, y_src: float, radius_src: float) -> Optional[Tuple[int, int]]:
        """(pid, joint) nearest to a source-px position."""
        st = self.tracker.store if self.tracker else None
        if st is None:
            return None
        best = None
        for p in self.cut.persons:
            kp = st.kp(p.id, t)
            if kp is None or p.role == "ignore":
                continue
            d = np.hypot(kp[:, 0] - x_src, kp[:, 1] - y_src)
            d = np.where(np.isfinite(d), d, np.inf)
            j = int(np.argmin(d))
            if d[j] < radius_src and (best is None or d[j] < best[0]):
                best = (d[j], p.id, j)
        return (best[1], best[2]) if best else None

    def set_joint_live(self, pid: int, j: int, t: int, xy) -> None:
        st = self.tracker.store
        p = st.index(pid)
        st.final[p, t, j, :2] = xy
        st.final[p, t, j, 2] = 1.0
        self.interactive = True
        self.touch("geo")

    def commit_joint(self, pid: int, j: int, t: int, xy) -> None:
        self.cut.joint_keys.setdefault(pid, {}).setdefault(j, {})[int(t)] = [float(xy[0]), float(xy[1])]
        self.cut.stats.corrections += 1
        log_event("correction", self.project.name, self.cut.name, count=1)
        self.touch()

    def delete_joint_key(self, pid: int, j: int, t: int) -> bool:
        fr = self.cut.joint_keys.get(pid, {}).get(j, {})
        if int(t) in fr:
            del fr[int(t)]
            self.touch()
            return True
        return False

    def resolve_local(self, pid: int, t: int, frame_end: Optional[int] = None):
        rng = self.tracker.resolve_local(pid, t, frame_end=frame_end)
        self.touch()
        return rng

    def ghost_hit(self, pid: int, t: int, x_src: float, y_src: float, radius_src: float) -> Optional[int]:
        """Hidden joint of `pid` whose best guess is near the point (to revive it by dragging)."""
        if self.tracker is None or self.tracker.store is None:
            return None
        g = self.tracker.ghost(pid, t)
        best = None
        for j, v in g.items():
            d = float(np.hypot(v[0] - x_src, v[1] - y_src))
            if d < radius_src and (best is None or d < best[0]):
                best = (d, j)
        return best[1] if best else None

    def joint_key_frames(self, pid: int) -> List[int]:
        return sorted({f for fr in self.cut.joint_keys.get(pid, {}).values() for f in fr})

    def person_at(self, t: int, x_src: float, y_src: float) -> Optional[int]:
        if self.tracker is None or self.tracker.store is None:
            return None
        lab = self.tracker.labels(t)
        if lab is None:
            return None
        mx = int(np.clip(x_src * self.tracker.matte.w / self.cut.info.width, 0, self.tracker.matte.w - 1))
        my = int(np.clip(y_src * self.tracker.matte.h / self.cut.info.height, 0, self.tracker.matte.h - 1))
        p = int(lab[my, mx])
        if p >= 0 and self.tracker.matte is not None:
            al = self.tracker.matte.get(t)
            if al is not None and al.shape == lab.shape and al[my, mx] < 128:
                return None             # background near the body is nobody's
        return self.tracker.store.pids[p] if p >= 0 else None
