"""Edit session: the open project / cut, preview rendering and edit operations (UI-agnostic)."""
from __future__ import annotations

import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core import playcache as PC
from ..core.colour import PASSTHROUGH, ViewSettings, manager
from ..core.project import Cut, Project
from ..core.proxy import Proxy
from ..core.settings import settings
from ..stats.tracker import log_event
from ..track import skeleton as S
from ..track.analyze import CutTracker
from ..warp import field as F


_EPOCHS = __import__("itertools").count(1)


def _short(x) -> str:
    import hashlib
    return hashlib.sha1(repr(x).encode("utf-8")).hexdigest()[:16]


class EditSession:
    """epoch: the generation of the open cut, a new number at every opening of a cut (unique over every session of
    the app). Work notes it when it starts and its result is dropped when the number is another one by the time it
    is done: nothing made across a change of cut is kept or shown. It is in no cache key (instruction 2026-11-20
    §1-3; until 1.22.2 it was, so a cut opened again never found the frames made for it the last time).

    The keys of results say what a result is made from instead: the cut (cut_key: project path and cut id), what
    its frames are read from and the settings that choose colours and models (src_key), and the processing
    revisions of the cut (vers: one number each for the shape values, the retouch values and the tracking; a new
    number, never used before, whenever one of them changes). A cut left with everything saved keeps its
    revisions (core.playcache.Revisions) and gets them back when it is opened again in the same state, so the
    finished frames kept for it (core.playcache.PlayCache: all cuts, one memory budget) are found again."""

    def __init__(self):
        self.project: Optional[Project] = None
        self.cut: Optional[Cut] = None
        self.proxy: Optional[Proxy] = None
        self.tracker: Optional[CutTracker] = None
        self.version = 0                     # bump to invalidate previews (any change)
        self.epoch = next(_EPOCHS)
        self.vers = PC.new_revisions()       # what changed: shape params / retouch params / tracking
        self.cut_key: tuple = ("", "")       # (project path, cut id) of the open cut
        self.src_key = ""                    # what its frames are read from, the colour view, the model settings
        self._origin: tuple = ()
        self._home = None                    # (project, cut) this session speaks for when the cut is left
        self.kept = 0                        # finished frames found again when the cut was opened (diagnosis)
        self.interactive = False             # a slider is being dragged: coarser preview field
        self._cache: "OrderedDict[tuple, object]" = OrderedDict()
        self._frames: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._mask_cache: dict = {}
        self._last_field = None
        self._lock = threading.RLock()
        self._compute = threading.RLock()    # one heavy frame computation at a time (preview / playback cache)
        self.last_touch = 0.0
        self.last_seek = 0.0          # the user moved to another frame (the playback cache waits a moment)
        self.preview_wait = 0         # preview frames waiting: the playback cache gives way to them
        self.dirty = False

    @property
    def play_limit(self) -> int:
        """Bytes the finished frames of all cuts may take together (설정: play_cache_mb)."""
        return PC.cache().limit

    @play_limit.setter
    def play_limit(self, v: int) -> None:
        PC.cache().set_limit(int(v))

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
            self._new_epoch()                 # first: work started on the cut being left drops what it makes
            self._remember()
            if self.proxy is not None:
                self.proxy.close()
            self.cut = cut
            self.proxy = None
            self.tracker = None
            self._cache.clear()
            self._frames.clear()
            self._mask_cache = {}
            self._last_field = None
            self.cut_key, self.src_key, self._origin = ("", ""), "", ()
            self.version += 1
            if cut is None or self.project is None:
                self.vers = PC.new_revisions()
                return
            t0 = time.perf_counter()
            px = self.make_frames(cut)
            t1 = time.perf_counter()
            if px is not None:
                self.proxy = px
                self.tracker = CutTracker(self.project, cut, px)
            t2 = time.perf_counter()
            self._enter(cut, px)
            # seconds, for the project-open timing: the frames source, the tracker (with tracks.npz), the state
            # the cut's kept frames are checked against
            self.open_times = {"frames": t1 - t0, "tracker": t2 - t1, "state": time.perf_counter() - t2}

    def _enter(self, cut: Cut, px) -> None:
        """With the lock held, the cut just opened: its keys, and the processing revisions it had the last time it
        was left if it is in the same state (then the finished frames kept for it are found again); else new
        ones, and what was kept for it is given back."""
        from ..core import frames as FR
        prj = self.project
        self.cut_key = (str(prj.path), cut.id)
        try:
            self._origin = FR.origin(px, cut.info, self.view_settings(cut))
        except Exception:
            self._origin = ("?", next(_EPOCHS))
        self.src_key = _short((self._origin, PC.setting_values()))
        self._home = (prj, cut)
        vers = None
        if px is not None:
            try:
                vers = PC.revisions().recall(self.cut_key, PC.state_digest(prj, cut, self._origin))
            except Exception:
                import traceback
                traceback.print_exc()
        if vers is None:
            self.vers = PC.new_revisions()
            PC.cache().drop(self.cut_key)
            self.kept = 0
        else:
            self.vers = vers
            self.kept = len(self.play_frames())
        PC.cache().use(self.cut_key)

    def _remember(self) -> None:
        """With the lock held, before another cut takes this one's place: the cut keeps its processing revisions
        for the next time it is opened in this state. Not with changes that were never saved: they are gone when
        the cut is opened again, and what was made from them with them."""
        home, self._home = self._home, None
        if home is None:
            return
        ck = self.cut_key
        if self.dirty or self.proxy is None:
            PC.revisions().forget(ck)
            return
        try:
            PC.revisions().remember(ck, PC.state_digest(home[0], home[1], self._origin), self.vers)
        except Exception:
            import traceback
            traceback.print_exc()
            PC.revisions().forget(ck)

    def adopt(self, other: "EditSession") -> None:
        """Takes over a cut another session opened and pre-rendered in the background (playback going on into the
        next cut without a pause): its proxy, tracker, caches and the revisions their keys were made with."""
        with self._lock:
            self._new_epoch()
            self._remember()
            if self.proxy is not None and self.proxy is not other.proxy:
                self.proxy.close()
            with other._lock:
                self.cut, self.proxy, self.tracker = other.cut, other.proxy, other.tracker
                self._cache = OrderedDict(other._cache)
                self._frames = OrderedDict(other._frames)
                self._mask_cache = dict(other._mask_cache)
                if isinstance(self._mask_cache.get("__tok__"), dict):   # (the cache's own table: its own copy too)
                    self._mask_cache["__tok__"] = dict(self._mask_cache["__tok__"])
                self._last_field = None
                self.vers = dict(other.vers)
                self.cut_key, self.src_key, self._origin = other.cut_key, other.src_key, other._origin
                self._home, other._home = other._home, None     # the cut is this session's from now on
            self.version += 1
            PC.cache().use(self.cut_key)

    def _new_epoch(self) -> None:
        with self._lock:
            self.epoch = next(_EPOCHS)

    def generation(self) -> int:
        with self._lock:
            return self.epoch

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
                self.vers[k] = PC.new_revision()
            if "trk" in kinds:
                self._mask_cache = self._masks_after_tracking()
            # entries keyed by stale versions simply stop being hit; drop the obviously stale ones
            for key in [k for k in self._cache if not self._valid(k)]:
                self._cache.pop(key, None)
        self.dirty = True

    def _masks_after_tracking(self) -> dict:
        """The mask cache to go on with when tracking changed: empty, but for the bodies' areas and spots, which are
        kept with what the tracks said of the frames they were made from and made again only where that changed
        (retouch.engine.kept_over_tracks; instruction 2026-11-20 §2)."""
        if not self._mask_cache:
            return {}
        try:
            from ..retouch.engine import kept_over_tracks
            return kept_over_tracks(self._mask_cache)
        except Exception:
            return {}

    def _valid(self, key) -> bool:
        kind = key[0]
        v = self.vers
        if kind == "field":
            return key[2:4] == (v["geo"], v["trk"])
        if kind in ("retouch", "ret_img"):
            return key[2:4] == (v["ret"], v["trk"])
        if kind == "res":
            return key[2:5] == (v["geo"], v["ret"], v["trk"])
        return False

    # ------------------------------------------------------------------ preview
    def frame_peek(self, t: int) -> Optional[np.ndarray]:
        """Frame t only if it can be had without decoding (for the UI thread): from this session's frames, a proxy's
        mapped file, or the decoded-frame pool. None means: ask frame() on a worker thread."""
        with self._lock:
            px = self.proxy
            if px is None:
                return None
            f = self._frames.get(t)
            if f is not None:
                return f
        if getattr(px, "kind", "") != "source":
            return self.frame(t)
        a = px.peek(t)
        if a is None:
            return None
        with self._lock:
            if self.proxy is px:
                self._frames[t] = a
                while len(self._frames) > 8:
                    self._frames.popitem(last=False)
        return a

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
        with self._lock:
            ep = self.epoch                # the opening this field is made in; kept only if it is still that one
            key = ("field", t, self.vers["geo"], self.vers["trk"], coarse)
            fine = ("field", t, self.vers["geo"], self.vers["trk"], False)
            if key in self._cache:
                return self._cache[key]
            if coarse and fine in self._cache:
                return self._cache[fine]
            cut, tr, px = self.cut, self.tracker, self.proxy
        if px is None or tr is None or cut is None:     # the cut was closed while this was being asked for
            return None
        fi = F.build_input(cut, tr, t)
        div = 240 if coarse else 480
        fr = F.compute_field(fi, px.w, px.h, max(2, int(round(px.w / div))))
        if not self._put(key, fr, ep):
            return None                    # another cut was opened meanwhile: this is not a field of it
        self._last_field = (t, fr)
        return fr

    def peek_field(self, t: int) -> Optional[F.FieldResult]:
        """The newest field already computed for frame t (current settings first, else the last one), without
        computing anything: for drawing overlays where the picture is."""
        with self._lock:
            best = None
            for key, val in reversed(self._cache.items()):
                if key[0] != "field" or key[1] != t:
                    continue
                if key[2:4] == (self.vers["geo"], self.vers["trk"]):
                    return val
                if best is None:
                    best = val
            if best is None:
                lf = self._last_field
                if lf is not None and lf[0] == t:
                    best = lf[1]
            return best

    # ------------------------------------------------------------------ playback cache
    def play_get(self, t: int) -> Optional[np.ndarray]:
        """Finished result of frame t from the playback cache (current settings only)."""
        if self.cut is None:
            return None
        data = PC.cache().get(self.cut_key, t, self.result_key(t, True))
        if data is None:
            return None
        a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    def play_has(self, t: int) -> bool:
        return self.cut is not None and PC.cache().has(self.cut_key, t, self.result_key(t, True))

    def play_put(self, t: int, key: tuple, img: np.ndarray, near: int = 0) -> None:
        """Keeps a finished frame under its key (the cut it belongs to is in the key: a frame finished for a cut
        that is no longer the open one still goes to that cut). Over the memory budget of all cuts together:
        see PlayCache."""
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return
        ck = key[-2]
        current = None
        if ck == self.cut_key:
            tail = self.result_key(0, True)[2:]
            current = lambda f, k: k[2:] == tail and k[1] == f      # noqa: E731
        PC.cache().put(ck, t, key, buf.tobytes(), near=near, current=current)

    def play_render(self, t: int, near: int = 0) -> bool:
        """Frame t's finished result into the playback cache. False: it could not be made, or the cut or its
        values changed while it was being made (then nothing is kept: it is the picture of neither state)."""
        with self._lock:
            ep = self.epoch
            if self.cut is None:
                return False
            key = self.result_key(t, True)
        img = self.result(t, final=True)
        with self._lock:
            if img is None or self.epoch != ep or self.cut is None or self.result_key(t, True) != key:
                return False
        self.play_put(t, key, img, near=near)
        return True

    def play_frames(self) -> List[int]:
        if self.cut is None:
            return []
        tail = self.result_key(0, True)[2:]
        return [f for f, k in PC.cache().frames(self.cut_key) if k[2:] == tail and k[1] == f]

    def play_bytes(self) -> int:
        """Bytes the open cut's finished frames take (all cuts: PC.cache().bytes)."""
        return PC.cache().size(self.cut_key) if self.cut is not None else 0

    def _put(self, key, val, ep: Optional[int] = None) -> bool:
        """Keeps an intermediate result of the open cut. ep: the generation the work started in; False (nothing
        kept) when another cut was opened since."""
        with self._lock:
            if ep is not None and self.epoch != ep:
                return False
            self._cache[key] = val
            while len(self._cache) > 40:
                self._cache.popitem(last=False)
        return True

    def result_key(self, t: int, final: bool = False) -> tuple:
        """What frame t's result is made from: the processing revisions (shape, retouch, tracking), whether it is
        the coarse one of a slider being dragged, the mode, the cut (project path and cut id) and what its frames
        are read from with the settings that choose colours and models. Nothing about when it was asked for:
        the same cut opened again in the same state has the same keys."""
        return ("res", t, self.vers["geo"], self.vers["ret"], self.vers["trk"], False if final else self.interactive,
                self.cut.mode if self.cut else "", self.cut_key, self.src_key)

    def cached_result(self, t: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._cache.get(self.result_key(t))

    def retouched(self, t: int) -> Optional[np.ndarray]:
        with self._lock:
            ep = self.epoch                 # before the frame is read: all of it within one opening of the cut
            px = self.proxy
        img = self.frame(t)
        if img is None or px is None:
            return None
        live = bool(self.interactive)
        key = ("ret_img", t, self.vers["ret"], self.vers["trk"], live)
        with self._lock:
            if self.epoch != ep:
                return None
            if key in self._cache:
                return self._cache[key]
            full = ("ret_img", t, self.vers["ret"], self.vers["trk"], False)
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
                ds = apply_rgb(small, rf, px.sx * 0.5).astype(np.int16) - small.astype(np.int16)
                d = _cv2.resize(ds, (w, h), interpolation=_cv2.INTER_LINEAR)
                img = np.clip(img.astype(np.int16) + d, 0, 255).astype(np.uint8)
            else:
                img = apply_rgb(img, rf, px.sx)
        if not self._put(key, img, ep):
            return None
        return img

    def render(self, t: int, final: bool = False):
        """(key, picture, generation) of frame t's result, all of it made within one opening of the cut. None
        when there is no picture, or another cut was opened while it was being made: what was computed then is
        partly of one cut and partly of the other, and is neither kept nor handed out."""
        with self._lock:
            ep = self.epoch
            if self.cut is None or self.proxy is None:
                return None
            key = self.result_key(t, final)
            img = self._cache.get(key)
        if img is not None:
            return key, img, ep
        if self.frame(t) is None:
            return None
        with self._compute:
            img = self._result(t, key, final, ep)
        with self._lock:
            if img is None or self.epoch != ep:
                return None
        return key, img, ep

    def prewarm_skin(self, t: int, stop=None) -> bool:
        """(1.26.1) A cut just opened: what the face retouch needs of frame t made ahead, on the caller's thread, while
        nobody works on it, with every value the basic adjustment can switch on, on a copy of the cut (the cut itself
        is not touched). The first 표준 / 약하게 / 강하게 of the frame then finds the frame's detections, colour models and
        face layer made (they are kept by what they are made from, whatever values are on): measured 11 s for the first
        picture of a 6-person 2560 px frame cold, under 1 s warm. False when another cut was opened or work came first."""
        import copy as _copy
        with self._lock:
            ep, cut, tr, masks = self.epoch, self.cut, self.tracker, self._mask_cache
        if cut is None or tr is None or tr.store is None or not cut.analyzed:
            return False
        from ..core.params import SKIN_LEVEL_VALUES
        from ..core.project import Curve
        from ..retouch.engine import prepare
        top = {}
        for vals in SKIN_LEVEL_VALUES.values():
            for k, v in vals.items():
                top[k] = max(top.get(k, 0.0), float(v))
        c2 = _copy.deepcopy(cut)
        for person in c2.targets():
            pp = c2.params_for(person.id)
            for k, v in top.items():
                cur = pp.amounts.get(k)
                if v > 0 and (cur is None or (abs(cur.base) < 1e-4 and not cur.keys)):
                    pp.amounts[k] = Curve(base=v)
        if stop is not None and stop():
            return False
        with self._compute:
            with self._lock:
                if self.epoch != ep or self.cut is not cut:
                    return False
            if stop is not None and stop():
                return False
            try:
                prepare(c2, tr, t, masks, quick=True)
            except Exception:
                traceback.print_exc()
                return False
            # the tracker's table of raw frames was made for the copy's values: started again for the cut's own
            tr.__dict__.pop("_steady_lru", None)
            tr.__dict__.pop("_steady_sig", None)
        return True

    def render_quick(self, t: int):
        """(key, picture, generation): the first picture of frame t after a change (1.26.1, the user: "바로 바로 적용되게
        못하나? 로딩이 너무 오래 걸려"). This frame's own masks (engine.prepare quick: kept, so the steady result made right
        after uses them) with the values of now, laid on at half size and its change added back onto the full picture
        (as while a slider is dragged). None when the full result is there already or there is nothing to retouch."""
        with self._lock:
            ep = self.epoch
            cut, tr, px, masks = self.cut, self.tracker, self.proxy, self._mask_cache
            if cut is None or px is None or tr is None or tr.store is None or not cut.analyzed:
                return None
            if self.result_key(t) in self._cache:
                return None
            key = ("quick", t, self.vers["geo"], self.vers["ret"], self.vers["trk"], self.cut_key, self.src_key)
            img = self._cache.get(key)
        if img is not None:
            return key, img, ep
        img = self.frame(t)
        if img is None:
            return None
        from ..retouch.engine import apply_rgb, prepare
        with self._compute:
            with self._lock:
                if self.epoch != ep:
                    return None
            try:
                rf = prepare(cut, tr, t, masks, quick=True)
            except Exception:
                traceback.print_exc()
                rf = None
            if rf is not None and not rf.empty():
                h, w = img.shape[:2]
                small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                ds = apply_rgb(small, rf, px.sx * 0.5).astype(np.int16) - small.astype(np.int16)
                d = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
                img = np.clip(img.astype(np.int16) + d, 0, 255).astype(np.uint8)
            fr = self.field(t, None)
            with self._lock:
                if self.epoch != ep:
                    return None
            out = img if fr is None or fr.empty() else F.warp_image(img, fr)
        if not self._put(key, out, ep):
            return None
        return key, out, ep

    def result(self, t: int, final: bool = False) -> Optional[np.ndarray]:
        r = self.render(t, final)
        return None if r is None else r[1]

    def _result(self, t: int, key: tuple, final: bool, ep: int) -> Optional[np.ndarray]:
        with self._lock:
            if self.epoch != ep:
                return None
            if key in self._cache:
                return self._cache[key]
        # each step is made within one opening of the cut (it notes the generation itself); the generation checked
        # after each makes that the one this result was asked in
        img = self.retouched(t)
        with self._lock:
            if img is None or self.epoch != ep:
                return None
        fr = self.field(t, False if final else None)
        with self._lock:
            if self.epoch != ep:
                return None
        if fr is None or fr.empty():
            out = img
        else:
            out = F.warp_image(img, fr)
        if not self._put(key, out, ep):
            return None
        return out

    def retouch(self, t: int):
        """Retouch masks and values for frame t (None when nothing is active)."""
        with self._lock:
            ep = self.epoch
            cut, tr, masks = self.cut, self.tracker, self._mask_cache
            if tr is None or tr.store is None or cut is None:
                return None
            key = ("retouch", t, self.vers["ret"], self.vers["trk"])
            if key in self._cache:
                return self._cache[key]
        from ..retouch.engine import prepare
        try:
            rf = prepare(cut, tr, t, masks)
        except Exception:
            rf = None
        return rf if self._put(key, rf, ep) else None

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
