"""Finished frames kept for playback, for every cut looked at in this run of the app (instruction 2026-11-20 §1-3).

Until 1.22.2 the playback cache belonged to one opening of one cut: its keys carried the number of that opening
(the session's epoch), so going A -> B -> A rendered A again from nothing, and every cut prepared ahead for playback
had a budget of its own.

- PlayCache: the finished frames (JPEG) of all cuts in one place, at most `limit` bytes together. An entry is found
  by the cut it belongs to (project path, cut id), the frame and the key of the result: what the frames are read
  from (file, version of the file, stretch, size, colour view), the settings a result depends on and the processing
  revisions of the cut. Nothing in a key says when it was made: the number of an opening (the generation of a
  request) is only used by the session to drop the results of work started before a cut was changed.
- Revisions: the processing revisions a cut had when it was left, with the state they stand for (state_digest).
  Opened again in the same state, the cut gets the same revisions and so finds its frames; in any other state it
  gets new ones and what was kept for it is dropped.
- state_digest: everything a finished frame of a cut depends on that can change while the cut is not open: its
  values in the project (retouch amounts, joints, people, range ...), the files an analysis or an edit writes for
  it (tracks, mattes and their revision, face layer, clothing state), the settings that choose models or colours,
  and what its frames are read from.

The proxy file on disk has its own check (Proxy.is_valid: view, size, frame count, version of the original); this
cache never asks whether there is a proxy, only what the pixels come from (frames.origin)."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_MB = 1200

# values of a cut that no finished frame depends on (names, notes, review state, counters, output file settings)
SKIP_CUT = ("name", "qc", "stats", "created", "benchmark_id", "learn", "pending_copy", "retouch_ref",
            "style_learned", "tl", "status", "version", "versions", "output")
# files in a cut's folder that no finished frame depends on
SKIP_FILES = ("prescan.json", "batch_result.json", "identity.npz", "matte_prompts.npz", "matte_undo.npz",
              "matte_swap.json", "matte_stage.json")
# settings a finished frame depends on (the colour view and the frame size are part of frames.origin as well)
SETTING_KEYS = ("display", "view", "proxy_long_side", "face_model", "cloth_model", "unified_model", "gpu")

_REVS = itertools.count(1)


def new_revision() -> int:
    """A processing revision: a number no cut, session or earlier state ever had."""
    return next(_REVS)


def new_revisions() -> dict:
    return {"geo": new_revision(), "ret": new_revision(), "trk": new_revision()}


# ---------------------------------------------------------------------------------------------- the state of a cut
def _stamps(folder: Path) -> list:
    """The files an analysis or an edit writes for a cut: name, size and (for files written whole) the time.
    Mapped files (.u8) are told by name and size only: their time moves late on Windows; what is written into
    them goes with a meta file written whole (seg_meta.json, the clothing state)."""
    out = []
    for d in (folder, folder / "cache"):
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            continue
        for e in entries:
            n = e.name
            if n.startswith("proxy") or n in SKIP_FILES or "recovery" in n or ".tmp" in n or ".stage" in n:
                continue
            ext = os.path.splitext(n)[1].lower()
            if ext not in (".json", ".npz", ".u8"):
                continue
            try:
                if not e.is_file():
                    continue
                st = e.stat()
            except OSError:
                continue
            rel = n if d is folder else "cache/" + n
            out.append((rel, st.st_size) if ext == ".u8" else (rel, st.st_size, st.st_mtime_ns))
    return out


def setting_values() -> tuple:
    try:
        from .settings import settings
        s = settings()
        return tuple((k, str(getattr(s, k, ""))) for k in SETTING_KEYS)
    except Exception:
        return ()


def _model_versions() -> list:
    """The model packages in use (instruction 2026-11-20 §4-2: the keys of results carry the versions they were
    made with). Models that are not packages are told by the settings and the cut's own record."""
    from ..track import modelpkg
    return sorted(modelpkg.store().active_versions().items())


EXTRA: List[Callable[[], object]] = [_model_versions]      # more that a finished frame depends on


def state_digest(project, cut, origin=()) -> str:
    """One value for the state a cut's finished frames are made from (see the module text)."""
    d = {k: v for k, v in cut.to_dict().items() if k not in SKIP_CUT}
    folder = Path(project.folder) / "cuts" / cut.id
    extra = []
    for f in EXTRA:
        try:
            extra.append(f())
        except Exception:
            extra.append(None)
    blob = json.dumps([d, _stamps(folder), setting_values(), repr(origin), repr(extra)], sort_keys=True,
                      ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


class Revisions:
    """cut -> (state digest, processing revisions) as of the last time the cut was left with everything saved."""

    def __init__(self):
        self._d: Dict[tuple, Tuple[str, dict]] = {}
        self._lock = threading.Lock()

    def remember(self, cut_key: tuple, digest: str, vers: dict) -> None:
        with self._lock:
            self._d[cut_key] = (digest, dict(vers))

    def recall(self, cut_key: tuple, digest: str) -> Optional[dict]:
        """The revisions the cut had in this state, or None (never opened, or its state is another one now)."""
        with self._lock:
            e = self._d.get(cut_key)
        if e is None or e[0] != digest:
            return None
        return dict(e[1])

    def forget(self, cut_key: Optional[tuple] = None) -> None:
        with self._lock:
            if cut_key is None:
                self._d.clear()
            else:
                self._d.pop(cut_key, None)


# ---------------------------------------------------------------------------------------------- the frames
class PlayCache:
    """Finished frames of every cut, at most `limit` bytes for all of them together.

    Over the limit, in this order: frames of the cut being written whose key is no longer its current one; frames
    of other cuts (the cut used longest ago first, of a cut the frames furthest from where it was last looked at);
    then the frames of the cut being written furthest from where it is looked at now."""

    def __init__(self, limit: int):
        self.limit = int(limit)
        self._cuts: "OrderedDict[tuple, Dict[int, Tuple[tuple, bytes]]]" = OrderedDict()
        self._near: Dict[tuple, int] = {}
        self.bytes = 0
        self.peak = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ reading
    def get(self, cut_key: tuple, t: int, key: tuple) -> Optional[bytes]:
        with self._lock:
            e = self._cuts.get(cut_key, {}).get(int(t))
            if e is None or e[0] != key:
                self.misses += 1
                return None
            self.hits += 1
            return e[1]

    def has(self, cut_key: tuple, t: int, key: tuple) -> bool:
        with self._lock:
            e = self._cuts.get(cut_key, {}).get(int(t))
            return e is not None and e[0] == key

    def frames(self, cut_key: tuple) -> List[Tuple[int, tuple]]:
        """(frame, key) of everything kept for a cut (the caller tells which keys are its current ones)."""
        with self._lock:
            return [(t, e[0]) for t, e in self._cuts.get(cut_key, {}).items()]

    def size(self, cut_key: Optional[tuple] = None) -> int:
        with self._lock:
            if cut_key is None:
                return self.bytes
            return sum(len(e[1]) for e in self._cuts.get(cut_key, {}).values())

    def cuts(self) -> List[tuple]:
        with self._lock:
            return list(self._cuts)

    # ------------------------------------------------------------------ writing
    def use(self, cut_key: tuple, near: Optional[int] = None) -> None:
        """The cut is the one being looked at (the last to give up frames)."""
        with self._lock:
            if cut_key in self._cuts:
                self._cuts.move_to_end(cut_key)
            if near is not None:
                self._near[cut_key] = int(near)

    def put(self, cut_key: tuple, t: int, key: tuple, data: bytes, near: int = 0,
            current: Optional[Callable[[int, tuple], bool]] = None) -> None:
        t = int(t)
        with self._lock:
            d = self._cuts.get(cut_key)
            if d is None:
                d = self._cuts[cut_key] = {}
            self._cuts.move_to_end(cut_key)
            self._near[cut_key] = int(near)
            old = d.pop(t, None)
            if old is not None:
                self.bytes -= len(old[1])
            d[t] = (key, data)
            self.bytes += len(data)
            if self.bytes > self.limit:
                self._trim(cut_key, t, near, current)
            self.peak = max(self.peak, self.bytes)

    def _trim(self, cut_key: tuple, keep_t: int, near: int, current) -> None:
        """With the lock held."""
        goal = int(self.limit * 0.9)
        d = self._cuts[cut_key]

        def take(owner, frames) -> bool:
            dd = self._cuts[owner]
            for f in frames:
                if self.bytes <= goal:
                    return True
                self.bytes -= len(dd.pop(f)[1])
            return self.bytes <= goal

        stale = []
        if current is not None:
            stale = [f for f, e in d.items() if f != keep_t and not current(f, e[0])]
        done = take(cut_key, stale)
        if not done:
            for other in [k for k in self._cuts if k != cut_key]:
                dd = self._cuts[other]
                at = self._near.get(other, 0)
                if take(other, sorted(dd, key=lambda f: -abs(f - at))):
                    done = True
                    break
        if not done:
            take(cut_key, sorted((f for f in d if f != keep_t), key=lambda f: -abs(f - near)))
        for k in [k for k, dd in self._cuts.items() if not dd and k != cut_key]:
            del self._cuts[k]
            self._near.pop(k, None)

    def drop(self, cut_key: Optional[tuple] = None, keep: Optional[Callable[[tuple], bool]] = None) -> int:
        """Forgets a cut's frames (all cuts: None), or only those keep(key) is False for. Bytes given back."""
        freed = 0
        with self._lock:
            for k in ([cut_key] if cut_key is not None else list(self._cuts)):
                dd = self._cuts.get(k)
                if dd is None:
                    continue
                for f in [f for f, e in dd.items() if keep is None or not keep(e[0])]:
                    freed += len(dd.pop(f)[1])
                if not dd:
                    del self._cuts[k]
                    self._near.pop(k, None)
            self.bytes -= freed
        return freed

    def drop_project(self, project_path: str) -> int:
        with self._lock:
            ks = [k for k in self._cuts if k and k[0] == project_path]
        return sum(self.drop(k) for k in ks)

    def set_limit(self, limit: int) -> None:
        with self._lock:
            self.limit = int(limit)
            goal = self.limit
            for k in list(self._cuts):
                if self.bytes <= goal:
                    break
                dd = self._cuts[k]
                at = self._near.get(k, 0)
                for f in sorted(dd, key=lambda f: -abs(f - at)):
                    if self.bytes <= goal:
                        break
                    self.bytes -= len(dd.pop(f)[1])
                if not dd:
                    del self._cuts[k]
                    self._near.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._cuts.clear()
            self._near.clear()
            self.bytes = 0
            self.peak = 0
            self.hits = self.misses = 0


_cache: Optional[PlayCache] = None
_revs = Revisions()
_glock = threading.Lock()


def limit_bytes() -> int:
    mb = DEFAULT_MB
    try:
        from .settings import settings
        mb = int(getattr(settings(), "play_cache_mb", DEFAULT_MB) or DEFAULT_MB)
    except Exception:
        pass
    mb = int(os.environ.get("DIAP_PLAY_CACHE_MB", mb))
    return max(16, mb) * 1024 * 1024


def cache() -> PlayCache:
    global _cache
    with _glock:
        if _cache is None:
            _cache = PlayCache(limit_bytes())
        return _cache


def revisions() -> Revisions:
    return _revs


def memory() -> dict:
    """Bytes held by the caches that share the app's frame memory (diagnosis, the AI review report)."""
    from . import frames as FR
    p = FR.pool()
    c = cache()
    return dict(frames=dict(bytes=p.bytes, limit=p.limit, peak=p.peak),
                playback=dict(bytes=c.bytes, limit=c.limit, peak=c.peak, cuts=len(c.cuts()), hits=c.hits,
                              misses=c.misses))
