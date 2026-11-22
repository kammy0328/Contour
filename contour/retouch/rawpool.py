"""The masks of single frames (engine._prepare_raw) made ahead in helper processes, for the render.

The render's serial stage is the preparation of a frame: its masks on their own (the raw frame, about two thirds of the stage) and then those masks
held steady with the frames round it. The raw frames depend on nothing but the frame itself (its pictures, the tracks, the values), so they can be
made anywhere: here they are made ahead by helper processes, each a whole Python of its own (the GIL holds one process to about one core in this
stage), and handed to engine._raw_cached, which takes a frame from here before it would make it itself. The steady part stays where it was, in the
render's own order. What a frame gets is what it would have got: the same function on the same inputs (the helpers are given the engine's settings and
the cut's values as they are when the render starts). A frame a helper did not make, or could not, is made by the render as before.

Frames are given to the helpers in runs of BLOCK frames, so each helper reads its pictures in order (the readers of the original restart on a jump).

Used by warp.render when the environment says CONTOUR_RAW_PROCS > 0 (and only for footage with tracks). CONTOUR_RAW_SETUP: "module:function" run first
in a helper (with the render's sys.path), for what the caller set up in its own process before the render (DLL paths, settings).
"""
from __future__ import annotations

import os
import pickle
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional

BLOCK = 6            # frames given to one helper in a row
AHEAD = 6            # frames made (or being made) ahead of the last one taken, at most (a raw 4K frame is 130 to 370 MB); CONTOUR_RAW_AHEAD


def _engine_settings(RE) -> dict:
    """The engine's module settings (upper-case names) that can be sent: what the caller set on it (iwaly_cfg, control.json) goes with them."""
    out = {}
    for k, v in vars(RE).items():
        if not k[:1].isupper() or not k.replace("_", "").isalnum() or k != k.upper():
            continue
        if callable(v) or isinstance(v, type) or type(v).__module__ not in ("builtins", "numpy"):
            continue
        try:
            pickle.dumps(v)
        except Exception:
            continue
        out[k] = v
    return out


def _module_settings(names) -> dict:
    out = {}
    for mn in names:
        m = sys.modules.get(mn)
        if m is None:
            continue
        out[mn] = _engine_settings(m)
    return out


def _settings_state() -> dict:
    try:
        from ..core.settings import settings
        s = settings()
        d = {}
        for k, v in vars(s).items():
            try:
                pickle.dumps(v)
                d[k] = v
            except Exception:
                pass
        return d
    except Exception:
        return {}


def _send(f, obj) -> None:
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
    f.flush()


def _send_bytes(f, b: bytes) -> None:
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
    f.flush()


def _recv(f):
    h = f.read(8)
    if len(h) < 8:
        raise EOFError
    n = struct.unpack("<Q", h)[0]
    b = bytearray()
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return pickle.loads(bytes(b))


class RawPool:
    """n helper processes making raw frames of `frames` (in order, BLOCK at a time) for the render of cut. take(s) -> the raw frame (as
    engine._raw_cached keeps it: masks in float16) or None when it is not made here."""

    def __init__(self, project, cut, frames: List[int], n: int, log_dir: Optional[str] = None):
        from . import engine as RE
        self.frames = list(frames)
        self.pos = {s: i for i, s in enumerate(self.frames)}
        self.n = max(1, int(n))
        self.lock = threading.Condition()
        self.done: Dict[int, object] = {}
        self.failed: set = set()
        self.asked: set = set()
        self.next_block = 0                       # index into frames of the next block to give out
        self.last_taken = -1                      # index of the last frame taken
        self.busy: Dict[int, int] = {}            # helper -> frames it has still to send
        self.closed = False
        try:
            self.ahead = max(BLOCK, int(os.environ.get("CONTOUR_RAW_AHEAD", "") or AHEAD))
        except ValueError:
            self.ahead = AHEAD
        self.stats = dict(made=0, taken=0, local=0, wait_s=0.0, bytes=0)
        self.err: List[str] = []
        d = tempfile.mkdtemp(prefix="contour_raw_")
        self.dir = d
        spec = dict(sys_path=[p for p in sys.path if p], setup=os.environ.get("CONTOUR_RAW_SETUP", ""),
                    project=str(project.path), cut_id=cut.id, cut=cut, engine=_engine_settings(RE),
                    modules=_module_settings(["contour.retouch.bodytone", "contour.retouch.detect", "contour.retouch.skin",
                                              "contour.retouch.bodyskin", "contour.retouch.bodyspots"]),
                    settings=_settings_state())
        self.spec_path = os.path.join(d, "spec.pkl")
        with open(self.spec_path, "wb") as fh:
            pickle.dump(spec, fh, protocol=pickle.HIGHEST_PROTOCOL)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(spec["sys_path"])
        env["CONTOUR_RAW_PROCS"] = "0"
        # each helper's own threads (OpenCV, the BLAS of numpy) kept few: a helper per core group, not every helper on every core
        hthr = str(max(1, int(os.environ.get("CONTOUR_HELPER_THREADS", "3") or 3)))
        for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CONTOUR_HELPER_THREADS"):
            env[k_] = hthr
        self.procs = []
        self.logs = []
        for k in range(self.n):
            lp = os.path.join(log_dir or d, f"rawhelper_{k}.log")
            lf = open(lp, "w", encoding="utf-8")
            self.logs.append(lp)
            flags = 0x08000000 if os.name == "nt" else 0       # (no console window)
            p = subprocess.Popen([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "rawhelper.py"), self.spec_path], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=lf, env=env, creationflags=flags)
            self.procs.append(p)
            self.busy[k] = 0
            threading.Thread(target=self._reader, args=(k,), daemon=True, name=f"raw-read-{k}").start()
        with self.lock:
            self._feed()

    # ---- giving out work
    def _feed(self) -> None:
        """(lock held) blocks to idle helpers while the frames made ahead are fewer than AHEAD"""
        for k, p in enumerate(self.procs):
            if self.closed or self.next_block >= len(self.frames):
                return
            if self.busy.get(k, 0) > 0 or p.poll() is not None:
                continue
            if self.next_block - self.last_taken > self.ahead:
                return
            blk = self.frames[self.next_block:self.next_block + BLOCK]
            self.next_block += len(blk)
            try:
                _send(p.stdin, ("make", blk))
                self.busy[k] = len(blk)
                self.asked.update(blk)
            except Exception as e:
                self.err.append(f"helper {k}: {e!r}")
                self.failed.update(blk)
                self.lock.notify_all()

    def _reader(self, k: int) -> None:
        p = self.procs[k]
        try:
            while True:
                h = p.stdout.read(8)
                if len(h) < 8:
                    raise EOFError
                nb_ = struct.unpack("<Q", h)[0]
                buf = p.stdout.read(nb_)
                if len(buf) < nb_:
                    raise EOFError
                msg = pickle.loads(buf)
                kind = msg[0]
                with self.lock:
                    if kind == "raw":
                        _, s, rf, _nb = msg
                        self.done[s] = rf
                        self.stats["made"] += 1
                        self.stats["bytes"] += int(nb_)
                    elif kind == "fail":
                        _, s, tb = msg
                        self.failed.add(s)
                        self.err.append(f"f{s}: {tb[-300:]}")
                    elif kind == "ready":
                        pass
                    if kind in ("raw", "fail"):
                        self.busy[k] = max(0, self.busy.get(k, 0) - 1)
                    self._feed()
                    self.lock.notify_all()
        except Exception:
            with self.lock:
                self.busy[k] = 0
                # what this helper had still to make is made by the render
                for s in list(self.asked):
                    if s not in self.done:
                        pass
                self.err.append(f"helper {k} ended (rc {p.poll()})")
                self._dead(k)
                self.lock.notify_all()

    def _dead(self, k: int) -> None:
        """(lock held) a helper is gone: the frames given to it and not made are no longer waited for"""
        alive = any(p.poll() is None for p in self.procs)
        if not alive:
            for s in self.asked:
                if s not in self.done:
                    self.failed.add(s)

    # ---- taking
    def take(self, s: int, timeout: float = 600.0):
        """the raw frame s when a helper makes it (waits for it), or None (the caller makes it)"""
        if s not in self.pos:
            return None
        t0 = time.time()
        with self.lock:
            i = self.pos[s]
            if i > self.last_taken:
                self.last_taken = i
                self._feed()
            if s not in self.asked and s not in self.done:
                # not given out yet (the render is ahead of the helpers): the render makes it, and the helpers skip ahead past it
                if self.next_block <= i:
                    self.next_block = i + 1
                    self._feed()
                self.stats["local"] += 1
                return None
            while s not in self.done and s not in self.failed and not self.closed:
                if not any(p.poll() is None for p in self.procs):
                    break
                if time.time() - t0 > timeout:
                    break
                self.lock.wait(0.5)
            self.stats["wait_s"] += time.time() - t0
            rf = self.done.pop(s, None)
            if rf is None:
                self.stats["local"] += 1
                return None
            self.stats["taken"] += 1
            return rf

    def close(self) -> dict:
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        for p in self.procs:
            try:
                _send(p.stdin, ("quit",))
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        st = dict(self.stats)
        st["errors"] = self.err[:5]
        st["helpers"] = self.n
        return st


def start_for_render(project, cut, tracker, a: int, b: int) -> Optional[RawPool]:
    """A pool for the frames a..b-1 of a render (and STEADY_R round them), when CONTOUR_RAW_PROCS asks for one; put on the tracker for
    engine._raw_cached. None when it is not asked for or cannot start."""
    try:
        n = int(os.environ.get("CONTOUR_RAW_PROCS", "0") or 0)
    except ValueError:
        n = 0
    if n <= 0 or tracker is None or getattr(tracker, "store", None) is None:
        return None
    try:
        from . import engine as RE
        N = int(tracker.store.final.shape[1])
        r = int(getattr(RE, "STEADY_R", 2))
        frames = list(range(max(0, a - r), min(N, b + r)))
        pool = RawPool(project, cut, frames, n, log_dir=os.environ.get("CONTOUR_RAW_LOGDIR") or None)
        tracker.__dict__["_raw_pool"] = pool
        return pool
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def stop_for_render(tracker) -> Optional[dict]:
    pool = tracker.__dict__.pop("_raw_pool", None) if tracker is not None else None
    if pool is None:
        return None
    try:
        return pool.close()
    except Exception:
        return None
