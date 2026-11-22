"""The retouch of a frame (engine.apply_planes) done in helper processes, for the render.

Once the frames' masks come from helper processes (rawpool), the render's own process is held by the retouch it applies: its Python steps (the knee
tone, the body's smoothing, the carrying back into the log) take the one interpreter lock, so four render threads work about like one. Here each frame's
retouch is handed to one of n helper processes (a whole Python each) with its pictures and its prepared frame, and the result comes back; the render's
thread waits for it. apply_planes depends on nothing but what it is given (no state is kept between frames), so a frame gets what it would have got.
A frame a helper could not do is done by the render as before.

Used by warp.render when CONTOUR_APPLY_PROCS > 0. CONTOUR_RAW_SETUP is run first in a helper, as for rawpool.
"""
from __future__ import annotations

import os
import pickle
import queue
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

from .rawpool import _engine_settings, _module_settings, _settings_state, _send_bytes


def _read_msg(f):
    h = f.read(8)
    if len(h) < 8:
        raise EOFError
    n = struct.unpack("<Q", h)[0]
    b = f.read(n)
    if len(b) < n:
        raise EOFError
    return pickle.loads(b)


class ApplyPool:
    def __init__(self, n: int, subs, maxv: float, pf: str, vs, rng, log_dir: Optional[str] = None):
        from . import engine as RE
        self.n = max(1, int(n))
        self.free: "queue.Queue[int]" = queue.Queue()
        self.stats = dict(done=0, local=0, send_s=0.0, wait_s=0.0)
        self.slock = threading.Lock()
        self.err = []
        d = tempfile.mkdtemp(prefix="contour_apply_")
        spec = dict(sys_path=[p for p in sys.path if p], setup=os.environ.get("CONTOUR_RAW_SETUP", ""), engine=_engine_settings(RE),
                    modules=_module_settings(["contour.retouch.bodytone", "contour.retouch.detect", "contour.retouch.skin",
                                              "contour.retouch.bodyskin", "contour.retouch.bodyspots", "contour.retouch.logwrap"]),
                    settings=_settings_state(), subs=subs, maxv=maxv, pf=pf, vs=vs, rng=rng)
        self.spec_path = os.path.join(d, "spec.pkl")
        with open(self.spec_path, "wb") as fh:
            pickle.dump(spec, fh, protocol=pickle.HIGHEST_PROTOCOL)
        env = dict(os.environ)
        env["CONTOUR_RAW_PROCS"] = "0"
        # each helper's own threads (OpenCV, the BLAS of numpy) kept few: a helper per core group, not every helper on every core
        hthr = str(max(1, int(os.environ.get("CONTOUR_HELPER_THREADS", "3") or 3)))
        for k_ in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CONTOUR_HELPER_THREADS"):
            env[k_] = hthr
        env["CONTOUR_APPLY_PROCS"] = "0"
        self.procs = []
        self.logs = []
        for k in range(self.n):
            lp = os.path.join(log_dir or d, f"applyhelper_{k}.log")
            self.logs.append(lp)
            lf = open(lp, "w", encoding="utf-8")
            flags = 0x08000000 if os.name == "nt" else 0
            p = subprocess.Popen([sys.executable, "-u", os.path.join(os.path.dirname(os.path.abspath(__file__)), "applyhelper.py"), self.spec_path],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=lf, env=env, creationflags=flags)
            self.procs.append(p)
            self.free.put(k)

    def apply(self, t: int, planes, rf):
        """planes after the retouch of rf, done by a helper; None when no helper could (the caller does it)"""
        try:
            k = self.free.get(timeout=1800)
        except queue.Empty:
            return None
        p = self.procs[k]
        ok = False
        try:
            if p.poll() is not None:
                return None
            t0 = time.time()
            b = pickle.dumps(("apply", int(t), list(planes), rf), protocol=pickle.HIGHEST_PROTOCOL)
            _send_bytes(p.stdin, b)
            t1 = time.time()
            msg = _read_msg(p.stdout)
            t2 = time.time()
            with self.slock:
                self.stats["send_s"] += t1 - t0
                self.stats["wait_s"] += t2 - t1
            if msg[0] == "out" and msg[1] == int(t):
                ok = True
                with self.slock:
                    self.stats["done"] += 1
                return msg[2]
            with self.slock:
                self.err.append(f"f{t}: {str(msg[-1])[-300:]}")
            return None
        except Exception as e:
            with self.slock:
                self.err.append(f"f{t} helper {k}: {e!r}")
            try:
                p.kill()
            except Exception:
                pass
            return None
        finally:
            if not ok:
                with self.slock:
                    self.stats["local"] += 1
            if p.poll() is None:
                self.free.put(k)
            elif all(q.poll() is not None for q in self.procs):
                # every helper is gone: the waiting threads do their frames themselves
                for _ in range(self.n):
                    self.free.put(k)

    def close(self) -> dict:
        for p in self.procs:
            try:
                _send_bytes(p.stdin, pickle.dumps(("quit",)))
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


def start_for_render(subs, maxv, pf, disp, color_range) -> Optional[ApplyPool]:
    """A pool when CONTOUR_APPLY_PROCS asks for one. disp: the render's (ColourManager, ViewSettings, range) or None; color_range: apply_planes' rng."""
    try:
        n = int(os.environ.get("CONTOUR_APPLY_PROCS", "0") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        return None
    try:
        vs = disp[1] if disp is not None else None
        drng = disp[2] if (disp is not None and len(disp) > 2) else None
        return ApplyPool(n, subs, maxv, pf, vs, (drng, color_range), log_dir=os.environ.get("CONTOUR_RAW_LOGDIR") or None)
    except Exception:
        import traceback
        traceback.print_exc()
        return None
