"""A helper process of retouch.applypool: python applyhelper.py <spec.pkl>. ("apply", t, planes, prepared frame) -> ("out", t, planes) or
("fail", t, traceback); ("quit",)."""
from __future__ import annotations

import importlib
import os
import pickle
import struct
import sys
import time
import traceback


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    spec = pickle.load(open(sys.argv[1], "rb"))
    want = [root] + list(spec.get("sys_path") or []) + list(sys.path)
    seen, path = set(), []
    for p in want:
        k = os.path.normcase(os.path.abspath(p or "."))
        if k in seen or k == os.path.normcase(here):
            continue
        seen.add(k)
        path.append(p)
    sys.path[:] = path
    for m in [m for m in sys.modules if m == "contour" or m.startswith("contour.")]:
        del sys.modules[m]
    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    sys.stdout = sys.stderr
    if spec.get("setup"):
        mn, _, fn = spec["setup"].partition(":")
        try:
            getattr(importlib.import_module(mn), fn or "setup")()
        except Exception:
            traceback.print_exc()
    import warnings
    warnings.simplefilter("ignore")
    try:
        import cv2
        cv2.setNumThreads(int(os.environ.get("CONTOUR_HELPER_THREADS", "3") or 3))
    except Exception:
        pass
    from contour.core import settings as SE
    from contour.core.colour import manager
    from contour.retouch import engine as RE
    try:
        SE.AppSettings.save = lambda self: None
    except Exception:
        pass
    s_ = SE.settings()
    for k, v in (spec.get("settings") or {}).items():
        try:
            setattr(s_, k, v)
        except Exception:
            pass
    for k, v in (spec.get("engine") or {}).items():
        setattr(RE, k, v)
    RE.TIMING = None
    for mn, d in (spec.get("modules") or {}).items():
        try:
            m = importlib.import_module(mn)
            for k, v in d.items():
                setattr(m, k, v)
        except Exception:
            traceback.print_exc()
    drng, rng = spec["rng"]
    disp = (manager(), spec["vs"], drng) if spec.get("vs") is not None else None
    subs, maxv, pf = spec["subs"], spec["maxv"], spec["pf"]
    print(time.strftime("%H:%M:%S"), "apply helper ready, pid", os.getpid(), flush=True)

    def send(obj):
        b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        out.write(struct.pack("<Q", len(b)))
        out.write(b)
        out.flush()

    while True:
        h = inp.read(8)
        if len(h) < 8:
            break
        n = struct.unpack("<Q", h)[0]
        b = inp.read(n)
        if len(b) < n:
            break
        msg = pickle.loads(b)
        if msg[0] == "quit":
            break
        if msg[0] != "apply":
            continue
        _, t, planes, rf = msg
        t0 = time.time()
        try:
            RE.masks_f32(rf)
            res = RE.apply_planes(planes, subs, maxv, pf, rf, 1.0, disp, rng)
            send(("out", t, [x for x in res]))
            print(time.strftime("%H:%M:%S"), f"f{t} {time.time() - t0:.1f} s", flush=True)
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            send(("fail", t, tb))


if __name__ == "__main__":
    main()
