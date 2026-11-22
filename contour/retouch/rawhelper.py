"""A helper process of retouch.rawpool: python rawhelper.py <spec.pkl> (run by its path: an embedded Python may ignore PYTHONPATH). Makes the raw frames (engine._prepare_raw) it is asked for,
over its stdin/stdout (length-prefixed pickles): ("make", [frames]) -> ("raw", frame, raw frame, bytes) or ("fail", frame, traceback) for each; ("quit",)."""
from __future__ import annotations

import importlib
import os
import pickle
import sys
import time
import traceback


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]      # (this file's folder is not a top-level package folder)
    spec = pickle.load(open(sys.argv[1], "rb"))
    for p in reversed(spec.get("sys_path") or []):
        if p not in sys.path:
            sys.path.insert(0, p)
    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    sys.stdout = sys.stderr                       # (prints of the app go to the log, not into the pipe)
    if spec.get("setup"):
        mn, _, fn = spec["setup"].partition(":")
        try:
            getattr(importlib.import_module(mn), fn or "setup")()
        except Exception:
            traceback.print_exc()
    import warnings
    warnings.simplefilter("ignore")
    import numpy as np
    from contour.core import settings as SE
    from contour.core import media as MD
    from contour.core.project import Project
    from contour.core.frames import frames_for
    from contour.track.analyze import CutTracker
    from contour.retouch import engine as RE
    from contour.retouch.rawpool import _send, _send_bytes, _recv
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
    try:
        MD.configure_ffmpeg()
    except Exception:
        traceback.print_exc()
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
    prj = Project.load(spec["project"])
    cut = spec["cut"]
    for i, c in enumerate(prj.cuts):
        if c.id == spec["cut_id"]:
            prj.cuts[i] = cut
    tr = CutTracker(prj, cut, frames_for(prj, cut))
    print(time.strftime("%H:%M:%S"), "raw helper ready, pid", os.getpid(), flush=True)
    _send(out, ("ready", os.getpid()))
    while True:
        try:
            msg = _recv(inp)
        except EOFError:
            break
        if msg[0] == "quit":
            break
        if msg[0] != "make":
            continue
        for s in msg[1]:
            t0 = time.time()
            try:
                rf = RE._prepare_raw(cut, tr, int(s), None)
                if rf is not None:
                    for j in rf.jobs:
                        j.masks = {k: v.astype(np.float16) for k, v in j.masks.items()}
                b = pickle.dumps(("raw", int(s), rf, 0), protocol=pickle.HIGHEST_PROTOCOL)
                _send_bytes(out, b)
                print(time.strftime("%H:%M:%S"), f"f{s} {time.time() - t0:.1f} s {len(b) / 1e6:.1f} MB", flush=True)
            except Exception:
                tb = traceback.format_exc()
                print(tb, flush=True)
                try:
                    _send(out, ("fail", int(s), tb))
                except Exception:
                    return


if __name__ == "__main__":
    main()
