"""Training data from the footage itself, gathered while a cut is analysed (no labels, originals only).

For every analysed cut a spread of frames is taken and every tracked face on them is cut out level at the face
parser's framing (512 px), with the landmarks in crop coordinates. The files go to <install>\\Cloud\\data\\
<project>\\<cut id>\\ and stay on this computer: nothing is sent anywhere. Cloud\\step_train.bat (run by the
developer, not by the app) trains the face parser on one project at a time from this folder.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

FRAMES = 14          # frames sampled per cut
PER_FRAME = 8        # faces per frame
PER_CUT = 60
MIN_FACE = 40.0      # px at the proxy


def cloud_dir() -> Optional[Path]:
    """<install>\\Cloud (the installer makes it writable); DIAP_CLOUD overrides; the data folder as a fallback."""
    cands = []
    if os.environ.get("DIAP_CLOUD"):
        cands.append(Path(os.environ["DIAP_CLOUD"]))
    cands.append(Path(__file__).resolve().parents[3] / "Cloud")
    try:
        from ..core.settings import settings
        cands.append(Path(settings().data_path()) / "Cloud")
    except Exception:
        pass
    for d in cands:
        try:
            (d / "data").mkdir(parents=True, exist_ok=True)
            probe = d / "data" / ".w"
            probe.write_text("x")
            probe.unlink()
            return d
        except OSError:
            continue
    return None


def _safe(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", s).strip("_")[:80] or "project"


def collect(tracker) -> int:
    """Called at the end of an analysis. Returns the number of crops written (0 when switched off or nothing to take)."""
    try:
        from ..core.settings import settings
        if not getattr(settings(), "collect_training", True):
            return 0
    except Exception:
        pass
    st = tracker.store
    px = tracker.proxy
    if st is None or px is None or not getattr(st, "has_face", False) or not len(st.pids):
        return 0
    root = cloud_dir()
    if root is None:
        return 0
    from . import faceparse as FP
    prj, cut = tracker.project, tracker.cut
    out = root / "data" / _safe(prj.name) / cut.id
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)          # analysed again: this cut's crops are replaced
    out.mkdir(parents=True, exist_ok=True)
    N = st.n
    meta = []
    for t in sorted({int(x) for x in np.linspace(0, N - 1, min(N, FRAMES))}):
        ps = [p for p in range(len(st.pids)) if st.present[p, t] and np.isfinite(st.face[p, t, :, :2]).all()]
        if not ps or len(meta) >= PER_CUT:
            continue
        rgb = np.asarray(px.frame(t))
        for p in ps[:PER_FRAME]:
            lm = st.face[p, t].astype(np.float64).copy()
            lm[:, 0] *= px.sx
            lm[:, 1] *= px.sy
            wf = float(np.hypot(*(lm[16, :2] - lm[0, :2])))
            if wf < MIN_FACE:
                continue
            Mx = FP.FaceParser.crop_affine(lm)
            if Mx is None:
                continue
            crop = cv2.warpAffine(rgb, Mx, (FP.SIZE, FP.SIZE), flags=cv2.INTER_AREA if Mx[0, 0] < 1 else cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            name = f"{t:05d}_p{int(st.pids[p])}.jpg"
            cv2.imwrite(str(out / name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            lc = (np.c_[lm[:, :2], np.ones(len(lm))] @ Mx.T)
            meta.append(dict(f=name, t=int(t), pid=int(st.pids[p]), wf=round(wf, 1), vis=round(float(st.face_vis[p, t]), 2),
                             lum=round(float(crop[96:416, 96:416].mean() / 255.0), 3),
                             lm=[[round(float(a), 1), round(float(b), 1)] for a, b in lc]))
    if not meta:
        shutil.rmtree(out, ignore_errors=True)
        return 0
    (out / "meta.json").write_text(json.dumps(dict(project=prj.name, cut=cut.name, cut_id=cut.id, time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                                   frames=int(N), items=meta), ensure_ascii=False), encoding="utf-8")
    return len(meta)
