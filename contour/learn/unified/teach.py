"""Teacher labels for the combined model, from analysed originals only (no retouched result needed).

Every sample is the picture exactly as the app hands it to that teacher, and the teacher's answer:
- face   : the face crop of faceparse (raw picture, landmark-aligned, 512 px) and face_parse's class
           probabilities (19 x 128 x 128) -- the original SegFormer-B5, never a parser trained further;
- person : the person's box crop of clothing.run (exposure-evened picture, resized to 512 x 512) with
           parse_clothes' probabilities (18 x 128 x 128) and the RVM alpha cut out of the same box;
- frame  : the whole exposure-evened frame at 512 px wide with the RVM alpha. RVM's alpha is everybody at once:
           it teaches "a person is here, this soft", never who it is (the owner of a pixel stays the tracking's
           person labels, as now).
Frames are taken in pairs (t, t+1) so the check can measure flicker. RVM runs over the shot in order from its
start (its memory reset at the start and at every jump), as in the analysis.
Files: <out>/<project>/<cut id>/<kind>_<t>_p<pid>.npz (img: JPEG bytes, labels uint8, meta JSON). Nothing is written
into the project."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

STEP = 10            # frames between pairs
PAIRS = 6            # pairs per cut at most
PERSONS = 3          # people per frame at most
MIN_FACE = 60        # face width (proxy px) for a face sample
FRAME_W = 512        # whole-frame samples: this wide, height to a multiple of 32
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _softmax(lo: np.ndarray) -> np.ndarray:
    lo = lo - lo.max(0, keepdims=True)
    e = np.exp(lo)
    return e / e.sum(0, keepdims=True)


def _q(p: np.ndarray) -> np.ndarray:
    return np.clip(p * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _jpg(img: np.ndarray) -> np.ndarray:
    ok, b = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return np.frombuffer(b.tobytes(), np.uint8)


def unjpg(b: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.asarray(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def to_input(img_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> (1, 3, H, W) float32 normalised as every teacher here (ImageNet)."""
    x = (img_u8.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def frame_size(w: int, h: int) -> tuple:
    hh = max(32, int(round(FRAME_W * h / max(1, w) / 32.0)) * 32)
    return FRAME_W, hh


class Teachers:
    """The app's three models, as the app runs them (ONNX)."""

    def __init__(self):
        from ...track import models as M
        from ...track import faceparse as FP
        from ...track.clothing import Parser
        from ...track.infer import Matter
        self.face = M.session(M.ensure_model(FP.MODEL_KEY))          # the original parser, not one trained further
        self.face_in = self.face.get_inputs()[0].name
        self.cloth = Parser()
        self.matter = Matter()

    def face_probs(self, crop: np.ndarray) -> np.ndarray:
        return _softmax(self.face.run(None, {self.face_in: to_input(crop)})[0][0])

    def cloth_probs(self, crop512: np.ndarray) -> np.ndarray:
        lg = self.cloth.sess.run(None, {self.cloth.inp: to_input(crop512)})[0][0]
        return _softmax(lg)


def _write(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)


def run_cut(prj, cut, tracker, teachers: Teachers, out: Path, cancel: Optional[Callable[[], bool]] = None) -> dict:
    from ...track import faceparse as FP
    from ...track.clothing import _crop_box
    from ...track.exposure import analysis_view
    st, px = tracker.store, tracker.proxy
    a, b = cut.work_range
    starts = list(range(a, b - 1, STEP))[:PAIRS]
    want = sorted({t for s in starts for t in (s, s + 1) if t < b})
    n = dict(face=0, person=0, frame=0)
    if not want or st is None:
        return n
    mw, mh = tracker.matte.w, tracker.matte.h
    W, H = px.w, px.h
    fw, fh = frame_size(W, H)
    teachers.matter.reset()
    base = dict(project=prj.name, cut=cut.name, cut_id=cut.id, source=Path(str(cut.source)).name, created=time.time())
    for t in range(a, want[-1] + 1):
        if cancel and cancel():
            break
        if t > a and st.jump[t - 1]:
            teachers.matter.reset()
        rgb = np.ascontiguousarray(np.asarray(px.frame(t)))
        view = analysis_view(rgb)
        al = teachers.matter(cv2.resize(view, (mw, mh), interpolation=cv2.INTER_AREA))
        if t not in want:
            continue
        pair = t - 1 if (t - 1) in want and (t - a) % STEP == 1 else None
        meta = dict(base, t=int(t), pair_of=pair)
        # the whole frame
        fimg = cv2.resize(view, (fw, fh), interpolation=cv2.INTER_AREA)
        fal = cv2.resize(al, (fw, fh), interpolation=cv2.INTER_LINEAR)
        _write(out / prj.name / cut.id / f"frame_{t:05d}.npz", img=_jpg(fimg), alpha=_q(fal),
               meta=np.array(json.dumps(dict(meta, kind="frame"))))
        n["frame"] += 1
        al_px = cv2.resize(al, (W, H), interpolation=cv2.INTER_LINEAR)
        ps = [p for p in range(len(st.pids)) if st.present[p, t]][:PERSONS]
        for p in ps:
            pid = int(st.pids[p])
            box = _crop_box(st, p, t, px.sx, px.sy, W, H)
            if box is not None:
                x0, y0, x1, y1 = box
                c512 = cv2.resize(view[y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_LINEAR)
                cp = teachers.cloth_probs(c512)
                ca = cv2.resize(al_px[y0:y1, x0:x1], (512, 512), interpolation=cv2.INTER_LINEAR)
                _write(out / prj.name / cut.id / f"person_{t:05d}_p{pid}.npz", img=_jpg(c512), cloth=_q(cp),
                       alpha=_q(ca), box=np.array(box, np.int32),
                       meta=np.array(json.dumps(dict(meta, kind="person", pid=pid))))
                n["person"] += 1
            lm = st.face[p, t].astype(np.float64).copy() if getattr(st, "has_face", False) else None
            if lm is None or not np.isfinite(lm[:, :2]).all():
                continue
            lm[:, 0] *= px.sx
            lm[:, 1] *= px.sy
            if float(np.hypot(*(lm[16, :2] - lm[0, :2]))) < MIN_FACE:
                continue
            Mx = FP.FaceParser.crop_affine(lm)
            if Mx is None:
                continue
            crop = cv2.warpAffine(rgb, Mx, (FP.SIZE, FP.SIZE), flags=cv2.INTER_AREA if Mx[0, 0] < 1 else cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            fp = teachers.face_probs(crop)
            _write(out / prj.name / cut.id / f"face_{t:05d}_p{pid}.npz", img=_jpg(crop), face=_q(fp),
                   M=Mx.astype(np.float32), meta=np.array(json.dumps(dict(meta, kind="face", pid=pid))))
            n["face"] += 1
    return n


def run_project(project_path: str, out: str, cuts: int = 25, log=print, cancel=None, teachers=None) -> dict:
    """Labels for up to `cuts` analysed cuts of one project (read only)."""
    from ...core.frames import frames_for
    from ...core.project import Project
    from ...track.analyze import CutTracker
    prj = Project.load(project_path)
    teachers = teachers or Teachers()
    tot = dict(face=0, person=0, frame=0, cuts=0)
    lst = prj.timeline_cuts() if hasattr(prj, "timeline_cuts") else prj.cuts
    for c in lst:
        if tot["cuts"] >= cuts:
            break
        if not getattr(c, "analyzed", False):
            continue
        try:
            tr = CutTracker(prj, c, frames_for(prj, c))
        except Exception:
            continue
        if tr.store is None:
            continue
        tot["cuts"] += 1
        n = run_cut(prj, c, tr, teachers, Path(out), cancel)
        for k in n:
            tot[k] += n[k]
        log(f"{prj.name} / {c.name}: {n}")
    return tot


def index(root: str) -> list:
    """[(path, kind, project, cut_id, t, pair_of)] of every sample under root."""
    out = []
    for f in sorted(Path(root).rglob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        try:
            with np.load(f) as d:
                m = json.loads(str(d["meta"]))
        except Exception:
            continue
        out.append((str(f), m["kind"], m["project"], m["cut_id"], int(m["t"]), m.get("pair_of"), m.get("pid")))
    return out
