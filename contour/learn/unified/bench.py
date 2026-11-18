"""The comparison: the three models the app uses now and the combined model, on held-out projects only.

The teachers are the commercially usable ones (teach.Teachers). The reference is each teacher's answer on the
picture as it was. Rows:
- teacher, hard view : the teacher itself on the same picture made hard (dark, coloured light, soft, noisy, JPEG)
- combined, clean    : the combined model on the picture as it was
- combined, hard view: the combined model on the hard view
so a combined model that only copies its teacher shows on the clean row, and one that holds up better on hard
frames shows by beating the teacher's own hard row. Measures:
- face : face skin and body skin (neck) IoU, edges of the face skin and the hair (F1 within 1 px at 128 px),
         leak (combined skin or neck where the reference says something else, share of its skin),
         flicker (mean change of the skin map from t to t+1, landmark-aligned crops; the reference's own beside it)
- cloth: garment IoU (top, skirt, trousers, dress), skirt and dress IoU, garment-on-skin (garment where the
         reference says face, arm or leg: the low-contrast case)
- alpha: mean absolute difference, at the edges (0.02 < reference < 0.98, widened), gradient difference,
         flicker on whole frames (|change of alpha t->t+1 - the reference's change|)
- cost : seconds per call and the GPU memory the process holds, per model and execution provider
         (python -m contour.learn.unified.bench cost <model> <provider>, one process each)."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List

import cv2
import numpy as np

from . import teach as T
from .train import CLOTH_KEY, degrade, load

HAIR, BODY, FACE = (1,), (2,), (3,)           # MediaPipe multiclass
SKIN_CLOTH = (11, 12, 13, 14, 15)           # face, legs, arms in the clothes model


def _iou(a, b):
    u = (a | b).sum()
    return None if u < 20 else float((a & b).sum() / u)


def _edge(m):
    m = m.astype(np.uint8)
    return (cv2.dilate(m, np.ones((3, 3), np.uint8)) - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def _edge_f1(p, t, tol=1):
    ep, et = _edge(p), _edge(t)
    if et.sum() < 10:
        return None
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    prec = (ep & cv2.dilate(et.astype(np.uint8), k).astype(bool)).sum() / max(1, ep.sum())
    rec = (et & cv2.dilate(ep.astype(np.uint8), k).astype(bool)).sum() / max(1, et.sum())
    return float(2 * prec * rec / max(1e-6, prec + rec))


def _softmax(lo):
    lo = lo - lo.max(0, keepdims=True)
    e = np.exp(lo)
    return e / e.sum(0, keepdims=True)


class Student:
    def __init__(self, path: str, provider: str = "CPUExecutionProvider"):
        import onnxruntime as ort
        self.s = ort.InferenceSession(path, providers=[provider])

    def __call__(self, img):
        f, c, a = self.s.run(None, {"image": T.to_input(img)})
        return _softmax(f[0]), _softmax(c[0]), a[0, 0]


def face_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Face skin, body skin (the neck under the face), hair: IoU and edges (F1 within 1 px at 128 px)."""
    ps, rs = pred[list(FACE)].sum(0) > 0.5, ref[list(FACE)].sum(0) > 0.5
    pn, rn = pred[list(BODY)].sum(0) > 0.5, ref[list(BODY)].sum(0) > 0.5
    pa, ra = pred.argmax(0), ref.argmax(0)
    out = dict(skin=_iou(ps, rs), neck=_iou(pn, rn),
               skin_edge=_edge_f1(np.isin(pa, FACE), np.isin(ra, FACE)),
               hair_edge=_edge_f1(np.isin(pa, HAIR), np.isin(ra, HAIR)))
    pk = ps | pn
    out["leak"] = float((pk & ~(rs | rn)).sum() / max(1, pk.sum())) if pk.sum() > 50 else None
    return out


def cloth_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    pa, ra = pred.argmax(0), ref.argmax(0)
    pg, rg = np.isin(pa, CLOTH_KEY), np.isin(ra, CLOTH_KEY)
    rsk = np.isin(ra, SKIN_CLOTH)
    return dict(garment=_iou(pg, rg), skirt=_iou(pa == 5, ra == 5), dress=_iou(pa == 7, ra == 7),
                garment_on_skin=float((pg & rsk).sum() / max(1, rsk.sum())) if rsk.sum() > 50 else None)


def alpha_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    d = np.abs(pred - ref)
    edge = (ref > 0.02) & (ref < 0.98)
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    g = lambda z: np.hypot(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3))
    return dict(mad=float(d.mean()), edge_mad=float(d[edge].mean()) if edge.sum() > 50 else None,
                grad=float(np.abs(g(pred.astype(np.float32)) - g(ref.astype(np.float32))).mean()))


def _mean(rows: List[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(float(np.mean(v)), 4) if v else None
    out["n"] = len(rows)
    return out


def compare(items: list, student: Student, teachers=None, n: int = 300, seed: int = 5) -> dict:
    """items: held-out samples (teach.index rows). teachers: teach.Teachers for the teacher-on-hard-view rows
    (None: skipped). Returns {feature: {row: means}} plus flicker."""
    rng = random.Random(seed)
    by = {k: [it for it in items if it[1] == k] for k in ("face", "person", "frame")}
    res = {}
    rows: Dict[str, Dict[str, list]] = {k: dict(student_clean=[], student_hard=[], teacher_hard=[]) for k in ("face", "cloth", "alpha")}
    for kind, its in by.items():
        for it in rng.sample(its, min(n, len(its))):
            s = load(it[0])
            img = s["img"]
            hard = degrade(img, random.Random(hash(it[0]) & 0xFFFF))
            fs, cs, as_ = student(img)
            fh, ch, ah = student(hard)
            if kind == "face":
                ref = s["face"].astype(np.float32) / 255.0
                rows["face"]["student_clean"].append(face_metrics(fs, ref))
                rows["face"]["student_hard"].append(face_metrics(fh, ref))
                if teachers is not None:
                    rows["face"]["teacher_hard"].append(face_metrics(teachers.face_probs(hard), ref))
            if kind == "person":
                ref = s["cloth"].astype(np.float32) / 255.0
                rows["cloth"]["student_clean"].append(cloth_metrics(cs, ref))
                rows["cloth"]["student_hard"].append(cloth_metrics(ch, ref))
                if teachers is not None:
                    rows["cloth"]["teacher_hard"].append(cloth_metrics(teachers.cloth_probs(hard), ref))
            if "alpha" in s:
                ref = s["alpha"].astype(np.float32) / 255.0
                rows["alpha"]["student_clean"].append(alpha_metrics(as_, ref))
                rows["alpha"]["student_hard"].append(alpha_metrics(ah, ref))
                if teachers is not None and kind == "frame":
                    # RVM on the hard view of a single frame (no memory: what one frame alone gives)
                    teachers.matter.reset()
                    ta = teachers.matter(hard)
                    rows["alpha"]["teacher_hard"].append(alpha_metrics(cv2.resize(ta, ref.shape[::-1]), ref))
    for f, rr in rows.items():
        res[f] = {k: _mean(v) for k, v in rr.items() if v}
    res["flicker"] = flicker(items, student)
    return res


def flicker(items: list, student: Student, n: int = 150) -> dict:
    """Pairs (t, t+1): face skin map change (combined and reference), alpha change difference on frames."""
    idx = {(it[2], it[3], it[1], it[6], it[4]): it for it in items}
    fs, fr, al = [], [], []
    for it in items:
        if it[5] is None or it[1] not in ("face", "frame"):
            continue
        prev = idx.get((it[2], it[3], it[1], it[6], it[5]))
        if prev is None:
            continue
        a, b = load(prev[0]), load(it[0])
        if it[1] == "face":
            pa, pb = student(a["img"])[0], student(b["img"])[0]
            sa, sb = pa[list(FACE)].sum(0), pb[list(FACE)].sum(0)
            ra = a["face"][list(FACE)].astype(np.float32).sum(0) / 255.0
            rb = b["face"][list(FACE)].astype(np.float32).sum(0) / 255.0
            fs.append(float(np.abs(sb - sa).mean()))
            fr.append(float(np.abs(rb - ra).mean()))
        else:
            xa, xb = student(a["img"])[2], student(b["img"])[2]
            ra, rb = a["alpha"] / 255.0, b["alpha"] / 255.0
            al.append(float(np.abs((xb - xa) - (rb - ra)).mean()))
        if len(fs) + len(al) >= n:
            break
    m = lambda v: round(float(np.mean(v)), 5) if v else None
    return dict(face_skin_change_combined=m(fs), face_skin_change_reference=m(fr), face_pairs=len(fs),
                alpha_change_vs_reference=m(al), frame_pairs=len(al))


# ------------------------------------------------------------------ cost (one process per model)
def _gpu_mb() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            p, m = [x.strip() for x in line.split(",")[:2]]
            if int(p) == os.getpid():
                return float(m)
    except Exception:
        pass
    return 0.0


def cost(model: str, provider: str, reps: int = 20) -> dict:
    """Seconds per call and GPU memory for one model on one execution provider, as the app opens it."""
    import onnxruntime as ort
    from ...track import models as M
    if provider not in ort.get_available_providers():
        return dict(model=model, provider=provider, supported=False, reason="이 ONNX Runtime에 없음")
    shapes = {"mp_multiclass": [(1, 256, 256, 3)], "matte_rvm": [(1, 3, 288, 512)],
              "unified": [(1, 3, 512, 512), (1, 3, 288, 512)]}[model]
    if model == "unified":
        p = os.environ.get("DIAP_UNIFIED_FILE")
    elif model == "mp_multiclass":
        from ...track import faceparse as FP
        p = str(FP.MP_FILE)
    else:
        p = str(M.ensure_model(model))
    t0 = time.time()
    try:
        s = ort.InferenceSession(p, providers=[provider] + (["CPUExecutionProvider"] if provider != "CPUExecutionProvider" else []))
    except Exception as e:
        return dict(model=model, provider=provider, supported=False, reason=str(e)[:200])
    used = s.get_providers()[0]
    load_s = time.time() - t0
    names = [i.name for i in s.get_inputs()]
    times = []
    for sh in shapes:
        x = np.random.default_rng(0).standard_normal(sh).astype(np.float32)
        feed = {names[0]: x}
        if model == "mp_multiclass":
            feed = {names[0]: np.clip(x * 0.2 + 0.5, 0, 1).astype(np.float32)}
        if model == "matte_rvm":
            dt = np.float16 if "float16" in s.get_inputs()[0].type else np.float32
            feed = {names[0]: np.clip(x * 0.2 + 0.5, 0, 1).astype(dt)}
            for nm in names[1:5]:
                feed[nm] = np.zeros((1, 1, 1, 1), dt)
            if "downsample_ratio" in names:
                feed["downsample_ratio"] = np.array([1.0], np.float32)
        s.run(None, feed)
        t1 = time.time()
        for _ in range(reps):
            s.run(None, feed)
        times.append(round((time.time() - t1) / reps, 4))
    return dict(model=model, provider=used, asked=provider, supported=used == provider, load_s=round(load_s, 2),
                sec_per_call=times, gpu_mb=_gpu_mb())


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "cost":
        print("COST " + json.dumps(cost(sys.argv[2], sys.argv[3])), flush=True)
