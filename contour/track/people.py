"""People in one frame: detection, pose, doubles removed, colours, body re-ID and face.

Shared by the shot analysis (pass 1) and the dance-video batch tool, so both see people the same way."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .assoc import _oks, appearance

MIN_H = 0.07          # boxes shorter than this part of the frame are ignored
MAX_PEOPLE = 14


def dedupe(boxes: np.ndarray, scores: np.ndarray, kps: List[np.ndarray], skip=()) -> List[int]:
    """Indices to keep: two boxes that carry the same skeleton are one person found twice.
    `skip`: indices already known to be doubles (they neither stay nor push another out)."""
    n = len(boxes)
    skip = set(skip)
    if n < 2:
        return [i for i in range(n) if i not in skip]
    norm = []
    for k in kps:
        k = k.copy()
        m = np.nanmax(k[:, 2]) if np.isfinite(k[:, 2]).any() else 1.0
        k[:, 2] = k[:, 2] / max(m, 1e-6)
        norm.append(k)
    drop = set(skip)
    order = list(np.argsort(-scores))
    for ii, i in enumerate(order):
        if i in drop:
            continue
        b = boxes[i]
        area = (b[2] - b[0]) * (b[3] - b[1])
        for j in order[ii + 1:]:
            if j not in drop and _oks(norm[i], norm[j], area) > 0.75:
                drop.add(j)
    return [i for i in range(n) if i not in drop]


def _average(per_model, n):
    return [np.stack([np.column_stack([kp[i, :17], sc[i, :17]]) for kp, sc in per_model.values()]).mean(0)
            for i in range(n)]


def _extent(kp: np.ndarray):
    """(top, bottom, left, right) of the confidently placed joints, or None."""
    c = kp[:, 2]
    ok = np.isfinite(kp[:, :2]).all(1) & np.isfinite(c)
    if ok.sum() < 5:
        return None
    ok &= c >= 0.35 * np.nanmax(c[ok])
    if ok.sum() < 5:
        return None
    P = kp[ok, :2]
    return float(P[:, 1].min()), float(P[:, 1].max()), float(P[:, 0].min()), float(P[:, 0].max())


def _upright(kp: np.ndarray) -> bool:
    head = [kp[j, 1] for j in (0, 1, 2, 3, 4) if np.isfinite(kp[j, 1])]
    hips = [kp[j, 1] for j in (11, 12) if np.isfinite(kp[j, 1])]
    return bool(head and hips and np.mean(head) < np.mean(hips))


def _trim_below_feet(boxes: np.ndarray, kps: List[np.ndarray], H: int) -> Dict[int, np.ndarray]:
    out = {}
    for i, (b, kp) in enumerate(zip(boxes, kps)):
        e = _extent(kp)
        if e is None or not _upright(kp):
            continue
        top, bot, _, _ = e
        h = max(1.0, bot - top)
        if b[3] > bot + 0.20 * h:
            nb = b.copy()
            nb[3] = min(b[3], bot + 0.10 * h)
            out[i] = nb
    return out


def _reflections(boxes: np.ndarray, kps: List[np.ndarray]) -> set:
    """Detections that are another person's mirror image under their feet (glossy floor): about the same size,
    right below that person's feet, and seen less clearly."""
    ext = [_extent(k) for k in kps]
    conf = [float(np.nanmean(k[:, 2])) if np.isfinite(k[:, 2]).any() else 0.0 for k in kps]
    out = set()
    for j, ej in enumerate(ext):
        bj = boxes[j]
        hj = bj[3] - bj[1]
        for i, ei in enumerate(ext):
            if i == j or ei is None:
                continue
            bi = boxes[i]
            hi = max(1.0, ei[1] - ei[0])
            ov = min(bi[2], bj[2]) - max(bi[0], bj[0])
            if ov < 0.5 * (bj[2] - bj[0]):
                continue
            top_j = ej[0] if ej is not None else bj[1]
            below = top_j > ei[1] - 0.1 * hi
            similar = 0.5 * hi < (ej[1] - ej[0] if ej is not None else hj) < 1.6 * hi
            if below and similar and (ej is None or not _upright(kps[j]) or conf[j] < conf[i]):
                out.add(j)
                break
            # a box over someone's legs and their mirror image: no clear body of its own (far less sure of its
            # joints, or upside down)
            inter = max(0.0, ov) * max(0.0, min(bi[3], bj[3]) - max(bi[1], bj[1]))
            area_j = max(1.0, (bj[2] - bj[0]) * hj)
            if inter > 0.25 * area_j and bj[3] > bi[3] and _upright(kps[i]) and \
                    (conf[j] < 0.5 * conf[i] or (not _upright(kps[j]) and conf[j] < conf[i])):
                out.add(j)
                break
    return out


FACE_ANCHOR_W = 0.045     # faces at least this share of the frame width can anchor a person (close-ups)
FACE_ANCHOR_SCORE = 0.75


def _head_in(kp: np.ndarray, fb: np.ndarray, pad: float = 0.3) -> int:
    """How many of nose and eyes (placed with some confidence) fall inside the face box (padded)."""
    x, y, w, h = fb
    x0, y0, x1, y1 = x - pad * w, y - pad * h, x + w * (1 + pad), y + h * (1 + pad)
    n = 0
    cmax = np.nanmax(kp[:, 2]) if np.isfinite(kp[:, 2]).any() else 0.0
    for j in (0, 1, 2):
        q = kp[j]
        if np.isfinite(q[:2]).all() and np.isfinite(q[2]) and q[2] > 0.25 * cmax and x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
            n += 1
    return n


def _big_faces(rgb: np.ndarray, fdet) -> np.ndarray:
    W = rgb.shape[1]
    f = fdet.detect(rgb, None, target=960)
    if not len(f):
        return f
    return f[(f[:, 14] >= FACE_ANCHOR_SCORE) & (f[:, 2] >= FACE_ANCHOR_W * W)]


def _face_anchors(rgb: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """A head-and-shoulders box under each clearly visible (large) face.

    In close-ups (a face filling a third of the frame, the body cut off, hands up in front) the person
    detector draws boxes around arms, or one box over two people, and the pose model then scatters the joints,
    the face points with them. A face detector finds such faces reliably, and the pose model given a box
    framed on the face places the head, the face and the shoulders properly."""
    H, W = rgb.shape[:2]
    out = []
    for f in faces:
        x, y, w, h = [float(v) for v in f[:4]]
        cx = x + w / 2
        out.append([max(0.0, cx - 2.3 * w), max(0.0, y - 0.7 * h), min(float(W), cx + 2.3 * w),
                    min(float(H), y + 6.5 * h)])
    return np.array(out, np.float32).reshape(-1, 4)


def _box_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


def detect_people(rgb: np.ndarray, det, poses: Dict[str, object], reid=None, face_id=None, fdet=None,
                  face_always: bool = False) -> Tuple[List[dict], List[dict]]:
    """rgb: the analysis view of one frame. Returns (people, raw): people are dicts with box, kp (17, 3) averaged
    over the pose models, score, app (colours), reid (512) or None, face (dict) or None; raw holds each pose
    model's own output per person (all its keypoints) for the caller that keeps it."""
    H = rgb.shape[0]
    boxes, scores = det(rgb)
    if len(boxes):
        hh = boxes[:, 3] - boxes[:, 1]
        keep = np.nonzero(hh >= MIN_H * H)[0]
        keep = keep[np.argsort(-(hh[keep]))][:MAX_PEOPLE]
        boxes, scores = boxes[keep], scores[keep]
    per_model = {k: pe(rgb, boxes) for k, pe in poses.items()}
    avgs = _average(per_model, len(boxes))
    # boxes that reach far below the feet hold the person's mirror image on a glossy floor (or a shadow):
    # cut them at the feet and look again, the pose model then sees the person at a proper size
    trim = _trim_below_feet(boxes, avgs, H)
    if trim:
        boxes = boxes.copy()
        for i, nb in trim.items():
            boxes[i] = nb
        idx = list(trim)
        again = {k: pe(rgb, boxes[idx]) for k, pe in poses.items()}
        for k in per_model:
            kp, sc = per_model[k]
            kp, sc = kp.copy(), sc.copy()
            kp[idx], sc[idx] = again[k][0], again[k][1]
            per_model[k] = (kp, sc)
        avgs = _average(per_model, len(boxes))
    # close-ups: every clearly visible large face gets a person framed on it; detections that put their head
    # on that face too are the same person seen badly (a box around the arms, one box over two people)
    faces = np.zeros((0, 15), np.float32)
    owner: Dict[int, int] = {}
    drop = set()
    if fdet is not None:
        try:
            faces = _big_faces(rgb, fdet)
        except Exception:
            faces = np.zeros((0, 15), np.float32)
        extra = _face_anchors(rgb, faces) if len(faces) else np.zeros((0, 4), np.float32)
        if len(extra):
            more = {k: pe(rgb, extra) for k, pe in poses.items()}
            add = _average(more, len(extra))
            ok = [i for i in range(len(extra)) if _head_in(add[i], faces[i][:4]) >= 2]
            n0 = len(boxes)
            for i in ok:
                for j in range(n0):
                    if j in drop:
                        continue
                    b = boxes[j]
                    fx, fy, fw, fh = faces[i][:4]
                    holds = b[0] <= fx + fw / 2 <= b[2] and b[1] <= fy + fh / 2 <= b[3]
                    if _head_in(avgs[j], faces[i][:4]) >= 2 and (holds or _box_iou(b, extra[i]) > 0.2):
                        drop.add(j)
            if ok:
                boxes = np.vstack([boxes.reshape(-1, 4), extra[ok]]).astype(np.float32)
                scores = np.concatenate([scores.reshape(-1),
                                         np.array([max(0.55, 0.8 * float(faces[i][14])) for i in ok], np.float32)])
                for k in per_model:
                    kp, sc = per_model[k]
                    per_model[k] = (np.concatenate([kp, more[k][0][ok]]), np.concatenate([sc, more[k][1][ok]]))
                avgs = avgs + [add[i] for i in ok]
                for n_, i in enumerate(ok):
                    owner[n0 + n_] = i
    keep = [i for i in dedupe(boxes, scores, avgs, drop) if i not in _reflections(boxes, avgs)]
    feats = reid(rgb, boxes[keep]) if (reid is not None and keep) else None
    # each clearly visible face goes to the kept detection whose nose and eyes are on it (close-ups: the face
    # box is then a steady cue for following the person, whatever the body box does)
    fown = {i: faces[fi_][:4].astype(np.float32) for i, fi_ in owner.items() if i in keep}
    taken = {fi_ for i, fi_ in owner.items() if i in keep}
    for fi_, f in enumerate(faces):
        if fi_ in taken:
            continue
        cand = [(_head_in(avgs[i], f[:4]), -float((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])), i)
                for i in keep if i not in fown]
        cand = [c for c in cand if c[0] >= 2]
        if cand:
            fown[max(cand)[2]] = f[:4].astype(np.float32)
    people, raw = [], []
    for n_, i in enumerate(keep):
        avg = avgs[i]
        d = dict(box=boxes[i].copy(), kp=avg, score=float(scores[i]), app=appearance(rgb, avg, boxes[i]),
                 reid=feats[n_] if feats is not None else None, face=None, fbox=fown.get(i))
        # alone in the frame nobody needs telling apart, but linking the same person across cuts does:
        # then every few frames (face_always)
        if face_id is not None and (len(keep) > 1 or face_always):
            try:
                d["face"] = face_id(rgb, avg)
            except Exception:
                d["face"] = None
        people.append(d)
        raw.append({k: (kp[i].copy(), sc[i].copy()) for k, (kp, sc) in per_model.items()})
    return people, raw


def load_identity_models(enabled: bool = True):
    """(ReID, FaceID), either None when unavailable or switched off."""
    if not enabled:
        return None, None
    from .reid import FaceID, ReID
    try:
        reid = ReID()
    except Exception:
        reid = None
    try:
        face_id = FaceID()
    except Exception:
        face_id = None
    return reid, face_id
