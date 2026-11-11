"""Group-dance tracking by segmentation: SAM 2 video memory follows each person through the shot.

The frame-to-frame matcher in assoc.py compares boxes, skeletons and looks from one frame to the next; where
dancers in the same outfit cross, all of that is alike and it guesses. SAM 2's video tracker (Meta, Apache-2.0)
instead keeps a memory of each person's own pixels (the frame where they were first seen, the last 6 frames and
16 object pointers) and finds that person again in every new frame by attention over the whole picture. It is
the kind of tracking-by-segmentation that has set the best identity scores on group-dance benchmarks.

How it is used here:
- every person the detector finds and no SAM 2 mask yet explains becomes a new SAM 2 object, prompted with the
  box and the torso joints on that frame;
- every frame, each object's memory finds its mask; detections are handed to the objects whose masks they fall
  in (share of the mask inside the box, joints inside the mask);
- where two objects end up on one person, the less sure one counts as hidden on that frame and does not learn
  from it, so one person's memory stays one person's;
- an object hidden or absent for 5 s ends.

The tracks then go through the usual joining of fragments and gap filling. Needs a GPU to be practical
(SAM 2.1 tiny at 1024 px: about 20 ms per person and frame on a recent NVIDIA card); on CPU the 512 px build is
used and costs about half a second per person and frame.
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

GRAPHS = ("vision_encoder", "mask_decoder", "memory_encoder", "memory_attention", "pointer_tpos")
LOST_MAX_S = 5.0          # an object not seen this long ends
NEW_SCORE = 0.5           # detector score for starting a new object
CLAIM = 0.5               # share of the smaller mask two objects must share to be on one person
HAND = 0.4                # mask / joints agreement to hand a detection to an object


def _iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class SAM2Video:
    """The five graphs of a SAM 2.1 video export (vision encoder, mask decoder, memory encoder, memory attention,
    object pointer positions) and their constants."""

    def __init__(self, folder: Path, make_session=None):
        if make_session is None:
            from .models import session as make_session
        folder = Path(folder)
        self.s = {k: make_session(folder / f"{k}.onnx") for k in GRAPHS}
        c = json.loads((folder / "constants.json").read_text(encoding="utf-8"))
        self.size = int(c["image_size"])
        self.tpos = np.array(c["memory_temporal_positional_encoding"], np.float32)
        self.mean = np.array(c["image_mean"], np.float32)
        self.std = np.array(c["image_std"], np.float32)
        self.F = self.size // 16
        self.n_mem = 7
        R = self.s["memory_attention"].get_inputs()[2].shape[0]
        self.n_ptr = (int(R) - self.n_mem * self.F * self.F) // 4 if isinstance(R, int) else 16

    def encode(self, rgb: np.ndarray) -> dict:
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        f0, f1, f2, f2n, pe = self.s["vision_encoder"].run(None, {"pixel_values": np.ascontiguousarray(x)})
        F2 = self.F * self.F
        return dict(f0=f0, f1=f1, f2=f2, f2n=f2n, pe=pe,
                    cur=np.ascontiguousarray(f2.reshape(256, F2).T[:, None, :]),
                    cpe=np.ascontiguousarray(pe.reshape(256, F2).T[:, None, :]))

    def decode(self, enc, feats2, pts, labels):
        o = self.s["mask_decoder"].run(None, {"feats0": enc["f0"], "feats1": enc["f1"], "feats2_cond": feats2,
                                               "input_points": np.asarray(pts, np.float32).reshape(1, 1, -1, 2),
                                               "input_labels": np.asarray(labels, np.int32).reshape(1, 1, -1)})
        _, high, iou, score, ptr = o
        return high, float(np.ravel(iou)[0]), float(np.ravel(score)[0]), np.ravel(ptr).astype(np.float32)

    def memory(self, enc, high, score: float, prompted: bool):
        return self.s["memory_encoder"].run(None, {"feats2": enc["f2"], "high_res_mask": high,
                                                   "object_score_logits": np.array([[score]], np.float32),
                                                   "binarize": np.array(1.0 if prompted else 0.0, np.float32)})

    def ptr_pos(self, diffs):
        return self.s["pointer_tpos"].run(None, {"normalized_diffs": np.asarray(diffs, np.float32)})[0]

    def attend(self, enc, mem, mpos):
        return self.s["memory_attention"].run(None, {"current_vision_features": enc["cur"],
                                                     "current_vision_position_embeddings": enc["cpe"],
                                                     "memory": mem, "memory_pos": mpos})[0]


class _Obj:
    def __init__(self, oid: int, t: int):
        self.id = oid
        self.t0 = t
        self.cond = None                   # (tokens, pos, frame) of the prompted frame
        self.cond_ptr = None               # (pointer, frame)
        self.recent = deque(maxlen=6)      # (tokens, pos, frame)
        self.ptrs = deque(maxlen=16)       # (pointer, frame)
        self.lost = 0
        self.frames: Dict[int, int] = {}
        self.free: Dict[int, np.ndarray] = {}   # frames followed without a detection -> mask box


class VideoTracker:
    def __init__(self, model: SAM2Video, fps: float = 30.0):
        self.m = model
        self.objs: List[_Obj] = []
        self.done: List[_Obj] = []
        self.nid = 0
        self.lost_max = int(LOST_MAX_S * max(1.0, fps))

    def _bank(self, o: _Obj, t: int):
        m = self.m
        blocks = [(o.cond[0], o.cond[1] + m.tpos[6])]
        for tok, pos, f in sorted(o.recent, key=lambda r: -r[2])[:6]:
            k = max(1, min(6, t - f))
            blocks.append((tok, pos + m.tpos[k - 1]))
        while len(blocks) < m.n_mem:
            blocks.append(blocks[-1])
        ptrs = [o.cond_ptr] + sorted(o.ptrs, key=lambda r: -r[1])[:m.n_ptr - 1]
        while len(ptrs) < m.n_ptr:
            ptrs.append(ptrs[-1])
        diffs = np.array([(t - f) / max(1, m.n_ptr - 1) for _, f in ptrs], np.float32)
        pp = m.ptr_pos(diffs)
        P = np.stack([p for p, _ in ptrs])
        mem = np.concatenate([b[0] for b in blocks] + [P.reshape(-1, 1, 64)], 0)
        mpos = np.concatenate([b[1] for b in blocks] + [np.repeat(pp, 4, axis=0)[:, None, :]], 0)
        return mem.astype(np.float32), mpos.astype(np.float32)

    def step(self, t: int, rgb: np.ndarray, dets: List[dict], scale: float = 1.0) -> Dict[int, np.ndarray]:
        """One frame. rgb: the analysed frame; dets: its detections (boxes / joints in rgb px times `scale`).
        Returns the object masks of this frame (object id -> bool mask at model resolution)."""
        m = self.m
        H, W = rgb.shape[:2]
        sx, sy = m.size / (W * scale), m.size / (H * scale)
        enc = m.encode(rgb)
        outs = []
        for o in self.objs:
            mem, mpos = self._bank(o, t)
            cf = m.attend(enc, mem, mpos)
            high, iou, score, ptr = m.decode(enc, cf, [[0.0, 0.0]], [-1])
            outs.append((o, high, iou, score, ptr))
        ms = [(high[0, 0] > 0) if score > 0 else np.zeros(high.shape[2:], bool) for _, high, _, score, _ in outs]
        hidden = set()
        for i in range(len(outs)):
            for j in range(i + 1, len(outs)):
                a, b = ms[i], ms[j]
                if not a.any() or not b.any():
                    continue
                if (a & b).sum() / max(1, min(a.sum(), b.sum())) > CLAIM:
                    si = outs[i][3] + 2 * outs[i][2]
                    sj = outs[j][3] + 2 * outs[j][2]
                    hidden.add(j if si >= sj else i)
        mboxes = []
        for mk in ms:
            ys, xs = np.nonzero(mk) if mk.any() else (np.zeros(0, int), np.zeros(0, int))
            mboxes.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None)
        rects = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            X1, Y1 = int(np.clip(x1 * sx, 0, m.size - 1)), int(np.clip(y1 * sy, 0, m.size - 1))
            X2, Y2 = int(np.clip(x2 * sx, X1 + 1, m.size)), int(np.clip(y2 * sy, Y1 + 1, m.size))
            rects.append((X1, Y1, X2, Y2))
        S = np.zeros((len(outs), len(dets)))
        KIN = np.zeros((len(outs), len(dets)))
        BIOU = np.zeros((len(outs), len(dets)))
        for i in range(len(outs)):
            mk = ms[i]
            if i in hidden or not mk.any():
                continue
            tot = mk.sum()
            for j, d in enumerate(dets):
                X1, Y1, X2, Y2 = rects[j]
                inbox = mk[Y1:Y2, X1:X2].sum()
                if inbox == 0:
                    continue
                kp = d["kp"]
                ok = (kp[:, 2] > 0.3) & np.isfinite(kp[:, :2]).all(1)
                if ok.sum() >= 3:
                    px = np.clip((kp[ok, 0] * sx).astype(int), 0, m.size - 1)
                    py = np.clip((kp[ok, 1] * sy).astype(int), 0, m.size - 1)
                    kin = float(mk[py, px].mean())
                else:
                    kin = 0.5
                bi = _iou(mboxes[i], rects[j])
                KIN[i, j], BIOU[i, j] = kin, bi
                # the person's joints on the mask, the mask's extent matching the box, the mask inside the box
                S[i, j] = 0.3 * kin + 0.4 * bi + 0.3 * inbox / tot
        pairs = []
        if S.size and linear_sum_assignment is not None:
            r, c = linear_sum_assignment(-S)
            pairs = [(i, j) for i, j in zip(r, c) if S[i, j] >= HAND]
        taken = {j for _, j in pairs}
        for i, j in pairs:
            outs[i][0].frames[t] = dets[j]["i"]
        got = {i for i, _ in pairs}
        for i, (o, high, iou, score, ptr) in enumerate(outs):
            # followed by SAM 2 but no detection here (motion blur, half hidden): where to look for the pose
            if i not in got and i not in hidden and score > 0 and mboxes[i] is not None:
                x1, y1, x2, y2 = mboxes[i]
                o.free[t] = np.array([x1 / sx, y1 / sy, x2 / sx, y2 / sy], np.float32)
        for i, (o, high, iou, score, ptr) in enumerate(outs):
            if i in hidden:
                o.lost += 1
                continue
            o.lost = 0 if score > 0 else o.lost + 1
            tok, pos = m.memory(enc, high, score, False)
            o.recent.append((tok, pos, t))
            o.ptrs.append((ptr, t))
        masks = {outs[i][0].id: ms[i] for i in range(len(outs)) if i not in hidden}
        for j, d in enumerate(dets):
            if j in taken or d.get("score", 1.0) < NEW_SCORE or d.get("recovered"):
                continue
            # someone already followed: their joints on an object's mask, or the box matching one
            if any(i not in hidden and (KIN[i, j] >= 0.5 or BIOU[i, j] >= 0.5) for i in range(len(outs))):
                continue
            x1, y1, x2, y2 = d["box"]
            pts, lab = [[x1 * sx, y1 * sy], [x2 * sx, y2 * sy]], [2, 3]
            kp = d["kp"]
            for k in (5, 6, 11, 12):
                if kp[k, 2] > 0.5 and np.isfinite(kp[k, :2]).all():
                    pts.append([kp[k, 0] * sx, kp[k, 1] * sy])
                    lab.append(1)
            high, iou, score, ptr = m.decode(enc, enc["f2n"], pts, lab)
            o = _Obj(self.nid, t)
            self.nid += 1
            tok, pos = m.memory(enc, high, max(score, 1.0), True)
            o.cond, o.cond_ptr = (tok, pos, t), (ptr, t)
            o.frames[t] = d["i"]
            self.objs.append(o)
            masks[o.id] = high[0, 0] > 0
        keep = []
        for o in self.objs:
            (self.done if o.lost > self.lost_max else keep).append(o)
        self.objs = keep
        return masks

    def tracks(self) -> List[Dict[int, int]]:
        return [o.frames for o in self.done + self.objs if len(o.frames) >= 3]

    def results(self):
        """(tracks, free boxes per track): frame -> detection index, and frame -> mask box where SAM 2 followed
        the person but no detection was handed over."""
        objs = [o for o in self.done + self.objs if len(o.frames) >= 3]
        return [o.frames for o in objs], [o.free for o in objs]


def recover_free(tracks, frees, dets, recover, keep: float = 0.6) -> int:
    """Poses where SAM 2 follows someone the detector missed (motion blur, half hidden), on the mask's box.
    Kept when the pose is about as sure as the person's usual and is not someone already found there."""
    from .assoc import _core_conf, oks
    by_t: Dict[int, list] = {}
    for k, fb in enumerate(frees):
        ref = [_core_conf(dets[f][i]["kp"]) for f, i in tracks[k].items()]
        ref = [c for c in ref if np.isfinite(c)]
        cref = float(np.median(ref)) if ref else float("nan")
        for t, box in fb.items():
            if t not in tracks[k] and np.isfinite(cref):
                by_t.setdefault(t, []).append((k, box, cref))
    added = 0
    for t in sorted(by_t):
        items = by_t[t]
        res = recover(t, np.array([b for _, b, _ in items], np.float32))
        if not res:
            continue
        for (k, box, cref), r in zip(items, res):
            if r is None:
                continue
            kp = r["kp"]
            c = _core_conf(kp)
            if not np.isfinite(c) or c < keep * cref:
                continue
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            if any(oks(kp, d["kp"], area) > 0.5 for d in dets[t]):
                continue
            d = dict(i=len(dets[t]), box=np.asarray(box, np.float32), kp=kp, score=0.0,
                     app=r.get("app", np.zeros(72, np.float32)), reid=None, face=None, recovered=True)
            if "raw" in r:
                d["raw"] = r["raw"]
            dets[t].append(d)
            tracks[k][t] = d["i"]
            added += 1
    return added


def make_tracker(mode: str, fps: float, progress=None) -> Optional[VideoTracker]:
    """The video tracker for this analysis, or None (switched off, or 'auto' without a GPU)."""
    from . import models as M
    if mode == "off":
        return None
    if mode == "auto" and M.device_label() == "CPU":
        return None
    path = M.ensure_model("track_sam2v", progress)
    return VideoTracker(SAM2Video(path), fps)
