"""Model inference wrappers: YOLOX detector, RTMPose/RTMW (SimCC), YuNet faces, RVM matting.

Pre/post-processing follows the RTMPose reference implementation (rtmlib, Apache-2.0).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import models as M


# --------------------------------------------------------------------------- helpers

def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> List[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thr)[0] + 1]
    return keep


def _warp_matrix(center, scale, output_size, rot: float = 0.0):
    """Affine from bbox (center, scale=[w,h]) to the model input. rot (radians): the crop is turned by this
    much about its centre (its x axis runs along that angle in the image), so a tilted face comes out upright."""
    src_w = scale[0]
    dst_w, dst_h = output_size
    c, s_ = float(np.cos(rot)), float(np.sin(rot))
    src_dir = np.array([s_ * src_w * 0.5, -c * src_w * 0.5])
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), np.float32)
    dst = np.zeros((3, 2), np.float32)
    src[0] = center
    src[1] = center + src_dir
    d = src[0] - src[1]
    src[2] = src[1] + np.array([-d[1], d[0]])
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    d = dst[0] - dst[1]
    dst[2] = dst[1] + np.array([-d[1], d[0]])
    return cv2.getAffineTransform(src, dst)


# --------------------------------------------------------------------------- detector

class Detector:
    """YOLOX person detector (expects BGR input, like its training pipeline)."""

    def __init__(self, key: str = "det_yolox_m", score_thr: float = 0.45, nms_thr: float = 0.45):
        self.key = key
        self.sess = M.session(M.ensure_model(key))
        self.inp = self.sess.get_inputs()[0].name
        self.size = M.MODELS[key]["input"]
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        hs = [self.size[0] // s for s in (8, 16, 32)]
        ws = [self.size[1] // s for s in (8, 16, 32)]
        grids, strides = [], []
        for h, w, s in zip(hs, ws, (8, 16, 32)):
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            g = np.stack((xv, yv), 2).reshape(-1, 2)
            grids.append(g)
            strides.append(np.full((g.shape[0], 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def __call__(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns boxes (N,4) xyxy in image px and scores (N,)."""
        img = rgb[..., ::-1]
        H, W = self.size
        ratio = min(H / img.shape[0], W / img.shape[1])
        rs = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)
        pad = np.full((H, W, 3), 114, np.uint8)
        pad[:rs.shape[0], :rs.shape[1]] = rs
        x = pad.transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0]
        if out.shape[-1] == 5:   # model with embedded NMS
            boxes = out[0, :, :4] / ratio
            scores = out[0, :, 4]
            m = scores > self.score_thr
            return boxes[m], scores[m]
        pred = out[0].copy()
        pred[:, :2] = (pred[:, :2] + self.grids) * self.strides
        pred[:, 2:4] = np.exp(pred[:, 2:4]) * self.strides
        scores = pred[:, 4] * pred[:, 5]      # objectness * person class
        m = scores > self.score_thr
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        b = pred[m, :4]
        s = scores[m]
        boxes = np.empty_like(b)
        boxes[:, 0] = b[:, 0] - b[:, 2] / 2
        boxes[:, 1] = b[:, 1] - b[:, 3] / 2
        boxes[:, 2] = b[:, 0] + b[:, 2] / 2
        boxes[:, 3] = b[:, 1] + b[:, 3] / 2
        boxes /= ratio
        keep = _nms(boxes, s, self.nms_thr)
        return boxes[keep].astype(np.float32), s[keep].astype(np.float32)


# --------------------------------------------------------------------------- pose

def _flip_pairs(k: int) -> np.ndarray:
    """Index permutation that swaps left/right keypoints (COCO body 17 or COCO-WholeBody 133)."""
    perm = np.arange(k)
    body = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
    pairs = list(body)
    if k == 133:
        pairs += [(17, 20), (18, 21), (19, 22)]
        f = 23
        face = [(i, 16 - i) for i in range(8)] + [(17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
                                                  (31, 35), (32, 34), (36, 45), (37, 44), (38, 43), (39, 42),
                                                  (40, 47), (41, 46), (48, 54), (49, 53), (50, 52), (55, 59),
                                                  (56, 58), (60, 64), (61, 63), (65, 67)]
        pairs += [(a + f, b + f) for a, b in face]
        pairs += [(91 + i, 112 + i) for i in range(21)]
    elif k != 17:
        return perm
    for a, b in pairs:
        perm[a], perm[b] = b, a
    return perm


class PoseEstimator:
    """Top-down SimCC pose model (RTMPose / RTMW / DWPose, also the face and hand models). Input RGB.

    flip: also run on the mirrored crop and average (test-time augmentation, ~2x slower, steadier)."""

    MEAN = np.array([123.675, 116.28, 103.53], np.float32)
    STD = np.array([58.395, 57.12, 57.375], np.float32)

    def __init__(self, key: str, flip: bool = False, padding: float = 1.25):
        self.key = key
        spec = M.MODELS[key]
        self.sess = M.session(M.ensure_model(key))
        self.inp = self.sess.get_inputs()[0].name
        self.size = spec["input"]        # (w, h)
        self.kpts = spec["kpts"]
        self.padding = padding
        shp = self.sess.get_inputs()[0].shape
        self.batchable = not isinstance(shp[0], int) or shp[0] != 1
        self.flip = flip and self.kpts in (17, 133)
        self.perm = _flip_pairs(self.kpts)

    def _prep(self, rgb: np.ndarray, box: np.ndarray, padding: float = 1.25, rot: float = 0.0):
        x1, y1, x2, y2 = box
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
        scale = np.array([(x2 - x1) * padding, (y2 - y1) * padding], np.float32)
        w, h = self.size
        ar = w / h
        if scale[0] > scale[1] * ar:
            scale = np.array([scale[0], scale[0] / ar], np.float32)
        else:
            scale = np.array([scale[1] * ar, scale[1]], np.float32)
        mat = _warp_matrix(center, scale, (w, h), rot)
        crop = cv2.warpAffine(rgb, mat, (int(w), int(h)), flags=cv2.INTER_LINEAR)
        x = ((crop.astype(np.float32) - self.MEAN) / self.STD).transpose(2, 0, 1)
        return x, center, scale, (cv2.invertAffineTransform(mat) if rot else None)

    def __call__(self, rgb: np.ndarray, boxes: np.ndarray, rots=None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns keypoints (N,K,2) in image px and scores (N,K). rots: optional angle per box (radians),
        see _warp_matrix."""
        n = len(boxes)
        if n == 0:
            return np.zeros((0, self.kpts, 2), np.float32), np.zeros((0, self.kpts), np.float32)
        if rots is None:
            rots = [0.0] * n
        preps = [self._prep(rgb, b, self.padding, float(r)) for b, r in zip(boxes, rots)]
        xs = np.stack([p[0] for p in preps]).astype(np.float32)
        locs, vals = self._run(xs)
        if self.flip:
            lf, vf = self._run(np.ascontiguousarray(xs[..., ::-1]))
            w_in = self.size[0]
            lf[..., 0] = (w_in - 1) - lf[..., 0]
            lf = lf[:, self.perm]
            vf = vf[:, self.perm]
            wa = np.maximum(vals, 1e-3)[..., None]
            wb = np.maximum(vf, 1e-3)[..., None]
            locs = (locs * wa + lf * wb) / (wa + wb)
            vals = 0.5 * (vals + vf)
        w, h = self.size
        kp = np.empty_like(locs)
        for i, (_, c, s, inv) in enumerate(preps):
            if inv is None:
                kp[i] = locs[i] / np.array([w, h], np.float32) * s + c - s / 2
            else:
                kp[i] = locs[i] @ inv[:, :2].T + inv[:, 2]
        return kp.astype(np.float32), vals.astype(np.float32)

    def _run(self, xs: np.ndarray):
        """Model input batch -> locations in input px (N,K,2) and scores (N,K)."""
        n = len(xs)
        if self.batchable:
            sx, sy = self.sess.run(None, {self.inp: xs})
        else:
            outs = [self.sess.run(None, {self.inp: xs[i:i + 1]}) for i in range(n)]
            sx = np.concatenate([o[0] for o in outs])
            sy = np.concatenate([o[1] for o in outs])
        N, K, Wx = sx.shape
        fx = sx.reshape(N * K, -1)
        fy = sy.reshape(N * K, -1)
        xl = np.argmax(fx, 1).astype(np.float32)
        yl = np.argmax(fy, 1).astype(np.float32)
        # sub-bin refinement (parabolic fit around the peak)
        xl = xl + _parabolic(fx, xl.astype(int))
        yl = yl + _parabolic(fy, yl.astype(int))
        vals = 0.5 * (fx.max(1) + fy.max(1))
        locs = np.stack([xl, yl], -1).reshape(N, K, 2) / 2.0   # simcc split ratio 2
        return locs.astype(np.float32), vals.reshape(N, K).astype(np.float32)


def _parabolic(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    n = arr.shape[1]
    i0 = np.clip(idx - 1, 0, n - 1)
    i2 = np.clip(idx + 1, 0, n - 1)
    r = np.arange(arr.shape[0])
    a, b, c = arr[r, i0], arr[r, idx], arr[r, i2]
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / np.where(np.abs(den) > 1e-9, den, 1), 0.0)
    off[(idx == 0) | (idx == n - 1)] = 0.0
    return np.clip(off, -0.5, 0.5).astype(np.float32)


# --------------------------------------------------------------------------- faces

class FaceDetector:
    """OpenCV YuNet on head crops. Returns faces as (x,y,w,h, 5 landmarks (10), score)."""

    def __init__(self, key: str = "face_yunet", score_thr: float = 0.6):
        self.path = str(M.ensure_model(key))
        self.det = cv2.FaceDetectorYN.create(self.path, "", (320, 320), score_thr, 0.3, 50)
        self._size = (320, 320)

    def detect(self, rgb: np.ndarray, box: Optional[np.ndarray] = None, target: int = 256) -> np.ndarray:
        """Detect faces inside `box` (xyxy) of the RGB image. Output (N,15) in image px."""
        H, W = rgb.shape[:2]
        if box is None:
            box = np.array([0, 0, W, H], np.float32)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return np.zeros((0, 15), np.float32)
        crop = rgb[y1:y2, x1:x2, ::-1]
        s = target / max(crop.shape[:2])
        if abs(s - 1) > 0.05:
            crop = cv2.resize(crop, (max(8, int(crop.shape[1] * s)), max(8, int(crop.shape[0] * s))),
                              interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
        else:
            s = 1.0
        size = (crop.shape[1], crop.shape[0])
        if size != self._size:
            self.det.setInputSize(size)
            self._size = size
        _, faces = self.det.detect(np.ascontiguousarray(crop))
        if faces is None or len(faces) == 0:
            return np.zeros((0, 15), np.float32)
        f = faces.astype(np.float32).copy()
        f[:, :14] /= s
        f[:, [0, 4, 6, 8, 10, 12]] += x1
        f[:, [1, 5, 7, 9, 11, 13]] += y1
        return f


# --------------------------------------------------------------------------- matting

class Matter:
    """Robust Video Matting (recurrent). Feed frames in order; call reset() at cuts."""

    def __init__(self, key: str = "matte_rvm"):
        self.sess = M.session(M.ensure_model(key))
        self.names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.dtype = np.float16 if "float16" in self.sess.get_inputs()[0].type else np.float32
        self.reset()

    def reset(self):
        z = np.zeros((1, 1, 1, 1), self.dtype)
        self.rec = [z, z, z, z]

    def __call__(self, rgb: np.ndarray, downsample: Optional[float] = None) -> np.ndarray:
        """rgb uint8 or float (H,W,3) -> alpha float32 (H,W) 0..1."""
        h, w = rgb.shape[:2]
        if downsample is None:
            downsample = min(1.0, 512.0 / max(h, w))
        src = rgb.astype(np.float32)
        if rgb.dtype == np.uint8:
            src /= 255.0
        src = src.transpose(2, 0, 1)[None].astype(self.dtype)
        feed = {"src": src, "r1i": self.rec[0], "r2i": self.rec[1], "r3i": self.rec[2], "r4i": self.rec[3],
                "downsample_ratio": np.array([downsample], np.float32)}
        feed = {k: v for k, v in feed.items() if k in self.names}
        outs = self.sess.run(None, feed)
        res = dict(zip(self.out_names, outs))
        self.rec = [res["r1o"], res["r2o"], res["r3o"], res["r4o"]]
        return res["pha"][0, 0].astype(np.float32)


# --------------------------------------------------------------------------- SAM 2 person segmentation

class Segmenter:
    """SAM 2.1 image predictor (ONNX): encode a picture once, then decode one mask per prompted person.

    Prompts come from tracking (person box, joints on the body as positive points, other people's joints as
    negative points), so every tracked person gets an exact silhouette of their own."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
    SIZE = 1024

    def __init__(self, key: str = "seg_sam2_s"):
        d = M.ensure_model(key)
        self.enc = M.session(d / "vision_encoder.onnx")
        self.dec = M.session(d / "prompt_encoder_mask_decoder.onnx")
        self.enc_in = self.enc.get_inputs()[0].name
        self.dec_names = [i.name for i in self.dec.get_inputs()]

    def encode(self, rgb: np.ndarray, region: Optional[Tuple[float, float, float, float]] = None) -> dict:
        """rgb uint8 (H,W,3). region (x0,y0,x1,y1) in rgb px encodes only that part (small people)."""
        H, W = rgb.shape[:2]
        if region is None:
            x0, y0, x1, y1 = 0.0, 0.0, float(W), float(H)
            img = rgb
        else:
            x0, y0, x1, y1 = [float(v) for v in region]
            img = rgb[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
            x1, y1 = x0 + img.shape[1], y0 + img.shape[0]
        interp = cv2.INTER_AREA if max(img.shape[:2]) > self.SIZE else cv2.INTER_LINEAR
        x = cv2.resize(img, (self.SIZE, self.SIZE), interpolation=interp).astype(np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None].astype(np.float32)
        emb = self.enc.run(None, {self.enc_in: x})
        return dict(emb=emb, rect=(x0, y0, x1, y1))

    def segment(self, enc: dict, box, pos: np.ndarray, neg: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """box (4,) and points (N,2) in rgb px. Returns (256x256 mask logits over enc['rect'], quality)
        or (None, 0) when SAM finds no object there."""
        x0, y0, x1, y1 = enc["rect"]
        kx, ky = self.SIZE / max(x1 - x0, 1e-6), self.SIZE / max(y1 - y0, 1e-6)

        def tr(pts):
            pts = np.asarray(pts, np.float32).reshape(-1, 2)
            return np.column_stack([(pts[:, 0] - x0) * kx, (pts[:, 1] - y0) * ky]).astype(np.float32)

        pos = tr(pos)
        neg = tr(neg)
        # negative points outside the encoded area (a close-up of a small person) say nothing useful
        neg = neg[(neg[:, 0] >= 0) & (neg[:, 0] < self.SIZE) & (neg[:, 1] >= 0) & (neg[:, 1] < self.SIZE)]
        pts = np.concatenate([pos, neg], 0)
        lab = np.concatenate([np.ones(len(pos), np.int64), np.zeros(len(neg), np.int64)])
        if len(pts) == 0:
            pts = np.zeros((1, 2), np.float32)
            lab = -np.ones(1, np.int64)                  # "not a point": box only
        bx = tr(np.asarray(box, np.float32).reshape(2, 2)).reshape(1, 1, 4)
        feed = {"input_points": pts[None, None], "input_labels": lab[None, None], "input_boxes": bx,
                "image_embeddings.0": enc["emb"][0], "image_embeddings.1": enc["emb"][1],
                "image_embeddings.2": enc["emb"][2]}
        feed = {k: v for k, v in feed.items() if k in self.dec_names}
        iou, masks, obj = self.dec.run(None, feed)
        if float(obj.reshape(-1)[0]) < 0:
            return None, 0.0
        masks = masks[0, 0]                               # (3, 256, 256)
        iou = iou.reshape(-1)
        # the candidate that best agrees with the prompts: high predicted quality, joints inside, others out
        best, bi = -1e9, 0
        for k in range(masks.shape[0]):
            m = masks[k]
            sc = float(iou[k])
            for P, want in ((pos, 1.0), (neg, -1.0)):
                if len(P):
                    gx = np.clip((P[:, 0] / self.SIZE * 256).astype(int), 0, 255)
                    gy = np.clip((P[:, 1] / self.SIZE * 256).astype(int), 0, 255)
                    inside = (m[gy, gx] > 0).mean()
                    sc += 0.35 * want * inside
            if sc > best:
                best, bi = sc, k
        return masks[bi].astype(np.float32), float(iou[bi])
