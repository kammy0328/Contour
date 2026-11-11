"""COCO-17 skeleton definition shared by tracking, warping and the UI."""
from __future__ import annotations

import numpy as np

K = 17
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16
# joints further out along the limb (hiding a knee usually means the ankle is gone too)
DESCENDANTS = {5: [7, 9], 6: [8, 10], 7: [9], 8: [10], 11: [13, 15], 12: [14, 16], 13: [15], 14: [16]}

JOINT_NAMES_KO = ["코", "왼눈", "오른눈", "왼귀", "오른귀", "왼어깨", "오른어깨", "왼팔꿈치", "오른팔꿈치",
                  "왼손목", "오른손목", "왼골반", "오른골반", "왼무릎", "오른무릎", "왼발목", "오른발목"]

BONES = [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_SH, L_HIP), (R_SH, R_HIP),
         (L_HIP, R_HIP), (L_HIP, L_KN), (L_KN, L_AN), (R_HIP, R_KN), (R_KN, R_AN),
         (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR)]

LIMB_BONES = [(L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_HIP, L_KN), (L_KN, L_AN),
              (R_HIP, R_KN), (R_KN, R_AN), (L_SH, L_HIP), (R_SH, R_HIP)]

CORE_JOINTS = [L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]

# COCO OKS sigmas
OKS_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0

FLAG_LOW = 1
FLAG_DISAGREE = 2
FLAG_FB = 4
FLAG_BLUR = 8
FLAG_BONE = 16
FLAG_MISSING = 32
FLAG_USER = 64
FLAG_ADAPT = 128

FLAG_TEXT = {
    FLAG_LOW: "신뢰도 낮음",
    FLAG_DISAGREE: "모델 간 불일치",
    FLAG_FB: "양방향 추적 불일치",
    FLAG_BLUR: "모션블러",
    FLAG_BONE: "관절 길이 급변",
    FLAG_MISSING: "검출 없음",
}

# RTMW 133: face landmarks occupy 23..90 (iBUG 68 order)
RTMW_FACE = slice(23, 91)


def person_scale(kp: np.ndarray) -> float:
    """Robust person height estimate (px) from one frame of keypoints (17,>=2). NaN if unknown."""
    xy = kp[:, :2]
    ok = np.isfinite(xy).all(1)
    if ok.sum() < 3:
        return float("nan")
    cands = []
    def d(a, b):
        if ok[a] and ok[b]:
            return float(np.hypot(*(xy[a] - xy[b])))
        return None
    torso = [v for v in (d(L_SH, L_HIP), d(R_SH, R_HIP)) if v]
    if torso:
        cands.append(max(torso) * 3.4)
    legs = []
    for a, b, c in ((L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN)):
        u, v = d(a, b), d(b, c)
        if u and v:
            legs.append((u + v) * 2.0)
    if legs:
        cands.append(max(legs))
    sh = d(L_SH, R_SH)
    if sh:
        cands.append(sh * 4.2)
    y = xy[ok, 1]
    cands.append((y.max() - y.min()) * 1.1)
    return float(max(cands)) if cands else float("nan")


def bbox_from_kp(kp: np.ndarray, frame_wh, pad_x: float = 0.25, top: float = 0.18, bottom: float = 0.08):
    xy = kp[:, :2]
    ok = np.isfinite(xy).all(1)
    if ok.sum() < 3:
        return None
    x1, y1 = xy[ok].min(0)
    x2, y2 = xy[ok].max(0)
    h = max(y2 - y1, 1.0)
    w = max(x2 - x1, h * 0.25)
    cx = (x1 + x2) / 2
    w2 = w * (1 + pad_x * 2) / 2
    W, H = frame_wh
    return np.array([max(0, cx - w2), max(0, y1 - top * h), min(W, cx + w2), min(H, y2 + bottom * h)], np.float32)
