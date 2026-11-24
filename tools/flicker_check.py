"""Looks for retouch that flickers in a rendered cut: a change that is there on a frame and not on the frames beside it (a mask that
jumps for a frame, a spot filled on one frame only, a nose that blinks), before the cut goes to the client.

    python tools/flicker_check.py <original> <render> <out dir> [--width 960] [--thr 2.5] [--min-px 40]

Both films are read at the same small size (luma). The retouch of each frame is the render less the original (D_t). The neighbours'
retouch is carried onto the frame along the picture's own motion (DIS optical flow of the original), so a face that moves does not
count. A pixel flickers where the frame's retouch is far from what its two neighbours agree on: |D_t - (D_t-1 + D_t+1) / 2| over
--thr (8-bit levels) while |D_t-1 - D_t+1| is under half of that (the neighbours agree; a real change through time, a fade or a turn,
moves both). Events of up to MAX_LEN frames (a mask there for two frames) the same way, between the frame before and the frame after.
--thr is the most the threshold is; it follows the cut's own retouch (REL of its 99th percentile, not under 0.8): a log original's retouch moves
few code values. Blobs of at least --min-px such pixels are reported: frame, length, place, size, strength.

Writes <out dir>/flicker.json and, for each flagged frame, <out dir>/flicker_f<frame>.jpg (the original | the retouch of t-1, t, t+1
x8, the flagged pixels in red). Reads only; prints a summary line per flagged frame."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _probe(path: str):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
                        "-of", "json", path], capture_output=True, text=True)
    s = json.loads(r.stdout or "{}").get("streams", [{}])[0]
    return int(s.get("width", 0)), int(s.get("height", 0)), int(s.get("nb_frames", 0) or 0)


def frames(path: str, w: int, h: int):
    """Luma frames (h, w) float32 0..255 of a film, read by ffmpeg at this size (code values as they are: a log original stays log)."""
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}:flags=area", "-pix_fmt", "gray16le", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE)
    n = w * h * 2
    try:
        while True:
            b = p.stdout.read(n)
            if len(b) < n:
                break
            yield np.frombuffer(b, np.uint16).reshape(h, w).astype(np.float32) / 257.0
    finally:
        p.stdout.close()
        p.wait()


def warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """img (of frame s) laid on frame t by the flow t -> s."""
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


MAX_LEN = 3      # events of up to this many frames (a mask that is there for two frames and gone)
REL = 0.25       # the threshold: this share of the cut's strongest retouch (99th percentile of |render - original|) ...


def check(src: str, out: str, out_dir: str, width: int = 960, thr: float = 2.5, min_px: int = 40, sheets: bool = True, thr_min: float = 0.8) -> dict:
    W, H, N = _probe(src)
    W2, H2, N2 = _probe(out)
    if (W, H) != (W2, H2):
        raise SystemExit(f"sizes differ: {W}x{H} / {W2}x{H2}")
    w = int(width)
    h = int(round(H * w / W / 2)) * 2
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    S, D = [], []
    for a, b in zip(frames(src, w, h), frames(out, w, h)):
        S.append(a)
        D.append(b - a)
    T = len(D)
    # the threshold follows how strong this cut's retouch is (a log original's retouch moves few code values): a share of its 99th percentile
    allv = np.concatenate([np.abs(d_[np.abs(d_) > 0.25]).ravel()[::7] for d_ in D] + [np.zeros(1, np.float32)])
    p99 = float(np.percentile(allv, 99)) if allv.size > 100 else 0.0
    thr_e = max(float(thr_min), min(float(thr), REL * p99)) if p99 > 0 else float(thr)
    found = []
    u8 = [np.clip(s_, 0, 255).astype(np.uint8) for s_ in S]
    flows: dict = {}

    def carried(t, s):
        """D_s laid on frame t along the picture's motion"""
        k = (t, s)
        if k not in flows:
            flows[k] = dis.calc(u8[t], u8[s], None)
            if len(flows) > 24:
                flows.pop(next(iter(flows)))
        return warp(D[s], flows[k])

    flagged_at = set()
    for t in range(1, T - 1):
        if t in flagged_at:
            continue
        for L in range(MAX_LEN, 0, -1):          # an event of L frames: t .. t+L-1, between t-1 and t+L (the longest first: a two-frame event
            b_ = t + L                           # tried as one frame has its second frame for a neighbour)
            if b_ >= T:
                continue
            if max(float(np.abs(D[t + i]).max()) for i in range(L)) <= thr_e and float(np.abs(D[t - 1]).max()) <= thr_e:
                continue
            Dp, Dn = carried(t, t - 1), carried(t, b_)
            mid = 0.5 * (Dp + Dn)
            agree = np.abs(Dp - Dn) < 0.5 * thr_e
            m = agree.copy()
            for i in range(L):
                Di = D[t] if i == 0 else carried(t, t + i)
                m &= np.abs(Di - mid) > thr_e
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
            blobs = []
            spike = np.abs(D[t] - mid)
            for i in range(1, n):
                if st[i, 4] < min_px:
                    continue
                sel = lab == i
                blobs.append(dict(x=int(cen[i][0] * W / w), y=int(cen[i][1] * H / h), w=int(st[i, 2] * W / w), h=int(st[i, 3] * H / h),
                                  px=int(st[i, 4]), strength=round(float(spike[sel].mean()), 2), frames=L))
            if not blobs:
                continue
            blobs.sort(key=lambda r: -r["px"] * r["strength"])
            found.append(dict(frame=t, frames=L, blobs=blobs[:8]))
            flagged_at.update(range(t, t + L))
            print(f"f{t}..f{t + L - 1}: {len(blobs)} place(s), largest {blobs[0]['px']} px at ({blobs[0]['x']}, {blobs[0]['y']}) strength {blobs[0]['strength']}", flush=True)
            if sheets:
                def vis(d):
                    return cv2.applyColorMap(np.clip(128 + 8 * d, 0, 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT_SHIFTED)
                base = cv2.cvtColor(u8[t], cv2.COLOR_GRAY2BGR)
                base[m > 0] = (0, 0, 255)
                row = np.hstack([base, vis(Dp), vis(D[t]), vis(Dn)])
                for k_, txt in enumerate(("t (flagged red)", "before, on t", "t", "after, on t")):
                    cv2.putText(row, f"f{t} {txt}", (8 + k_ * w, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(str(od / f"flicker_f{t:05d}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 85])
            break
    rep = dict(source=src, render=out, frames=T, size=[W, H], read_at=[w, h], thr=round(thr_e, 3), retouch_p99=round(p99, 3), min_px=min_px, flagged=found)
    (od / "flicker.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"{len(found)} of {T} frames flagged", flush=True)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("render")
    ap.add_argument("out_dir")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--thr", type=float, default=2.5)
    ap.add_argument("--min-px", type=int, default=40)
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args(argv)
    check(a.source, a.render, a.out_dir, a.width, a.thr, a.min_px, not a.no_sheets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
