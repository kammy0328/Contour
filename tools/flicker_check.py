"""Looks for retouch that flickers in a rendered cut: a change that is there on a frame and not on the frames beside it (a mask that
jumps for a frame, a spot filled on one frame only, a nose that blinks), before the cut goes to the client.

    python tools/flicker_check.py <original> <render> <out dir> [--width 960] [--thr 2.5] [--min-px 40]

Both films are read at the same small size (luma). The retouch of each frame is the render less the original (D_t). The neighbours'
retouch is carried onto the frame along the picture's own motion (DIS optical flow of the original), so a face that moves does not
count. A pixel flickers where the frame's retouch is far from what its two neighbours agree on: |D_t - (D_t-1 + D_t+1) / 2| over
--thr (8-bit levels) while |D_t-1 - D_t+1| is under half of that (the neighbours agree; a real change through time, a fade or a turn,
moves both). Blobs of at least --min-px such pixels are reported: frame, place, size, strength.

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


def check(src: str, out: str, out_dir: str, width: int = 960, thr: float = 2.5, min_px: int = 40, sheets: bool = True) -> dict:
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
    found = []
    u8 = [np.clip(s_, 0, 255).astype(np.uint8) for s_ in S]
    for t in range(1, T - 1):
        if not (np.abs(D[t]).max() > thr or np.abs(D[t - 1]).max() > thr or np.abs(D[t + 1]).max() > thr):
            continue
        fp = dis.calc(u8[t], u8[t - 1], None)          # t -> t-1
        fn = dis.calc(u8[t], u8[t + 1], None)          # t -> t+1
        Dp, Dn = warp(D[t - 1], fp), warp(D[t + 1], fn)
        mid = 0.5 * (Dp + Dn)
        spike = np.abs(D[t] - mid)
        agree = np.abs(Dp - Dn)
        m = ((spike > thr) & (agree < 0.5 * thr)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        blobs = []
        for i in range(1, n):
            if st[i, 4] < min_px:
                continue
            sel = lab == i
            blobs.append(dict(x=int(cen[i][0] * W / w), y=int(cen[i][1] * H / h), w=int(st[i, 2] * W / w), h=int(st[i, 3] * H / h),
                              px=int(st[i, 4]), strength=round(float(spike[sel].mean()), 2),
                              sign="on" if float((D[t] - mid)[sel].mean()) * float(np.sign(mid[sel].mean() or 1.0)) > 0 else "off"))
        if not blobs:
            continue
        blobs.sort(key=lambda r: -r["px"] * r["strength"])
        found.append(dict(frame=t, blobs=blobs[:8]))
        print(f"f{t}: {len(blobs)} place(s), largest {blobs[0]['px']} px at ({blobs[0]['x']}, {blobs[0]['y']}) strength {blobs[0]['strength']}", flush=True)
        if sheets:
            def vis(d):
                return cv2.applyColorMap(np.clip(128 + 8 * d, 0, 255).astype(np.uint8), cv2.COLORMAP_TWILIGHT_SHIFTED)
            base = cv2.cvtColor(u8[t], cv2.COLOR_GRAY2BGR)
            base[m > 0] = (0, 0, 255)
            row = np.hstack([base, vis(Dp), vis(D[t]), vis(Dn)])
            for k_, txt in enumerate(("t (flagged red)", "t-1 on t", "t", "t+1 on t")):
                cv2.putText(row, f"f{t} {txt}", (8 + k_ * w, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(str(od / f"flicker_f{t:05d}.jpg"), row, [cv2.IMWRITE_JPEG_QUALITY, 85])
    rep = dict(source=src, render=out, frames=T, size=[W, H], read_at=[w, h], thr=thr, min_px=min_px, flagged=found)
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
