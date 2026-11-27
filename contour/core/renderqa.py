"""The check of a rendered file, made right after the render (1.27.1, the user 11/27: "렌더 후 자동 검수를 앱에").

What the LNGSHOT BEEP BEEP delivery was checked by, by hand and by scripts, done by the app on every rendered cut:

1. the file: as many frames as the cut, the size and the frame rate of the cut (MD.probe);
2. the frames in their places: the first, the middle and the last frame of the file are nearer to the same frame of the source than to the one
   before or after it (a frame lost or doubled in the writing moves every frame after it);
3. every target's face, frame by frame: how much the retouch changed inside it (the mean change in the face's box). A frame where the change
   drops to a third of the frames round it while they have it is a frame where the retouch fell out (보정 풀림); a step up or down by more than
   half from one frame to the next is a stage coming or going;
4. what changes on one frame only: the retouch of a frame (file - source) against the median of its own and its neighbours' carried by the flow of
   the source (the same as retouch.temporal does before the writing): what is left after the render's own steadying, where it is larger than
   FLICK_LEVEL over FLICK_PX pixels, is a blink.

Each finding comes with a sheet (source | file | change x8) in the cut's folder (cuts/<id>/qa/<file name>/), and the report (report.json there). The
pictures are looked at small (QA_W wide), in luma."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from . import media as MD

QA_W = 960                    # the width the frames are looked at
FLICK_LEVEL = 2.0             # levels of 255: a change of one frame only, larger than this ...
FLICK_PX = 60                 # ... over at least this many pixels of the QA_W picture, is a blink
DROP_SHARE = 0.3              # a face's change below this share of its neighbours' ...
DROP_MIN = 0.12               # ... while they change it by more than this (levels of 255, the face's mean)
STEP_RATIO = 1.6              # from one frame to the next, more than this up or down ...
STEP_MIN = 0.15               # ... where the larger of the two is above this
MAX_SHEETS = 24


def _read_all(info, start: int, n: int, w: int, h: int, cancelled=None) -> List[np.ndarray]:
    rd = MD.FrameReader(info, int(start), int(n), "gray16le", size=(w, h), prefetch=4)
    out = []
    try:
        for _ in range(int(n)):
            if cancelled and cancelled():
                break
            pl = rd.read()
            if pl is None:
                break
            out.append(np.asarray(pl[0], np.float32) * (255.0 / 65535.0))
    finally:
        rd.close()
    return out


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from ..retouch import temporal as TM
    u = lambda x: np.clip(x + 0.5, 0, 255).astype(np.uint8)
    return TM._flow(u(a), u(b))


def one_frame_changes(src: List[np.ndarray], out: List[np.ndarray]) -> List[dict]:
    """[{t, px, level, x, y}] where a frame's retouch differs from the median of its own and its neighbours' (carried by the flow) by more than
    FLICK_LEVEL over FLICK_PX pixels (x, y: the place's centre, of the QA picture; level: the 95th percentile of the difference there)."""
    from ..retouch import temporal as TM
    found = []
    n = min(len(src), len(out))
    D = [out[i] - src[i] for i in range(n)]
    for t in range(1, n - 1):
        if max(np.abs(D[t]).max(), np.abs(D[t - 1]).max(), np.abs(D[t + 1]).max()) < FLICK_LEVEL:
            continue
        ds = []
        for k in (t - 1, t + 1):
            fw = _flow(src[t], src[k])
            bw = _flow(src[k], src[t])
            tr = TM._trust(fw, bw, TM.STEADY_FB)
            ds.append(tr * TM._pull(D[k], fw) + (1.0 - tr) * D[t])
        a, b, c = ds[0], D[t], ds[1]
        med = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
        r = np.abs(D[t] - med)
        m = (cv2.GaussianBlur(r, (0, 0), 1.0) > FLICK_LEVEL).astype(np.uint8)
        k_, lab, stats, cen = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, k_):
            if stats[i, cv2.CC_STAT_AREA] >= FLICK_PX:
                sel = lab == i
                found.append(dict(t=int(t), px=int(stats[i, cv2.CC_STAT_AREA]), level=round(float(np.percentile(r[sel], 95)), 2),
                                  x=int(cen[i][0]), y=int(cen[i][1])))
    return found


def face_series(src: List[np.ndarray], out: List[np.ndarray], boxes: dict) -> dict:
    """{pid: [mean |change| in the face's box, or None where the face is not there]} (boxes: {pid: [(x0, y0, x1, y1) of the QA picture | None]})"""
    res = {}
    for pid, bl in boxes.items():
        s_ = []
        for t, bx in enumerate(bl[:min(len(src), len(out))]):
            if bx is None:
                s_.append(None)
                continue
            x0, y0, x1, y1 = bx
            d = np.abs(out[t][y0:y1, x0:x1] - src[t][y0:y1, x0:x1])
            s_.append(round(float(d.mean()), 3) if d.size else None)
        res[pid] = s_
    return res


def series_findings(series: dict) -> List[dict]:
    """[{pid, t, kind: "drop" | "step", value, near}] in the faces' change series"""
    out = []
    for pid, ser in series.items():
        T = len(ser)
        for t, x in enumerate(ser):
            if x is None:
                continue
            near = [ser[u] for u in range(max(0, t - 3), min(T, t + 4)) if u != t and ser[u] is not None]
            if len(near) >= 2:
                md = float(np.median(near))
                if md > DROP_MIN and x < DROP_SHARE * md:
                    out.append(dict(pid=pid, t=t, kind="drop", value=x, near=round(md, 3)))
                    continue
            if t > 0 and ser[t - 1] is not None:
                b = ser[t - 1]
                if max(x, b) > STEP_MIN and (x > STEP_RATIO * b or x < b / STEP_RATIO):
                    out.append(dict(pid=pid, t=t, kind="step", value=x, near=b))
    return out


def _boxes(tracker, cut, n: int, start: int, sx: float, sy: float, W: int, H: int) -> dict:
    """the targets' face boxes of the QA picture per frame of the render (from the analysis' landmarks)"""
    st = getattr(tracker, "store", None)
    res = {}
    if st is None or not getattr(st, "has_face", False):
        return res
    for person in cut.targets():
        if not st.has(person.id):
            continue
        p = st.index(person.id)
        bl = []
        for i in range(n):
            t = start + i
            if t >= st.face.shape[1] or not np.isfinite(st.face[p, t, :, :2]).all():
                bl.append(None)
                continue
            f = st.face[p, t, :, :2]
            x0, y0 = f.min(0)
            x1, y1 = f.max(0)
            x0, x1 = int(max(0, x0 * sx)), int(min(W, x1 * sx + 1))
            y0, y1 = int(max(0, y0 * sy)), int(min(H, y1 * sy + 1))
            bl.append((x0, y0, x1, y1) if (x1 - x0) >= 6 and (y1 - y0) >= 6 else None)
        res[int(person.id)] = bl
    return res


def _sheet(path: Path, src: np.ndarray, out: np.ndarray, x: int, y: int, r: int, label: str) -> None:
    H, W = src.shape
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(W, x + r), min(H, y + r)
    tiles = [src[y0:y1, x0:x1], out[y0:y1, x0:x1], np.clip(128 + (out - src)[y0:y1, x0:x1] * 8, 0, 255)]
    tiles = [cv2.resize(np.clip(t_, 0, 255).astype(np.uint8), (320, 320), interpolation=cv2.INTER_CUBIC) for t_ in tiles]
    img = cv2.cvtColor(np.hstack(tiles), cv2.COLOR_GRAY2BGR)
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imencode(".jpg", img)[1].tofile(str(path))


def check(project, cut, tracker, out_path: str, a: int, b: int, progress: Optional[Callable] = None, cancelled=None) -> dict:
    """The rendered file out_path of the cut's frames a..b checked (see the top). Returns the report: {ok, problems: [text], format, frames,
    flicker: [...], faces: {pid: series}, findings: [...], folder}."""
    t0 = time.time()
    rep = dict(ok=True, problems=[], file=str(out_path))
    n = int(b) - int(a)
    info = cut.info
    try:
        got = MD.probe(str(out_path))
    except Exception as e:
        rep.update(ok=False, problems=[f"렌더 파일을 읽지 못했습니다 ({str(e)[:80]})"])
        return rep
    fmt = dict(frames=int(got.frames), width=int(got.width), height=int(got.height), fps=round(float(got.fps), 3))
    rep["format"] = fmt
    if int(got.frames) != n:
        rep["problems"].append(f"프레임 수가 다릅니다 (파일 {got.frames}, 컷 {n})")
    if (int(got.width), int(got.height)) != (int(info.width), int(info.height)):
        rep["problems"].append(f"크기가 다릅니다 (파일 {got.width}x{got.height})")
    if abs(float(got.fps) - float(info.fps)) > 0.01:
        rep["problems"].append(f"프레임 속도가 다릅니다 (파일 {got.fps:.3f})")
    w = min(QA_W, int(info.width))
    h = max(2, int(round(int(info.height) * w / float(info.width))) // 2 * 2)
    if progress:
        progress(0.0, "렌더 검수: 프레임 읽는 중")
    src = _read_all(info, a, n, w, h, cancelled)
    out = _read_all(got, 0, min(n, int(got.frames)), w, h, cancelled)
    m = min(len(src), len(out))
    rep["frames"] = m
    # 2. the frames in their places
    for t in sorted({0, m // 2, m - 1}):
        if t < 0 or t >= m:
            continue
        own = float(np.abs(out[t] - src[t]).mean())
        for u in (t - 1, t + 1):
            if 0 <= u < m and float(np.abs(out[t] - src[u]).mean()) < own * 0.8:
                rep["problems"].append(f"프레임이 밀렸습니다 (파일의 {t}번 프레임이 원본 {u}번과 더 같음)")
                break
    if progress:
        progress(0.5, "렌더 검수: 얼굴별 보정량")
    sx, sy = w / float(info.width), h / float(info.height)
    series = face_series(src[:m], out[:m], _boxes(tracker, cut, m, int(a), sx, sy, w, h))
    rep["faces"] = series
    fnd = series_findings(series)
    rep["findings"] = fnd
    if progress:
        progress(0.7, "렌더 검수: 한 프레임 변화")
    fl = one_frame_changes(src[:m], out[:m])
    rep["flicker"] = fl
    folder = Path(project.cut_dir(cut)) / "qa" / Path(str(out_path)).stem
    folder.mkdir(parents=True, exist_ok=True)
    for f_ in folder.glob("*.jpg"):
        try:
            f_.unlink()
        except OSError:
            pass
    rep["folder"] = str(folder)
    sheets = 0
    for e in sorted(fl, key=lambda e: -e["px"] * e["level"])[:MAX_SHEETS]:
        t = e["t"]
        _sheet(folder / f"flicker_f{a + t:05d}.jpg", src[t], out[t], e["x"], e["y"], 48, f"f{a + t} one frame {e['level']:.1f}")
        sheets += 1
    boxes = None
    for e in fnd[:max(0, MAX_SHEETS - sheets)]:
        if boxes is None:
            boxes = _boxes(tracker, cut, m, int(a), sx, sy, w, h)
        bx = (boxes.get(e["pid"]) or [None] * m)[e["t"]]
        if bx is None:
            continue
        x0, y0, x1, y1 = bx
        _sheet(folder / f"face_p{e['pid']}_f{a + e['t']:05d}_{e['kind']}.jpg", src[e["t"]], out[e["t"]], (x0 + x1) // 2, (y0 + y1) // 2,
               max(24, (x1 - x0) // 2 + 8), f"f{a + e['t']} p{e['pid']} {e['kind']} {e['value']} (near {e['near']})")
    if fl:
        rep["problems"].append(f"한 프레임에만 있는 변화 {len(fl)}곳 (프레임 {', '.join(str(a + e['t']) for e in fl[:6])}{' …' if len(fl) > 6 else ''})")
    drops = [e for e in fnd if e["kind"] == "drop"]
    if drops:
        rep["problems"].append(f"얼굴 보정이 빠진 프레임 {len(drops)}개 (프레임 {', '.join(str(a + e['t']) for e in drops[:6])}{' …' if len(drops) > 6 else ''})")
    rep["ok"] = not rep["problems"]
    rep["seconds"] = round(time.time() - t0, 1)
    try:
        (folder / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass
    return rep
