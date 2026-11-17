"""Contour 안무영상 도구: a folder of K-pop dance videos -> group-dance tracking data for testing and training.

    python tools/dance_batch.py menu                      (what 안무영상_도구.bat runs)
    python tools/dance_batch.py analyze <video folder> [--out DIR] [--seg 20] [--per-video 2] [--size 1920] [--max]
    python tools/dance_batch.py label    [--out DIR]
    python tools/dance_batch.py evaluate [--out DIR]
    python tools/dance_batch.py export   [--out DIR]

analyze   every video: people per frame (detection, pose, body re-ID, face) with the app's own models on the GPU,
          people linked through the video, and the hardest stretches (members crossing, formation changes,
          people lost and found) cut out as segments. Each segment gets a review video with the person numbers
          drawn on, a contact sheet of every number, and a fixes.txt to note what is wrong.
label     reads every fixes.txt and turns the numbers into the right answer for that segment (and redraws the
          review video with it, to check).
evaluate  runs the current person linking on every labelled segment and scores it against the answer.
export    one zip with the labelled segments (numbers only, no video) to send back for tuning.

Everything stays in Documents\\Contour_Data\\dance (or --out). Videos are only read."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".mxf", ".avi", ".m4v", ".webm", ".mts", ".m2ts"}
FORMAT = 1
TARGET_FPS = 30.0
MIN_TRACK = 3


# =========================================================================== storage

def out_root(arg: Optional[str]) -> Path:
    if arg:
        p = Path(arg)
    else:
        from contour.core.settings import settings
        p = settings().data_path / "dance"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_name(name: str) -> str:
    s = re.sub(r"[\\/:*?\"<>|&]+", "_", name.strip())
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:80] or "video"


def save_people(path: Path, frames: List[List[dict]], meta: dict) -> None:
    rows = [(t, d) for t, fr in enumerate(frames) for d in fr]
    n = len(rows)
    arr = dict(
        frame=np.array([t for t, _ in rows], np.int32),
        box=np.array([d["box"] for _, d in rows], np.float32).reshape(n, 4),
        kp=np.array([d["kp"] for _, d in rows], np.float32).reshape(n, 17, 3),
        score=np.array([d["score"] for _, d in rows], np.float32),
        app=np.array([d["app"] for _, d in rows], np.float16).reshape(n, 72),
        reid=np.zeros((n, 512), np.float16),
        has_reid=np.zeros(n, bool),
        face=np.zeros((n, 128), np.float16),
        face_score=np.zeros(n, np.float32),
        face_w=np.zeros(n, np.float32),
        has_face=np.zeros(n, bool),
    )
    for k, (_, d) in enumerate(rows):
        if d.get("reid") is not None and np.any(d["reid"]):
            arr["reid"][k] = d["reid"]
            arr["has_reid"][k] = True
        if d.get("face") is not None:
            arr["face"][k] = d["face"]["feat"]
            arr["face_score"][k] = d["face"]["score"]
            arr["face_w"][k] = d["face"]["w"]
            arr["has_face"][k] = True
    meta = dict(meta, rows=n, format=FORMAT)
    tmp = path.with_suffix(".tmp.npz")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, meta=np.array(json.dumps(meta, ensure_ascii=False)), **arr)
    tmp.replace(path)


class People:
    """All detections of one video (rows) and the per-frame lists the linking works on."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            self.meta = json.loads(str(z["meta"]))
            self.a = {k: z[k] for k in z.files if k != "meta"}
        self.n_frames = int(self.meta["n_frames"])
        self.rows_at: List[List[int]] = [[] for _ in range(self.n_frames)]
        for r, t in enumerate(self.a["frame"]):
            self.rows_at[int(t)].append(r)

    def dets(self, t0: int = 0, t1: Optional[int] = None) -> Tuple[List[List[dict]], List[List[int]]]:
        """Frames t0..t1-1 as the linking wants them (frame 0 = t0), and the row of each detection."""
        t1 = self.n_frames if t1 is None else t1
        a = self.a
        out, rows = [], []
        for t in range(t0, t1):
            fr, rr = [], []
            for r in self.rows_at[t]:
                d = dict(i=len(fr), box=a["box"][r].astype(np.float32), kp=a["kp"][r].astype(np.float32),
                         score=float(a["score"][r]), app=a["app"][r].astype(np.float32),
                         reid=a["reid"][r].astype(np.float32) if a["has_reid"][r] else None, face=None)
                if a["has_face"][r]:
                    fe = a["face"][r].astype(np.float32)
                    d["face"] = dict(feat=fe / (np.linalg.norm(fe) + 1e-9), score=float(a["face_score"][r]),
                                     w=float(a["face_w"][r]))
                fr.append(d)
                rr.append(r)
            out.append(fr)
            rows.append(rr)
        return out, rows


def link(dets: List[List[dict]], rows: List[List[int]], n_rows: int) -> np.ndarray:
    """The app's person linking on these frames -> person number per row (1, 2, ... by first appearance; 0 = none)."""
    from contour.track.assoc import track_people
    tr = track_people(dets)
    n = len(dets)
    tr = [t for t in tr if len(t) >= max(MIN_TRACK, int(0.03 * n))] or tr
    tr.sort(key=lambda t: (min(t), -len(t)))
    ids = np.zeros(n_rows, np.int32)
    for k, t in enumerate(tr):
        for f, i in t.items():
            ids[rows[f][i]] = k + 1
    return ids


# =========================================================================== analyze

def find_videos(folder: Path) -> List[Path]:
    out = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT and not p.name.startswith("._"):
            out.append(p)
    return out


def source_sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def analysis_size(w: int, h: int, long_side: int) -> Tuple[int, int]:
    s = min(1.0, long_side / max(w, h))
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


class Models:
    def __init__(self, best: bool):
        from contour.track.infer import Detector, PoseEstimator
        from contour.track.people import load_identity_models
        from contour.track.models import device_label
        print(f"  모델 불러오는 중 (추론 장치: {device_label()})", flush=True)
        self.det = Detector("det_yolox_x" if best else "det_yolox_m", score_thr=0.3, nms_thr=0.6)
        keys = ["pose_rtmw_x"] + (["pose_rtmpose_x"] if best else [])
        self.poses = {k: PoseEstimator(k) for k in keys}
        self.reid, self.face = load_identity_models(True)
        if self.reid is None or self.face is None:
            print("  [주의] 사람 재인식 모델을 불러오지 못했습니다. 생김새·얼굴 없이 진행합니다.", flush=True)


def frames_of(info, start: int, count: int, size: Tuple[int, int], stride: int):
    """(analysis frame index, rgb) for source frames start.. in steps of stride."""
    from contour.core.media import FrameReader
    with FrameReader(info, start, count, "rgb24", size=size) as rd:
        for i, planes in enumerate(rd):
            if i % stride:
                continue
            yield i // stride, np.ascontiguousarray(planes[0].reshape(size[1], size[0], 3))


def analyze_video(src: Path, vdir: Path, models: "Models", long_side: int, args) -> Optional[People]:
    from contour.core.media import probe
    from contour.track.exposure import analysis_view
    from contour.track.people import detect_people
    pp = vdir / "people.npz"
    sig = source_sig(src)
    if pp.exists():
        try:
            p = People(pp)
            ok = p.meta.get("sig") == sig and p.meta.get("format") == FORMAT
        except Exception:
            ok = False
        if ok:
            # the answers in the segment folders point at these rows: an existing analysis is never redone
            print("  이미 분석됨, 건너뜀", flush=True)
            return p
        old = vdir.with_name(f"{vdir.name}_old_{time.strftime('%Y%m%d_%H%M%S')}")
        vdir.rename(old)
        print(f"  영상이 바뀌어 새로 분석합니다 (이전 결과: {old.name})", flush=True)
    vdir.mkdir(parents=True, exist_ok=True)
    info = probe(str(src))
    stride = max(1, int(round(info.fps / TARGET_FPS)))
    size = analysis_size(info.width, info.height, long_side)
    n_src = info.frames
    n = (n_src + stride - 1) // stride
    print(f"  {info.width}x{info.height} {info.fps:.2f}fps {n_src}프레임 -> 분석 {size[0]}x{size[1]}, "
          f"{n}프레임({info.fps / stride:.1f}fps)", flush=True)
    frames: List[List[dict]] = []
    t0 = time.time()
    for t, rgb in frames_of(info, 0, n_src, size, stride):
        people, _ = detect_people(analysis_view(rgb), models.det, models.poses, models.reid, models.face)
        frames.append(people)
        if t % 25 == 0 or t == n - 1:
            el = time.time() - t0
            fps = (t + 1) / max(el, 1e-3)
            eta = (n - t - 1) / max(fps, 1e-3)
            print(f"\r  인물 분석 {t + 1}/{n}  {fps:.1f} fps  남은 시간 {eta / 60:.1f}분   ", end="", flush=True)
    print(flush=True)
    if not frames:
        raise RuntimeError("프레임을 읽지 못했습니다")
    meta = dict(video=str(src), name=src.name, sig=sig, fps=info.fps, stride=stride, eff_fps=info.fps / stride,
                width=info.width, height=info.height, size=list(size), long_side=long_side, n_frames=len(frames),
                n_src=n_src, seconds=round(time.time() - t0, 1), best=bool(args.max))
    save_people(pp, frames, meta)
    return People(pp)


def difficulty(p: People, ids: np.ndarray) -> np.ndarray:
    """Per frame: people overlapping each other, plus person numbers starting or ending there (lost and found)."""
    n = p.n_frames
    box = p.a["box"]
    d = np.zeros(n, np.float32)
    from contour.track.assoc import iou
    for t in range(n):
        rr = p.rows_at[t]
        for i in range(len(rr)):
            for j in range(i + 1, len(rr)):
                if iou(box[rr[i]], box[rr[j]]) > 0.15:
                    d[t] += 1
    fr = p.a["frame"]
    for k in np.unique(ids[ids > 0]):
        f = fr[ids == k]
        lo, hi = int(f.min()), int(f.max())
        if lo > 0:
            d[lo] += 3
        if hi < n - 1:
            d[hi] += 3
    return d


def pick_segments(p: People, ids: np.ndarray, seconds: float, per_video: int) -> List[Tuple[int, int]]:
    n = p.n_frames
    L = max(10, int(round(seconds * p.meta["eff_fps"])))
    if n <= L * 1.2:
        return [(0, n)]
    d = difficulty(p, ids)
    people = np.array([len(r) for r in p.rows_at], np.float32)
    c = np.concatenate([[0], np.cumsum(d)])
    cp = np.concatenate([[0], np.cumsum(people)])
    score = np.array([(c[s + L] - c[s]) for s in range(n - L + 1)])
    crowd = np.array([(cp[s + L] - cp[s]) / L for s in range(n - L + 1)])
    score[crowd < 2.5] = -1           # a stretch with hardly anyone in it teaches nothing about group dance
    out = []
    for _ in range(per_video):
        if not len(score) or score.max() <= 0:
            break
        s = int(np.argmax(score))
        out.append((s, s + L))
        score[max(0, s - L + 1):s + L] = -1
    return sorted(out)


PALETTE = [(255, 99, 132), (54, 162, 235), (255, 206, 86), (75, 192, 120), (153, 102, 255), (255, 159, 64),
           (0, 200, 200), (240, 80, 200), (160, 200, 60), (120, 120, 255), (255, 120, 90), (90, 220, 160),
           (200, 160, 255), (255, 230, 140), (130, 180, 255), (220, 120, 120)]


def color_of(k: int) -> Tuple[int, int, int]:
    return PALETTE[(k - 1) % len(PALETTE)] if k > 0 else (150, 150, 150)


def draw(rgb: np.ndarray, boxes: np.ndarray, ids: List[int], scale: float, t_rel: float, f_rel: int,
         title: str) -> np.ndarray:
    img = rgb.copy()
    th = max(1, int(round(2 * scale * 1.5)))
    for b, k in zip(boxes, ids):
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
        col = color_of(k)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, th if k > 0 else 1)
        if k > 0:
            lab = str(k)
            fs = 1.1
            (tw, tht), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
            cv2.rectangle(img, (x1, max(0, y1 - tht - 12)), (x1 + tw + 12, y1), col, -1)
            cv2.putText(img, lab, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, fs, (10, 10, 10), 2, cv2.LINE_AA)
    hud = f"{t_rel:6.2f}s  #{f_rel}"
    cv2.rectangle(img, (0, 0), (300, 44), (0, 0, 0), -1)
    cv2.putText(img, hud, (10, 31), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    if title:
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.rectangle(img, (img.shape[1] - tw - 20, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, title, (img.shape[1] - tw - 10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return img


class Mp4:
    def __init__(self, path: Path, w: int, h: int, fps: float):
        from contour.core.media import ffmpeg_bin
        self.path = path
        cmd = [ffmpeg_bin(), "-hide_banner", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", f"{fps:.4f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        flags = 0x08000000 if sys.platform == "win32" else 0
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)

    def write(self, rgb: np.ndarray):
        self.p.stdin.write(np.ascontiguousarray(rgb).tobytes())

    def close(self):
        self.p.stdin.close()
        err = self.p.stderr.read().decode(errors="replace")
        if self.p.wait() != 0:
            raise RuntimeError(f"리뷰 영상을 만들지 못했습니다: {err[:300]}")


def render_segment(p: People, seg: dict, ids: np.ndarray, path: Path, title: str = "",
                   sheet: Optional[Path] = None) -> None:
    """Review video of a segment with person numbers (and a contact sheet of every number)."""
    from contour.core.media import probe
    info = probe(p.meta["video"])
    stride = int(p.meta["stride"])
    aw, ah = p.meta["size"]
    vw, vh = analysis_size(aw, ah, 1280)
    sc = vw / aw
    s0, s1 = seg["start"], seg["end"]
    fps = p.meta["eff_fps"]
    mp = Mp4(path, vw, vh, fps)
    thumbs: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    life: Dict[int, List[int]] = {}
    fr = p.a["frame"]
    for k in np.unique(ids[ids > 0]):
        life[int(k)] = sorted(int(x) for x in fr[ids == k])
    want = {}
    for k, ff in life.items():
        picks = [ff[int(round(q * (len(ff) - 1)))] for q in (0.02, 0.35, 0.65, 0.98)]
        for f in picks:
            want.setdefault(f, []).append(k)
    try:
        for t, rgb in frames_of(info, s0 * stride, (s1 - s0) * stride, (aw, ah), stride):
            ta = s0 + t
            if ta >= s1:
                break
            rr = p.rows_at[ta]
            small = cv2.resize(rgb, (vw, vh), interpolation=cv2.INTER_AREA)
            mp.write(draw(small, p.a["box"][rr], [int(ids[r]) for r in rr], sc, t / fps, t, title))
            if sheet is not None and ta in want:
                for k in want[ta]:
                    r = [r for r in rr if ids[r] == k]
                    if r:
                        x1, y1, x2, y2 = [int(v) for v in p.a["box"][r[0]]]
                        crop = rgb[max(0, y1):max(y1 + 2, y2), max(0, x1):max(x1 + 2, x2)]
                        if crop.size:
                            thumbs.setdefault(k, []).append((ta, crop))
    finally:
        mp.close()
    if sheet is not None:
        contact_sheet(thumbs, life, s0, fps, sheet)


def contact_sheet(thumbs, life, s0: int, fps: float, path: Path) -> None:
    keys = sorted(life)
    if not keys:
        return
    cw, ch = 110, 200
    cols = 2
    rows = (len(keys) + cols - 1) // cols
    cell_w = 150 + 4 * (cw + 6)
    sheet = np.full((rows * (ch + 16) + 10, cols * cell_w + 10, 3), 24, np.uint8)
    for n, k in enumerate(keys):
        r, c = divmod(n, cols)
        x0, y0 = 10 + c * cell_w, 10 + r * (ch + 16)
        col = color_of(k)
        cv2.rectangle(sheet, (x0, y0), (x0 + 130, y0 + ch), col, -1)
        cv2.putText(sheet, str(k), (x0 + 12, y0 + 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, (10, 10, 10), 3, cv2.LINE_AA)
        ff = life[k]
        txt = f"{(ff[0] - s0) / fps:.1f}-{(ff[-1] - s0) / fps:.1f}s"
        cv2.putText(sheet, txt, (x0 + 8, y0 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"{len(ff)}f", (x0 + 8, y0 + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1,
                    cv2.LINE_AA)
        for m, (_, crop) in enumerate(thumbs.get(k, [])[:4]):
            h, w = crop.shape[:2]
            s = min(cw / w, ch / h)
            im = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            xx = x0 + 140 + m * (cw + 6) + (cw - im.shape[1]) // 2
            yy = y0 + (ch - im.shape[0]) // 2
            sheet[yy:yy + im.shape[0], xx:xx + im.shape[1]] = im
    cv2.imencode(".jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(path))


FIXES_HELP = """# {name} / 구간 {k} ({t0:.1f}s ~ {t1:.1f}s, {n}프레임)
#
# review.mp4를 보며 번호가 틀린 곳을 아래에 한 줄씩 적으세요. 시간은 영상 왼쪽 위 숫자(초)를 그대로 쓰면 됩니다.
# '#310'처럼 프레임 번호로 써도 됩니다. sheet.jpg에 번호마다 등장 구간과 모습이 있습니다.
#
#   같음 12 3            12번은 3번과 같은 사람 (12번을 3번으로)
#   바꿈 3 5 12.4        12.4초부터 끝까지 3번과 5번을 맞바꿈
#   바꿈 3 5 12.4 15.0   12.4초~15.0초만 맞바꿈
#   나눔 7 20.0          20.0초부터 7번을 새 사람으로 (새 번호는 101, 102 ... 순서로 붙습니다)
#   나눔 7 20.0 25.0     그 구간만 새 사람으로
#   제외 9               사람이 아니거나(소품, 거울 속 사람 등) 평가에서 뺄 번호
#   제외 9 20.0 25.0     그 구간만 빼기
#
# 줄은 위에서부터 차례로 적용됩니다. 다 봤으면 맨 아래 '확인완료' 앞의 # 을 지우세요.
# 틀린 곳이 없으면 '확인완료'만 남기면 됩니다. 확인완료가 없는 구간은 정답으로 쓰지 않습니다.

# 확인완료
"""


def analyze(args) -> None:
    folder = Path(args.folder)
    root = out_root(args.out)
    vids = find_videos(folder)
    if not vids:
        print(f"영상이 없습니다: {folder}")
        return
    print(f"영상 {len(vids)}개, 결과 폴더: {root}\n", flush=True)
    models = Models(bool(args.max))
    log = root / "log.txt"
    for n, src in enumerate(vids):
        name = safe_name(src.stem)
        vdir = root / name
        print(f"[{n + 1}/{len(vids)}] {src.name}", flush=True)
        try:
            p = analyze_video(src, vdir, models, args.size, args)
            make_segments(p, vdir, args.seg, args.per_video)
        except KeyboardInterrupt:
            print("\n중단했습니다. 다시 실행하면 끝난 영상은 건너뜁니다.")
            return
        except Exception as e:
            print(f"\n  [오류] {e}", flush=True)
            with open(log, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {src}\n{traceback.format_exc()}\n")
    print(f"\n끝났습니다. 영상마다 seg 폴더의 review.mp4를 보고 fixes.txt에 고칠 곳을 적은 뒤 '정답 만들기'를 실행하세요.\n"
          f"결과 폴더: {root}")


def make_segments(p: People, vdir: Path, seconds: float, per_video: int) -> None:
    dets, rows = p.dets()
    ids_full = link(dets, rows, len(p.a["frame"]))
    np.save(vdir / "linked.npy", ids_full)
    segs = pick_segments(p, ids_full, seconds, per_video)
    fps = p.meta["eff_fps"]
    summary = dict(video=p.meta["name"], frames=p.n_frames, fps=fps, people_max=int(max(len(r) for r in p.rows_at)),
                   numbers=int(ids_full.max()), segments=[])
    for k, (s0, s1) in enumerate(segs):
        sdir = vdir / f"seg{k + 1:02d}"
        sdir.mkdir(exist_ok=True)
        seg = dict(start=int(s0), end=int(s1), seconds=[round(s0 / fps, 2), round(s1 / fps, 2)])
        summary["segments"].append(seg)
        if (sdir / "auto.npy").exists() and json.loads((sdir / "seg.json").read_text(encoding="utf-8")) == seg:
            continue
        d, r = p.dets(s0, s1)
        ids = link(d, r, len(p.a["frame"]))
        (sdir / "seg.json").write_text(json.dumps(seg), encoding="utf-8")
        np.save(sdir / "auto.npy", ids)
        print(f"  구간 {k + 1}: {s0 / fps:.1f}s~{s1 / fps:.1f}s, 번호 {int(ids.max())}개 -> 리뷰 영상 만드는 중", flush=True)
        render_segment(p, seg, ids, sdir / "review.mp4", sheet=sdir / "sheet.jpg")
        fx = sdir / "fixes.txt"
        if not fx.exists():
            fx.write_text(FIXES_HELP.format(name=p.meta["name"], k=k + 1, t0=s0 / fps, t1=s1 / fps, n=s1 - s0),
                          encoding="utf-8")
    (vdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")


# =========================================================================== label

class FixError(ValueError):
    pass


def _time(tok: str, fps: float) -> int:
    tok = tok.strip()
    if tok.startswith("#"):
        return int(tok[1:])
    if tok.lower().endswith("s"):
        tok = tok[:-1]
    return int(round(float(tok) * fps))


CMD = {"같음": "merge", "merge": "merge", "바꿈": "swap", "swap": "swap", "나눔": "split", "split": "split",
       "제외": "drop", "drop": "drop", "확인완료": "done", "done": "done"}


def parse_fixes(text: str, fps: float) -> Tuple[List[tuple], bool]:
    """Commands of a fixes.txt (lines starting with # are comments; '#310' inside a command is a frame number)."""
    out, done = [], False
    for ln, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        toks = []
        for tok in line.replace(",", " ").split():
            if tok.startswith("#") and not tok[1:].isdigit():
                break                     # the rest of the line is a comment
            toks.append(tok)
        if not toks:
            continue
        cmd = CMD.get(toks[0].lower())
        if cmd is None:
            raise FixError(f"{ln}번째 줄: '{toks[0]}'은(는) 모르는 명령입니다")
        try:
            if cmd == "done":
                done = True
            elif cmd == "merge":
                out.append(("merge", int(toks[1]), int(toks[2])))
            elif cmd == "swap":
                a, b = int(toks[1]), int(toks[2])
                t0 = _time(toks[3], fps) if len(toks) > 3 else 0
                t1 = _time(toks[4], fps) if len(toks) > 4 else 10 ** 9
                out.append(("swap", a, b, t0, t1))
            else:
                a = int(toks[1])
                t0 = _time(toks[2], fps) if len(toks) > 2 else 0
                t1 = _time(toks[3], fps) if len(toks) > 3 else 10 ** 9
                out.append((cmd, a, t0, t1))
        except (IndexError, ValueError):
            raise FixError(f"{ln}번째 줄을 읽지 못했습니다: {line}")
    return out, done


def apply_fixes(ids: np.ndarray, rel_frame: np.ndarray, fixes: List[tuple]) -> Tuple[np.ndarray, List[str]]:
    """ids: automatic numbers per row (0 = none); rel_frame: segment frame of each row (rows outside: -1)."""
    gt = ids.astype(np.int32).copy()
    gt[ids <= 0] = -1
    inside = rel_frame >= 0
    notes = []
    nxt = 101
    for fx in fixes:
        if fx[0] == "merge":
            _, a, b = fx
            m = inside & (gt == a)
            if not m.any():
                notes.append(f"같음 {a} {b}: {a}번이 없습니다")
            gt[m] = b
        elif fx[0] == "swap":
            _, a, b, t0, t1 = fx
            win = inside & (rel_frame >= t0) & (rel_frame < t1)
            ma, mb = win & (gt == a), win & (gt == b)
            gt[ma], gt[mb] = b, a
            if not (ma.any() or mb.any()):
                notes.append(f"바꿈 {a} {b}: 그 구간에 두 번호가 없습니다")
        elif fx[0] == "split":
            _, a, t0, t1 = fx
            m = inside & (gt == a) & (rel_frame >= t0) & (rel_frame < t1)
            if m.any():
                gt[m] = nxt
                notes.append(f"나눔 {a} -> 새 번호 {nxt}")
            else:
                notes.append(f"나눔 {a}: 그 구간에 {a}번이 없습니다")
            nxt += 1
        elif fx[0] == "drop":
            _, a, t0, t1 = fx
            m = inside & (gt == a) & (rel_frame >= t0) & (rel_frame < t1)
            gt[m] = -1
    gt[~inside] = -1
    return gt, notes


def segments(root: Path):
    for vdir in sorted(d for d in root.iterdir() if d.is_dir() and (d / "people.npz").exists()
                       and "_old_" not in d.name):
        for sdir in sorted(vdir.glob("seg*")):
            if (sdir / "auto.npy").exists() and (sdir / "seg.json").exists():
                yield vdir, sdir


def rel_frames(p: People, seg: dict) -> np.ndarray:
    f = p.a["frame"].astype(np.int64)
    rel = f - seg["start"]
    rel[(f < seg["start"]) | (f >= seg["end"])] = -1
    return rel


def label(args) -> None:
    root = out_root(args.out)
    n_done = n_err = n_todo = 0
    cache: Dict[Path, People] = {}
    for vdir, sdir in segments(root):
        tag = f"{vdir.name}/{sdir.name}"
        fx = sdir / "fixes.txt"
        text = fx.read_text(encoding="utf-8") if fx.exists() else ""
        seg = json.loads((sdir / "seg.json").read_text(encoding="utf-8"))
        p = cache.get(vdir) or People(vdir / "people.npz")
        cache = {vdir: p}
        try:
            fixes, done = parse_fixes(text, p.meta["eff_fps"])
        except FixError as e:
            print(f"[오류] {tag}: {e} (이 구간은 정답에서 뺍니다)")
            (sdir / "gt.npy").unlink(missing_ok=True)
            n_err += 1
            continue
        if not done:
            (sdir / "gt.npy").unlink(missing_ok=True)
            n_todo += 1
            continue
        auto = np.load(sdir / "auto.npy")
        rel = rel_frames(p, seg)
        gt, notes = apply_fixes(auto, rel, fixes)
        np.save(sdir / "gt.npy", gt)
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        stamp = sdir / "gt_review.md5"
        for nt in notes:
            print(f"  {tag}: {nt}")
        if fixes and not args.no_video and (not stamp.exists() or stamp.read_text() != h):
            print(f"{tag}: 고친 번호로 리뷰 영상 만드는 중", flush=True)
            render_segment(p, seg, np.where(gt > 0, gt, 0), sdir / "review_정답.mp4", title="FIXED")
            stamp.write_text(h)
        n_done += 1
        print(f"{tag}: 정답 {len(np.unique(gt[gt > 0]))}명, 고친 줄 {len(fixes)}", flush=True)
    print(f"\n정답이 된 구간 {n_done}개, 아직 확인완료 전 {n_todo}개, 오류 {n_err}개")


# =========================================================================== evaluate

def score(gt: np.ndarray, pred: np.ndarray, rel: np.ndarray) -> dict:
    from scipy.optimize import linear_sum_assignment
    m = gt > 0
    g, q, f = gt[m], pred[m], rel[m]
    n_gt = int(m.sum())
    n_pred = int((q > 0).sum())
    gi, pi = np.unique(g), np.unique(q[q > 0])
    C = np.zeros((len(gi), len(pi)))
    for a, b in zip(g, q):
        if b > 0:
            C[np.searchsorted(gi, a), np.searchsorted(pi, b)] += 1
    idtp = C[linear_sum_assignment(-C)].sum() if C.size else 0.0
    idsw = 0
    frag = []
    for a in gi:
        sel = (g == a) & (q > 0)
        order = np.argsort(f[sel], kind="stable")
        seq = q[sel][order]
        idsw += int((seq[1:] != seq[:-1]).sum())
        frag.append(len(np.unique(seq)))
    return dict(IDF1=round(200.0 * idtp / max(1, n_gt + n_pred), 1), IDSW=idsw, members=len(gi),
                numbers=len(pi), per_member=round(float(np.mean(frag)) if frag else 0.0, 2), rows=n_gt)


def evaluate(args) -> None:
    from contour.core.settings import APP_VERSION
    root = out_root(args.out)
    rows = []
    for vdir, sdir in segments(root):
        if not (sdir / "gt.npy").exists():
            continue
        p = People(vdir / "people.npz")
        seg = json.loads((sdir / "seg.json").read_text(encoding="utf-8"))
        d, r = p.dets(seg["start"], seg["end"])
        pred = link(d, r, len(p.a["frame"]))
        res = score(np.load(sdir / "gt.npy"), pred, rel_frames(p, seg))
        res["seg"] = f"{vdir.name}/{sdir.name}"
        rows.append(res)
        print(f"{res['seg']:50s} IDF1 {res['IDF1']:5.1f}  번호 바뀜 {res['IDSW']:3d}  멤버 {res['members']:2d}명  "
              f"번호 {res['numbers']:2d}개", flush=True)
    if not rows:
        print("정답이 있는 구간이 없습니다. fixes.txt에 '확인완료'를 표시하고 '정답 만들기'를 먼저 실행하세요.")
        return
    mean = float(np.mean([r["IDF1"] for r in rows]))
    sw = sum(r["IDSW"] for r in rows)
    line = f"{time.strftime('%Y-%m-%d %H:%M')}  Contour {APP_VERSION}  구간 {len(rows)}개  평균 IDF1 {mean:.1f}  번호 바뀜 합계 {sw}"
    print("\n" + line)
    with open(root / "평가_기록.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
        for r in rows:
            f.write(f"    {r['seg']}  IDF1 {r['IDF1']}  IDSW {r['IDSW']}  members {r['members']}  numbers {r['numbers']}\n")


# =========================================================================== export

def export(args) -> None:
    root = out_root(args.out)
    zp = root / f"Contour_안무_정답_{time.strftime('%Y%m%d_%H%M')}.zip"
    n = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for vdir, sdir in segments(root):
            if not (sdir / "gt.npy").exists():
                continue
            p = People(vdir / "people.npz")
            seg = json.loads((sdir / "seg.json").read_text(encoding="utf-8"))
            m = (p.a["frame"] >= seg["start"]) & (p.a["frame"] < seg["end"])
            sub = {k: v[m] for k, v in p.a.items()}
            sub["frame"] = sub["frame"] - seg["start"]
            meta = dict(p.meta, video=p.meta["name"], n_frames=seg["end"] - seg["start"], seg=seg)
            import io
            buf = io.BytesIO()
            np.savez_compressed(buf, meta=np.array(json.dumps(meta, ensure_ascii=False)), gt=np.load(sdir / "gt.npy")[m],
                                auto=np.load(sdir / "auto.npy")[m], **sub)
            base = f"{vdir.name}/{sdir.name}/"
            z.writestr(base + "segment.npz", buf.getvalue())
            z.write(sdir / "fixes.txt", base + "fixes.txt")
            if (sdir / "sheet.jpg").exists():
                z.write(sdir / "sheet.jpg", base + "sheet.jpg")
            n += 1
        log = root / "평가_기록.txt"
        if log.exists():
            z.write(log, "평가_기록.txt")
    if n == 0:
        zp.unlink(missing_ok=True)
        print("정답이 있는 구간이 없어 보낼 파일을 만들지 않았습니다.")
        return
    print(f"구간 {n}개를 담았습니다 ({zp.stat().st_size / 1e6:.1f}MB): {zp}")


# =========================================================================== menu

def pick_folder() -> Optional[str]:
    try:
        from PySide6.QtWidgets import QApplication, QFileDialog
        app = QApplication.instance() or QApplication(sys.argv)
        d = QFileDialog.getExistingDirectory(None, "안무영상 폴더 선택")
        del app
        return d or None
    except Exception:
        d = input("안무영상 폴더 경로를 붙여 넣으세요: ").strip().strip('"')
        return d or None


def open_folder(p: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(p))  # noqa
    else:
        print(p)


def menu(args) -> None:
    root = out_root(args.out)
    while True:
        print("\n  Contour 안무영상 도구")
        print(f"  결과 폴더: {root}\n")
        print("  1. 영상 분석 (폴더 선택)")
        print("  2. 정답 만들기 (fixes.txt 반영)")
        print("  3. 평가")
        print("  4. 보낼 파일 만들기")
        print("  5. 결과 폴더 열기")
        print("  0. 끝내기")
        c = input("\n  번호: ").strip()
        try:
            if c == "1":
                d = pick_folder()
                if d:
                    analyze(argparse.Namespace(folder=d, out=args.out, seg=args.seg, per_video=args.per_video,
                                               size=args.size, max=args.max))
            elif c == "2":
                label(argparse.Namespace(out=args.out, no_video=False))
            elif c == "3":
                evaluate(argparse.Namespace(out=args.out))
            elif c == "4":
                export(argparse.Namespace(out=args.out))
            elif c == "5":
                open_folder(root)
            elif c in ("0", "q", ""):
                return
        except KeyboardInterrupt:
            print("\n중단했습니다.")
        except Exception:
            traceback.print_exc()


def main(argv=None):
    from contour.core.media import configure_ffmpeg
    from contour.core.settings import settings
    configure_ffmpeg()
    ap = argparse.ArgumentParser(description="Contour 안무영상 도구")
    ap.add_argument("--out", default=None, help="결과 폴더 (기본: 문서\\Contour_Data\\dance)")
    ap.add_argument("--seg", type=float, default=20.0, help="구간 길이(초)")
    ap.add_argument("--per-video", type=int, default=2, help="영상마다 고를 구간 수")
    ap.add_argument("--size", type=int, default=1920, help="분석 해상도(긴 변)")
    ap.add_argument("--max", action="store_true", help="큰 검출 모델과 포즈 모델 2개 (느림)")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("analyze")
    a.add_argument("folder")
    lb = sub.add_parser("label")
    lb.add_argument("--no-video", action="store_true")
    sub.add_parser("evaluate")
    sub.add_parser("export")
    sub.add_parser("menu")
    # options may come before or after the command
    args, rest = ap.parse_known_args(argv)
    if rest:
        args = ap.parse_args(rest, namespace=args)
    if args.cmd == "analyze":
        analyze(args)
    elif args.cmd == "label":
        label(args)
    elif args.cmd == "evaluate":
        evaluate(args)
    elif args.cmd == "export":
        export(args)
    else:
        menu(args)


if __name__ == "__main__":
    main()
