"""Full-resolution render: native-plane decode -> warp/separate -> ProRes/EXR encode (+ mattes)."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import field as F
from .separate import composite
from ..retouch import engine as RE
from ..core import media as M
from ..core.colour import PASSTHROUGH, manager
from ..core.project import OutputSettings, Project, safe_name, Cut

Progress = Optional[Callable[[float, str], None]]


@dataclass
class RenderReport:
    out_path: str = ""
    matte_paths: List[str] = field(default_factory=list)
    frames: int = 0
    seconds: float = 0.0
    inpaint_frames: List[Tuple[int, int]] = field(default_factory=list)   # (frame, pixels)
    warnings: List[str] = field(default_factory=list)
    cancelled: bool = False
    accel: str = ""                       # how the warp ran (GPU / CPU, and why)


def output_base(project: Project, cut: Cut, out: OutputSettings) -> Path:
    folder = Path(out.out_dir) if out.out_dir else (Path(project.path).parent / "DIAP_Renders")
    ver = int(getattr(cut, "version", 1) or 1)
    try:
        name = out.name_pattern.format(cut=cut.name, project=project.name, date=time.strftime("%Y%m%d"), version=ver)
    except (KeyError, IndexError):
        name = cut.name
    if "{version}" not in out.name_pattern:
        name = f"{name}_v{ver}"            # every render carries the cut's work version
    return folder / safe_name(name)


def _proc_format(cut: Cut, out: OutputSettings) -> str:
    rgb_needed = out.format == "exr_half" or cut.info.is_float or bool(out.output_cs)
    if rgb_needed:
        return "gbrpf32le"
    if out.format in ("prores_4444", "prores_4444_xq"):
        return "yuv444p10le"
    return "yuv422p10le"


def _matrix_name(info: M.MediaInfo) -> str:
    return M.swscale_matrix(info) if M.swscale_matrix(info) in ("bt709", "bt601", "bt2020") else "bt709"


class _Lookahead:
    """Sequential reader with a sliding window of decoded frames (for temporal background fill)."""

    def __init__(self, reader: M.FrameReader, start: int, end: int, behind: int, ahead: int):
        self.r = reader
        self.start, self.end = start, end
        self.buf: "OrderedDict[int, List[np.ndarray]]" = OrderedDict()
        self.next_read = start
        self.behind, self.ahead = behind, ahead

    def _fill_to(self, f: int):
        while self.next_read <= min(f, self.end - 1):
            p = self.r.read()
            if p is None:
                self.end = self.next_read
                break
            self.buf[self.next_read] = [np.array(x, copy=True) for x in p]
            self.next_read += 1

    def get(self, f: int) -> Optional[List[np.ndarray]]:
        if f < self.start or f >= self.end:
            return None
        self._fill_to(f)
        return self.buf.get(f)

    def advance(self, t: int):
        self._fill_to(t + self.ahead)
        for k in list(self.buf):
            if k < t - self.behind:
                del self.buf[k]
            else:
                break


def preview_rgb(planes, pf: str, maxv: float, width: int = 480) -> np.ndarray:
    """A small 8-bit RGB picture of output planes (as written, in the source encoding) for the render page."""
    import cv2
    if pf == "gbrpf32le":
        g, b, r = planes[:3]
        h, w = g.shape
        tw, th = width, max(2, int(round(width * h / w)))
        rgb = np.dstack([cv2.resize(np.asarray(x, np.float32), (tw, th), interpolation=cv2.INTER_AREA)
                         for x in (r, g, b)])
        return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    y = np.asarray(planes[0], np.float32)
    h, w = y.shape
    tw, th = width, max(2, int(round(width * h / w)))
    sc = (float(maxv) + 1.0) / 256.0                 # code values in 8-bit steps
    Y = (cv2.resize(y, (tw, th), interpolation=cv2.INTER_AREA) / sc - 16.0) / 219.0
    U = (cv2.resize(np.asarray(planes[1], np.float32), (tw, th), interpolation=cv2.INTER_AREA) / sc - 128.0) / 224.0
    V = (cv2.resize(np.asarray(planes[2], np.float32), (tw, th), interpolation=cv2.INTER_AREA) / sc - 128.0) / 224.0
    rgb = np.dstack([Y + 1.5748 * V, Y - 0.1873 * U - 0.4681 * V, Y + 1.8556 * U])
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def render_cut(project: Project, cut: Cut, tracker, out: OutputSettings, progress: Progress = None,
               cancel: Optional[Callable[[], bool]] = None, ext_writer=None,
               preview: Optional[Callable[[int, np.ndarray], None]] = None) -> RenderReport:
    """ext_writer: write the frames into this (an edited timeline's output) instead of a file of its own;
    it is neither closed nor aborted here, and no mattes are written.
    preview(frame, rgb): about once a second, a small picture of the frame just written."""
    rep = RenderReport()
    t_start = time.time()
    # the face layer (parsing, steady skin colour) for cuts analysed before it existed
    try:
        from ..track import facelayer as FL, faceparse as FP
        if tracker is not None and getattr(tracker, "store", None) is not None and FP.available() \
                and getattr(tracker.store, "has_face", False) and not FL.path_for(tracker).exists():
            FL.build(tracker, (lambda f, m: progress(0.0, m)) if progress else None, cancel)
    except Exception:
        import traceback
        traceback.print_exc()
    info = cut.info
    a, b = cut.work_range
    n = b - a
    W, H = info.width, info.height
    pf = _proc_format(cut, out)
    is_exr = out.format == "exr_half"
    base = output_base(project, cut, out)
    if ext_writer is None:
        base.parent.mkdir(parents=True, exist_ok=True)
    layout = M.plane_layout(pf, W, H)
    subs = layout.subsample
    maxv = layout.max_value
    grid_step = max(2, int(round(W / 480)))
    separate = cut.mode == "separate"
    need_alpha = separate or out.matte_person
    cm = manager()
    in_cs = cut.input_cs or PASSTHROUGH
    convert_cs = bool(out.output_cs) and in_cs != PASSTHROUGH and out.output_cs != in_cs
    # log footage: the retouch runs on the display picture the operator sees, its change is carried back into the log
    disp_ = None
    if in_cs != PASSTHROUGH:
        _v = _vs_for(cut)
        _v.exposure = 0.0
        disp_ = (cm, _v, cut.info.color_range)

    # ---------------- writers
    writer = None
    exr_dir = None
    seq_offset = info.seq_start if info.kind == "sequence" else 0
    tc = info.tc_at(a) if info.timecode else ""
    if ext_writer is not None:
        writer = ext_writer
        rep.out_path = getattr(ext_writer, "path", "")
    elif is_exr:
        exr_dir = Path(M.unique_path(str(base)))
        writer = M.SequenceWriter(str(exr_dir), exr_dir.name, W, H, seq_offset + a)
        rep.out_path = str(exr_dir)
    else:
        path = M.unique_path(str(base) + ".mov")
        audio = None
        if out.include_audio and info.has_audio and info.kind == "video":
            audio = (info.path, (a + info.offset) / info.fps, n / info.fps)
        writer = M.FrameWriter(path, W, H, info.fps_num, info.fps_den, pf, out.format, info, timecode=tc,
                               audio_from=audio)
        rep.out_path = path
    matte_writers: Dict[str, object] = {}
    for kind, on in (("person", out.matte_person), ("warp", out.matte_warp)):
        if not on or ext_writer is not None:
            continue
        if is_exr:
            d = Path(str(exr_dir) + f"_matte_{kind}")
            matte_writers[kind] = M.SequenceWriter(str(d), d.name, W, H, seq_offset + a, gray=True)
            rep.matte_paths.append(str(d))
        else:
            mp = M.unique_path(str(base) + f"_matte_{kind}.mov")
            matte_writers[kind] = M.FrameWriter(mp, W, H, info.fps_num, info.fps_den, "gray16le", "prores_4444",
                                                info, timecode=tc, gray=True)
            rep.matte_paths.append(mp)

    # ---------------- reading
    reader = M.FrameReader(info, a, n, pf, prefetch=4)
    look = _Lookahead(reader, a, b, behind=26 if separate else 0, ahead=26 if separate else 0)
    matter = None
    if need_alpha:
        try:
            from ..track.infer import Matter
            matter = Matter()
        except Exception as e:
            rep.warnings.append(f"매트 모델을 불러오지 못해 분석용 매트를 사용합니다: {e}")
    matte_scale = (tracker.matte.w / W, tracker.matte.h / H)
    store = tracker.store
    homog = (lambda f0, f1: store.homography(f0, f1)) if store is not None else (lambda f0, f1: np.eye(3))
    mtx = _matrix_name(info)

    alpha_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def people_alpha_full(f: int) -> Optional[np.ndarray]:
        if f in alpha_cache:
            return alpha_cache[f]
        al = tracker.matte.get(f)
        if al is None:
            return None
        res = cv2.resize(np.asarray(al), (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        alpha_cache[f] = res
        while len(alpha_cache) > 60:
            alpha_cache.popitem(last=False)
        return res

    def neighbour(f: int):
        pl = look.get(f)
        if pl is None:
            return None
        pa = people_alpha_full(f)
        if pa is None:
            return None
        pa = cv2.dilate(pa, np.ones((7, 7), np.uint8))
        return pl, pa

    def to_rgb_small(planes, max_side=1920):
        s = min(1.0, max_side / max(W, H))
        size = (max(2, int(W * s) // 2 * 2), max(2, int(H * s) // 2 * 2))
        if pf == "gbrpf32le":
            g, b_, r = planes
            rgb = np.dstack([r, g, b_])
            if in_cs != PASSTHROUGH:
                rgb = cm.to_display(rgb, _vs_for(cut))
            return cv2.resize(np.clip(rgb, 0, 1), size, interpolation=cv2.INTER_AREA)
        return np.clip(M.yuv_to_rgb(planes, layout, mtx, info.color_range, out_size=size), 0, 1)

    def target_alpha(f: int, planes) -> np.ndarray:
        if matter is not None:
            small = to_rgb_small(planes)
            al = matter(small.astype(np.float32))
            al = cv2.resize(al, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            al = people_alpha_full(f)
            if al is None:
                al = np.zeros((H, W), np.float32)
        lab = tracker.labels(f)
        if lab is None:
            return al * 0
        tgt_idx = [store.index(p.id) for p in cut.targets() if store.has(p.id)]
        mask = np.isin(lab, tgt_idx).astype(np.uint8)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
        return np.clip(al * mask, 0, 1)

    # ---------------- speed: frames in parallel (warp mode) and the warp on the GPU when that is faster here
    import os
    import threading
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    parallel = not separate and not out.matte_person
    n_workers = max(1, min(4, (os.cpu_count() or 2) // 2)) if parallel else 1
    if parallel and os.environ.get("CONTOUR_RENDER_THREADS", "").isdigit():
        n_workers = max(1, int(os.environ["CONTOUR_RENDER_THREADS"]))
    prep_lock = threading.Lock()
    warper = [None, False]                 # GPU warper (None = CPU), decided

    def do_warp(planes_, fr_):
        if warper[0] is not None:
            try:
                return warper[0].warp_planes(planes_, subs, fr_, maxv)
            except Exception as e:
                warper[0] = None
                rep.warnings.append(f"GPU 워프 오류로 CPU로 전환: {e.__class__.__name__}")
        return F.warp_planes(planes_, subs, fr_, maxv)

    def process(t_, planes_):
        with prep_lock:                    # tracker caches are not shared between threads
            rf_ = RE.prepare(cut, tracker, t_)
            fi_ = F.build_input(cut, tracker, t_)
        if rf_ is not None:
            planes_ = RE.apply_planes(planes_, subs, maxv, pf, rf_, 1.0, disp_)
        fr_ = F.compute_field(fi_, W, H, grid_step)
        if not warper[1] and not fr_.empty():
            with prep_lock:
                if not warper[1]:
                    from .gpu import choose
                    warper[0], line = choose(planes_, subs, fr_, maxv)
                    rep.accel = line
                    warper[1] = True
        return planes_, fr_, do_warp(planes_, fr_)

    last_preview = [0.0]

    def write_frame(t_, outp_, fr_, alpha_out_=None):
        if convert_cs and pf == "gbrpf32le":
            g, b_, r = outp_
            rgb = cm.convert(np.dstack([r, g, b_]), in_cs, out.output_cs)
            outp_ = [rgb[..., 1].copy(), rgb[..., 2].copy(), rgb[..., 0].copy()]
        writer.write(outp_)
        if preview is not None and time.time() - last_preview[0] >= 1.0:
            last_preview[0] = time.time()
            try:
                preview(int(t_), preview_rgb(outp_, pf, maxv))
            except Exception:
                pass
        if "person" in matte_writers:
            am = alpha_out_ if alpha_out_ is not None else np.zeros((H, W), np.float32)
            _write_matte(matte_writers["person"], am, is_exr, seq_offset + t_)
        if "warp" in matte_writers:
            if fr_.empty():
                wm = np.zeros((H, W), np.float32)
            else:
                D = F.upsample(fr_)
                wm = np.clip(np.hypot(D[..., 0], D[..., 1]) / 3.0, 0, 1)
            _write_matte(matte_writers["warp"], wm, is_exr, seq_offset + t_)
        rep.frames += 1
        if progress:
            el = time.time() - t_start
            fps = rep.frames / max(el, 1e-3)
            eta = (n - rep.frames) / max(fps, 1e-3)
            progress(rep.frames / max(1, n), f"렌더 {rep.frames}/{n} · {fps:.1f} fps · 남은 시간 {int(eta // 60)}분 {int(eta % 60)}초")

    if parallel and n_workers > 1:
        pool = ThreadPoolExecutor(max_workers=n_workers)
        pending: "deque" = deque()
        try:
            for t in range(a, b):
                if cancel and cancel():
                    rep.cancelled = True
                    break
                look.advance(t)
                planes = look.get(t)
                if planes is None:
                    rep.warnings.append(f"프레임 {t}에서 디코딩이 끝났습니다.")
                    break
                pending.append((t, pool.submit(process, t, planes)))
                while len(pending) > n_workers * 2 or (pending and pending[0][1].done()):
                    t0_, fut = pending.popleft()
                    _, fr0, outp0 = fut.result()
                    write_frame(t0_, outp0, fr0)
            while pending and not rep.cancelled:
                t0_, fut = pending.popleft()
                _, fr0, outp0 = fut.result()
                write_frame(t0_, outp0, fr0)
        except Exception:
            pool.shutdown(wait=False, cancel_futures=True)
            reader.close()
            if writer and ext_writer is None:
                writer.abort()
            for w in matte_writers.values():
                w.abort()
            raise
        pool.shutdown(wait=True, cancel_futures=True)
        a_range = range(0)
    else:
        a_range = range(a, b)

    try:
        for i, t in enumerate(a_range):
            if cancel and cancel():
                rep.cancelled = True
                break
            look.advance(t)
            planes = look.get(t)
            if planes is None:
                rep.warnings.append(f"프레임 {t}에서 디코딩이 끝났습니다.")
                break
            rf = RE.prepare(cut, tracker, t)
            if rf is not None:
                planes = RE.apply_planes(planes, subs, maxv, pf, rf, 1.0, disp_)
            fi = F.build_input(cut, tracker, t)
            fr = F.compute_field(fi, W, H, grid_step)
            alpha_out = None
            if separate and not fr.empty():
                ta = target_alpha(t, planes)
                outp, alpha_out, missing = composite(planes, subs, fr, ta, t, neighbour, homog, (1.0, 1.0), maxv)
                if missing:
                    rep.inpaint_frames.append((t, missing))
            else:
                if not warper[1] and not fr.empty():
                    from .gpu import choose
                    warper[0], line = choose(planes, subs, fr, maxv)
                    rep.accel = line
                    warper[1] = True
                outp = do_warp(planes, fr)
                if out.matte_person:
                    ta = target_alpha(t, planes)
                    alpha_out = np.clip(F.warp_planes([ta], [(1, 1)], fr, 1.0)[0], 0, 1) if not fr.empty() else ta
            write_frame(t, outp, fr, alpha_out)
    except Exception:
        reader.close()
        if writer and ext_writer is None:
            writer.abort()
        for w in matte_writers.values():
            w.abort()
        raise
    reader.close()
    if writer and ext_writer is None:
        if rep.cancelled:
            writer.abort()
        else:
            writer.close()
    for w in matte_writers.values():
        if rep.cancelled:
            w.abort()
        else:
            w.close()
    rep.seconds = time.time() - t_start
    if parallel and n_workers > 1:
        rep.accel = (rep.accel + " · " if rep.accel else "") + f"프레임 {n_workers}개 동시 처리"
    if rep.inpaint_frames:
        rep.warnings.append(f"{len(rep.inpaint_frames)}개 프레임에서 배경 일부를 공간 인페인팅으로 채웠습니다.")
    return rep


def _write_matte(w, alpha: np.ndarray, is_exr: bool, frame_no: int) -> None:
    if is_exr:
        w.write([alpha.astype(np.float32)])
    else:
        w.write([np.clip(alpha * 65535 + 0.5, 0, 65535).astype(np.uint16)])


def _vs_for(cut: Cut):
    from ..core.colour import ViewSettings
    from ..core.settings import settings
    s = settings()
    return ViewSettings(input_cs=cut.input_cs or PASSTHROUGH, display=s.display, view=s.view, lut_path=cut.lut_path)
