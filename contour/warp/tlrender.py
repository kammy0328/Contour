"""Output of an edited timeline: every shot, retouched or as it was, in edit order.

Modes
- full     one file of the whole baked length (same frame count, start timecode and audio as the baked video)
- each     one file per shot, named after the shot, with that shot's timecode
- replace  only the retouched shots as files, plus an XML (FCP7/Premiere/Resolve) that puts them on V2 over
           the baked video on V1, every one at its own place

Which shots are retouched: analysed shots whose status is wanted for export ("include"), with at least one
beauty target. A shot not exported is, in `full`: transparent (alpha 0) in ProRes 4444 XQ, the original
picture in every other format; in `each`/`replace`: no file.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import quote
from xml.sax.saxutils import escape

import numpy as np

from ..core import media as M
from ..core.project import OutputSettings, Project, safe_name
from ..track import status as ST

FORMATS = [("prores_4444_xq", "ProRes 4444 XQ + 알파 (.mov)"), ("prores_422_hq", "ProRes 422 HQ (.mov)"),
           ("hevc_mp4", "H.265 10bit 고화질 (.mp4)")]


@dataclass
class TimelineExport:
    mode: str = "full"                       # full | each | replace
    format: str = "prores_4444_xq"
    encoder: str = "auto"                    # H.265: auto | nvenc | x265
    quality: str = "high"                    # H.265: normal | high | max
    include: Dict[str, bool] = field(default_factory=lambda: dict(good=True, bad=True, small=True, none=True,
                                                                  pending=True))
    out_dir: str = ""
    name: str = ""
    audio: bool = True
    selected: List[str] = field(default_factory=list)   # cut ids; empty = all

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in (d or {})})


@dataclass
class TimelineReport:
    paths: List[str] = field(default_factory=list)
    frames: int = 0
    seconds: float = 0.0
    retouched: int = 0
    original: int = 0
    transparent: int = 0
    skipped: int = 0
    warnings: List[str] = field(default_factory=list)
    cancelled: bool = False
    out_path: str = ""


def plan(project: Project, exp: TimelineExport) -> List[dict]:
    """[{cut, action}] in edit order; action: retouch | original | alpha | skip."""
    cuts = project.timeline_cuts()
    sel = set(exp.selected or [])
    rows = []
    for c in cuts:
        s = ST.effective(c)
        wanted = exp.include.get(s, True) and (not sel or c.id in sel)
        can = c.analyzed and bool(c.targets()) and s not in ("none",)
        if wanted:
            act = "retouch" if can else "original"
        else:
            if exp.mode == "full":
                act = "alpha" if exp.format == "prores_4444_xq" else "original"
            else:
                act = "skip"
        if exp.mode == "replace" and act != "retouch":
            act = "skip"
        rows.append(dict(cut=c, action=act, status=s))
    return rows


def _ext(fmt: str) -> str:
    return ".mp4" if fmt == "hevc_mp4" else ".mov"


def _pf(fmt: str) -> str:
    return "yuv444p10le" if fmt in ("prores_4444", "prores_4444_xq") else "yuv422p10le"


class _Out:
    """A writer that takes planes in the processing format; adds an alpha plane for 4444 XQ."""

    def __init__(self, path: str, info: M.MediaInfo, exp: TimelineExport, timecode: str, audio):
        self.fmt = exp.format
        self.alpha = exp.format == "prores_4444_xq"
        self.pf = _pf(exp.format)
        lay = M.plane_layout(self.pf, info.width, info.height)
        self.maxv = lay.max_value
        self.ones = None
        self.zero_frame = None
        self.w = M.FrameWriter(path, info.width, info.height, info.fps_num, info.fps_den,
                               ("yuva444p10le" if self.alpha else self.pf), exp.format, info, timecode=timecode,
                               audio_from=audio, encoder=exp.encoder, quality=exp.quality)
        self.path = path
        self.info = info

    def write(self, planes, opaque: bool = True):
        if self.alpha:
            if self.ones is None:
                self.ones = np.full(planes[0].shape, int(self.maxv), np.uint16)
            planes = list(planes) + [self.ones if opaque else np.zeros_like(self.ones)]
        self.w.write(planes)

    def write_clear(self):
        """A transparent black frame (4444 XQ)."""
        if self.zero_frame is None:
            H, W = self.info.height, self.info.width
            mx = int(self.maxv)
            y = np.full((H, W), int(round(64 / 1023 * mx)), np.uint16)
            c = np.full((H, W), int(round(512 / 1023 * mx)), np.uint16)
            self.zero_frame = [y, c, c, np.zeros((H, W), np.uint16)]
        self.w.write(self.zero_frame)

    def close(self):
        self.w.close()

    def abort(self):
        self.w.abort()


def _copy(out: _Out, cut, cancel, tick, a: int = 0, b: Optional[int] = None) -> int:
    b = cut.info.frames if b is None else b
    n = b - a
    if n <= 0:
        return 0
    k = 0
    with M.FrameReader(cut.info, a, n, out.pf, prefetch=4) as rd:
        for pl in rd:
            if cancel and cancel():
                break
            out.write(pl)
            k += 1
            tick()
    if k < n:
        raise M.MediaError(f"{cut.name}: {k}/{n}프레임만 읽혔습니다")
    return k


def _clear(out: _Out, cut, cancel, tick) -> int:
    for _ in range(cut.info.frames):
        if cancel and cancel():
            break
        out.write_clear()
        tick()
    return cut.info.frames


def _retouch(project: Project, out: _Out, cut, cancel, tick, rep: TimelineReport, progress_msg) -> int:
    from ..core.proxy import Proxy
    from ..core.settings import settings
    from ..track.analyze import CutTracker
    from .render import render_cut
    px = Proxy(project.cache_dir(cut), cut.info, settings().proxy_long_side)
    if not px.has_frames():
        rep.warnings.append(f"{cut.name}: 프록시가 없어 원본으로 넣었습니다")
        return _copy(out, cut, cancel, tick)
    tr = CutTracker(project, cut, px)
    o = OutputSettings.from_dict(cut.output.to_dict())
    o.format = "prores_4444_xq" if _pf(out.fmt) == "yuv444p10le" else "prores_422_hq"
    o.output_cs = ""
    o.matte_person = o.matte_warp = False
    o.include_audio = False
    n0 = [0]

    def prog(f, m=""):
        k = int(round(f * cut.info.frames))
        while n0[0] < k:
            n0[0] += 1
            tick()
        progress_msg(m)

    a, b = cut.work_range
    k = _copy(out, cut, cancel, tick, 0, a)          # outside the cut's work range: as it was
    r = render_cut(project, cut, tr, o, prog, cancel, ext_writer=out)
    while n0[0] < r.frames:
        n0[0] += 1
        tick()
    if not r.cancelled and r.frames < b - a:
        raise M.MediaError(f"{cut.name}: 리터칭 프레임이 모자랍니다 ({r.frames}/{b - a})")
    k += r.frames + _copy(out, cut, cancel, tick, b, cut.info.frames)
    rep.warnings += [f"{cut.name}: {w}" for w in r.warnings]
    if r.cancelled:
        rep.cancelled = True
    return k


def render_timeline(project: Project, exp: TimelineExport, progress: Callable = None,
                    cancel: Optional[Callable[[], bool]] = None) -> TimelineReport:
    rep = TimelineReport()
    t0 = time.time()
    tlm = project.timeline or {}
    rows = plan(project, exp)
    if not rows:
        raise M.MediaError("타임라인 컷이 없습니다")
    base_info = M.probe(tlm["baked"]) if tlm.get("baked") and Path(tlm["baked"]).exists() else rows[0]["cut"].info
    folder = Path(exp.out_dir) if exp.out_dir else (Path(project.path).parent / "DIAP_Renders")
    folder.mkdir(parents=True, exist_ok=True)
    name = safe_name(exp.name or tlm.get("name") or project.name)
    work = [r for r in rows if r["action"] != "skip"]
    total = sum(r["cut"].info.frames for r in work) or 1
    done = [0]
    msg = [""]

    def tick():
        done[0] += 1
        if progress and done[0] % 2 == 0:
            el = time.time() - t0
            fps = done[0] / max(el, 1e-3)
            eta = (total - done[0]) / max(fps, 1e-3)
            progress(done[0] / total, f"{msg[0]} · {done[0]}/{total} · {fps:.1f} fps · 남은 시간 {int(eta // 60)}분 {int(eta % 60)}초")

    def set_msg(m):
        pass

    def run_cut(out: _Out, r: dict):
        c = r["cut"]
        act = r["action"]
        msg[0] = f"{c.name} ({ {'retouch': '리터칭', 'original': '원본', 'alpha': '투명'}[act] })"
        if act == "retouch":
            rep.retouched += 1
            return _retouch(project, out, c, cancel, tick, rep, set_msg)
        if act == "alpha":
            rep.transparent += 1
            return _clear(out, c, cancel, tick)
        rep.original += 1
        return _copy(out, c, cancel, tick)

    ext = _ext(exp.format)
    if exp.mode == "full":
        path = M.unique_path(str(folder / f"{name}{ext}"))
        audio = None
        if exp.audio and base_info.has_audio and base_info.kind == "video":
            audio = (base_info.path, base_info.offset / base_info.fps, base_info.frames / base_info.fps)
        tc = base_info.tc_at(0) if base_info.timecode else ""
        out = _Out(path, base_info, exp, tc, audio)
        try:
            for r in rows:
                if cancel and cancel():
                    rep.cancelled = True
                    break
                rep.frames += run_cut(out, r)
                if rep.cancelled:
                    break
        except Exception:
            out.abort()
            raise
        if rep.cancelled:
            out.abort()
        else:
            out.close()
            rep.paths.append(path)
    else:
        sub = folder / f"{name}_{'shots' if exp.mode == 'each' else 'retouched'}"
        sub.mkdir(parents=True, exist_ok=True)
        placed = []
        for r in rows:
            c = r["cut"]
            if r["action"] == "skip":
                rep.skipped += 1
                continue
            if cancel and cancel():
                rep.cancelled = True
                break
            path = M.unique_path(str(sub / f"{c.name}_v{int(getattr(c, 'version', 1) or 1)}{ext}"))
            audio = None
            if exp.audio and exp.mode == "each" and c.info.has_audio and c.info.kind == "video":
                audio = (c.info.path, c.info.offset / c.info.fps, c.info.frames / c.info.fps)
            out = _Out(path, c.info, exp, c.info.tc_at(0) if c.info.timecode else "", audio)
            try:
                rep.frames += run_cut(out, r)
            except Exception:
                out.abort()
                raise
            if rep.cancelled or (cancel and cancel()):
                out.abort()
                rep.cancelled = True
                break
            out.close()
            rep.paths.append(path)
            placed.append((c, path))
        if exp.mode == "replace" and placed and not rep.cancelled:
            xp = M.unique_path(str(folder / f"{name}_retouch_replace.xml"))
            write_replace_xml(xp, project, base_info, placed, name)
            rep.paths.append(xp)
    rep.out_path = rep.paths[0] if rep.paths else ""
    rep.seconds = time.time() - t0
    return rep


# ------------------------------------------------------------------ replacement XML (FCP7 xmeml)
def _url(p: str) -> str:
    p = str(Path(p).resolve()).replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p
    return "file://localhost" + quote(p, safe="/:")


def _rate(info) -> str:
    ntsc = info.fps_den == 1001
    tb = int(round(info.fps))
    return f"<rate><timebase>{tb}</timebase><ntsc>{'TRUE' if ntsc else 'FALSE'}</ntsc></rate>"


def _tc_xml(info, frames: int) -> str:
    tc = M.frames_to_tc(frames, info.fps, info.drop_frame)
    return (f"<timecode>{_rate(info)}<string>{tc}</string><frame>{frames}</frame>"
            f"<displayformat>{'DF' if info.drop_frame else 'NDF'}</displayformat></timecode>")


def _file_xml(fid: str, path: str, info, n: int, tc_frames: int) -> str:
    return (f'<file id="{fid}"><name>{escape(Path(path).name)}</name><pathurl>{escape(_url(path))}</pathurl>'
            f"{_rate(info)}<duration>{n}</duration>{_tc_xml(info, tc_frames)}"
            f"<media><video><samplecharacteristics>{_rate(info)}<width>{info.width}</width>"
            f"<height>{info.height}</height></samplecharacteristics></video></media></file>")


def write_replace_xml(path: str, project: Project, base, placed, name: str) -> None:
    """V1: the baked video over the whole length. V2: each retouched shot at its own place."""
    N = int(base.frames)
    start = base.start_tc_frames() - int(base.offset or 0) if base.timecode else 0
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE xmeml>", '<xmeml version="4">',
             f'<sequence id="sequence-1"><name>{escape(name)}_retouch</name><duration>{N}</duration>{_rate(base)}',
             _tc_xml(base, start), "<media><video>",
             f"<format><samplecharacteristics>{_rate(base)}<width>{base.width}</width><height>{base.height}</height>"
             "<pixelaspectratio>square</pixelaspectratio></samplecharacteristics></format>",
             "<track>",
             f'<clipitem id="clipitem-base"><name>{escape(Path(base.path).name)}</name><enabled>TRUE</enabled>'
             f"<duration>{N}</duration>{_rate(base)}<start>0</start><end>{N}</end><in>0</in><out>{N}</out>"
             + _file_xml("file-base", base.path, base, N, start) + "</clipitem>",
             "</track><track>"]
    for k, (c, p) in enumerate(placed):
        a = int(c.info.offset)
        n = int(c.info.frames)
        parts.append(f'<clipitem id="clipitem-{k + 1}"><name>{escape(Path(p).name)}</name><enabled>TRUE</enabled>'
                     f"<duration>{n}</duration>{_rate(base)}<start>{a}</start><end>{a + n}</end><in>0</in><out>{n}</out>"
                     + _file_xml(f"file-{k + 1}", p, c.info, n, start + a) + "</clipitem>")
    parts += ["</track></video></media></sequence></xmeml>"]
    Path(path).write_text("\n".join(parts), encoding="utf-8")
