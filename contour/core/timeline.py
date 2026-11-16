"""Edited timelines: where every shot sits in a baked (flattened) export of the sequence.

A beauty job often arrives as one baked video of the whole edit plus the editor's XML. Each shot of the
edit must be handled on its own (tracking restarts at every cut, a shot without people needs nothing),
and the result must line up frame for frame with the edit again.

Read here, in sequence (record) frames, frame 0 = first frame of the sequence:
- Final Cut Pro 7 / Premiere Pro XML (xmeml): clip items on all video tracks; the topmost enabled clip is
  what the baked export shows. Transitions become their own stretch (two shots mixed). start/end = -1
  next to a transition is resolved from the transition. Speed changes (time remap, speed filter) and
  nested sequences are noted per shot. Sequence start timecode and NTSC drop frame are read.
- FCPXML (Final Cut Pro X, DaVinci Resolve): spine items, connected clips on lanes, transitions, rational
  times ("1001/24000s"), sequence tcStart.
- CMX3600 EDL (Resolve, Avid, Premiere): events with record in/out, dissolves (D nnn), speed (M2), FCM.

Then `align()` checks the result against the baked video itself (picture changes at the boundaries) and
finds a constant offset when the export started at a different timecode or with handles.
"""
from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote

import numpy as np


@dataclass
class Shot:
    rec_in: int                      # first sequence frame
    rec_out: int                     # exclusive
    kind: str = "clip"               # clip | transition | gap
    name: str = ""                   # source clip / file name
    src_in: int = -1                 # source frame of rec_in (when known)
    track: int = 1
    speed: float = 100.0             # percent; 100 = normal
    nested: bool = False
    note: str = ""

    @property
    def length(self) -> int:
        return self.rec_out - self.rec_in

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class Timeline:
    shots: List[Shot] = field(default_factory=list)
    fps: float = 24.0
    fps_num: int = 24
    fps_den: int = 1
    start_tc: str = ""               # sequence start timecode
    start_frames: int = 0
    drop: bool = False
    name: str = ""
    fmt: str = ""                    # xmeml | fcpxml | edl
    width: int = 0
    height: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> int:
        return max((s.rec_out for s in self.shots), default=0)


# ------------------------------------------------------------------ helpers
def _num(v, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _rate(el) -> Tuple[float, int, int]:
    """(fps, num, den) from an xmeml <rate>."""
    if el is None:
        return 24.0, 24, 1
    tb = _num(el.findtext("timebase"), 24) or 24
    ntsc = (el.findtext("ntsc") or "").strip().upper() == "TRUE"
    if ntsc:
        return tb * 1000 / 1001, tb * 1000, 1001
    return float(tb), tb, 1


def _tc_frames(tc: str, nominal: int, drop: bool) -> int:
    parts = [int(p) for p in re.split(r"[:;.]", tc.strip()) if p.isdigit()]
    if len(parts) != 4:
        return 0
    h, m, s, f = parts
    fr = ((h * 60 + m) * 60 + s) * nominal + f
    if drop and nominal in (30, 60):
        d = 2 if nominal == 30 else 4
        tm = h * 60 + m
        fr -= d * (tm - tm // 10)
    return fr


def _frames_tc(fr: int, nominal: int, drop: bool) -> str:
    from .media import frames_to_tc
    return frames_to_tc(int(fr), float(nominal), drop)


def norm_name(s: str) -> str:
    base = Path(unquote(s or "").replace("\\", "/")).name
    base = unicodedata.normalize("NFC", base)
    return base


def _flatten(items: List[Tuple[int, int, int, str, dict]]) -> List[Shot]:
    """[(rec_in, rec_out, track, kind, info)] on many tracks -> what is on screen, as consecutive shots.
    The highest track wins; a transition on the winning track covers both of its clips."""
    if not items:
        return []
    bounds = sorted({v for a, b, *_ in items for v in (a, b)})
    out: List[Shot] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if hi <= lo:
            continue
        cover = [it for it in items if it[0] <= lo and it[1] >= hi]
        if not cover:
            sh = Shot(lo, hi, "gap")
        else:
            top = max(it[2] for it in cover)
            on = [it for it in cover if it[2] == top]
            tr = next((it for it in on if it[3] == "transition"), None)
            it = tr or on[0]
            info = it[4]
            src_in = info.get("src_in", -1)
            if src_in is not None and src_in >= 0 and it[3] == "clip":
                src_in = src_in + int(round((lo - it[0]) * info.get("speed", 100.0) / 100.0))
            sh = Shot(lo, hi, it[3], info.get("name", ""), src_in if it[3] == "clip" else -1, top,
                      info.get("speed", 100.0), info.get("nested", False), info.get("note", ""))
            sh._key = info.get("key", id(it))
        if out and out[-1].kind == sh.kind and getattr(out[-1], "_key", None) == getattr(sh, "_key", None) \
                and out[-1].rec_out == sh.rec_in and sh.kind != "gap":
            out[-1].rec_out = sh.rec_out
        elif out and out[-1].kind == "gap" and sh.kind == "gap" and out[-1].rec_out == sh.rec_in:
            out[-1].rec_out = sh.rec_out
        else:
            out.append(sh)
    for s in out:
        if hasattr(s, "_key"):
            del s._key
    return out


def _pad(shots: List[Shot], duration: int) -> List[Shot]:
    """Black before the first and after the last clip (the baked export includes it)."""
    if not shots:
        return shots
    if shots[0].rec_in > 0:
        shots.insert(0, Shot(0, shots[0].rec_in, "gap"))
    if duration > shots[-1].rec_out:
        shots.append(Shot(shots[-1].rec_out, duration, "gap"))
    return shots


# ------------------------------------------------------------------ FCP7 / Premiere xmeml
def _speed_xmeml(ci) -> float:
    for eff in ci.iter("effect"):
        eid = (eff.findtext("effectid") or "").lower()
        if eid in ("timeremap", "speed") or "speed" in (eff.findtext("name") or "").lower():
            for par in eff.iter("parameter"):
                pid = (par.findtext("parameterid") or "").lower()
                if pid == "speed":
                    v = par.findtext("value")
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
    return 100.0


def _read_xmeml(root) -> Timeline:
    seq = root.find("sequence")
    if seq is None:
        seq = next(root.iter("sequence"), None)
    if seq is None:
        raise ValueError("XML에 시퀀스가 없습니다")
    tl = Timeline(fmt="xmeml", name=seq.findtext("name") or "")
    tl.fps, tl.fps_num, tl.fps_den = _rate(seq.find("rate"))
    nominal = int(round(tl.fps))
    tce = seq.find("timecode")
    if tce is not None:
        tl.drop = (tce.findtext("displayformat") or "").upper() == "DF"
        tl.start_tc = tce.findtext("string") or ""
        fr = _num(tce.findtext("frame"))
        tl.start_frames = fr if fr is not None else _tc_frames(tl.start_tc, nominal, tl.drop)
        if not tl.start_tc:
            tl.start_tc = _frames_tc(tl.start_frames, nominal, tl.drop)
    sc = seq.find("media/video/format/samplecharacteristics")
    if sc is not None:
        tl.width = _num(sc.findtext("width"), 0) or 0
        tl.height = _num(sc.findtext("height"), 0) or 0
    files: Dict[str, str] = {}
    for f in root.iter("file"):
        if f.get("id") and (f.findtext("name") or f.findtext("pathurl")):
            files[f.get("id")] = f.findtext("name") or f.findtext("pathurl")
    video = seq.find("media/video")
    items = []
    if video is None:
        raise ValueError("XML에 비디오 트랙이 없습니다")
    tracks = video.findall("track")
    for ti, track in enumerate(tracks, start=1):
        if (track.findtext("enabled") or "TRUE").upper() == "FALSE":
            continue
        children = [c for c in track if c.tag in ("clipitem", "transitionitem", "generatoritem")]
        for k, c in enumerate(children):
            a, b = _num(c.findtext("start"), -1), _num(c.findtext("end"), -1)
            if c.tag == "transitionitem":
                if a is not None and b is not None and b > a >= 0:
                    items.append((a, b, ti, "transition", dict(name=c.findtext("name") or "transition",
                                                               key=f"t{ti}.{k}")))
                continue
            if (c.findtext("enabled") or "TRUE").upper() == "FALSE":
                continue
            # next to a transition: the transition holds the boundary
            if a == -1 and k > 0 and children[k - 1].tag == "transitionitem":
                a = _num(children[k - 1].findtext("end"), -1)
            if b == -1 and k + 1 < len(children) and children[k + 1].tag == "transitionitem":
                b = _num(children[k + 1].findtext("start"), -1)
            if a is None or b is None or a < 0 or b <= a:
                continue
            f = c.find("file")
            nm = ""
            if f is not None:
                nm = files.get(f.get("id") or "", "") or f.findtext("name") or ""
            nested = c.find("sequence") is not None
            nm = nm or c.findtext("name") or ("nested" if nested else "")
            sp = _speed_xmeml(c)
            src_in = _num(c.findtext("in"), -1)
            note = []
            if abs(sp - 100.0) > 0.01:
                note.append(f"속도 {sp:.0f}%")
            if nested:
                note.append("중첩 시퀀스")
            if c.tag == "generatoritem":
                note.append("생성 클립")
            items.append((a, b, ti, "clip", dict(name=norm_name(nm), src_in=src_in, speed=sp, nested=nested,
                                                 note=" · ".join(note), key=f"c{ti}.{k}")))
    tl.shots = _pad(_flatten(items), _num(seq.findtext("duration"), 0) or 0)
    if len(tracks) > 1:
        tl.warnings.append(f"비디오 트랙 {len(tracks)}개: 위쪽 트랙의 클립을 기준으로 나눴습니다")
    return tl


# ------------------------------------------------------------------ FCPXML
def _rt(v: Optional[str]) -> Fraction:
    """'1001/24000s', '10s', '0s' -> seconds."""
    if not v:
        return Fraction(0)
    v = v.strip().rstrip("s")
    if "/" in v:
        n, d = v.split("/")
        return Fraction(int(n), int(d))
    try:
        return Fraction(v)
    except ValueError:
        return Fraction(0)


def _read_fcpxml(root) -> Timeline:
    tl = Timeline(fmt="fcpxml")
    formats = {f.get("id"): f for f in root.iter("format")}
    assets = {}
    for a in root.iter("asset"):
        src = a.get("src") or ""
        mr = a.find("media-rep")
        if mr is not None and mr.get("src"):
            src = mr.get("src")
        assets[a.get("id")] = norm_name(a.get("name") or src)
    seq = next(root.iter("sequence"), None)
    if seq is None:
        raise ValueError("FCPXML에 시퀀스가 없습니다")
    proj = next(root.iter("project"), None)
    tl.name = (proj.get("name") if proj is not None else "") or ""
    fmt = formats.get(seq.get("format"))
    fd = _rt(fmt.get("frameDuration")) if fmt is not None and fmt.get("frameDuration") else Fraction(1, 24)
    if fd <= 0:
        fd = Fraction(1, 24)
    fps = 1 / fd
    tl.fps = float(fps)
    tl.fps_num, tl.fps_den = fps.numerator, fps.denominator
    if fmt is not None:
        tl.width = _num(fmt.get("width"), 0) or 0
        tl.height = _num(fmt.get("height"), 0) or 0
    tl.drop = (seq.get("tcFormat") or "").upper() == "DF"
    t0 = _rt(seq.get("tcStart"))
    tl.start_frames = int(round(t0 * fps))
    tl.start_tc = _frames_tc(tl.start_frames, int(round(tl.fps)), tl.drop)
    spine = seq.find("spine")
    if spine is None:
        raise ValueError("FCPXML에 spine이 없습니다")
    items = []
    clip_tags = ("asset-clip", "clip", "ref-clip", "mc-clip", "sync-clip", "video", "title")

    def fr(t: Fraction) -> int:
        return int(round(t * fps))

    def name_of(el):
        if el.get("ref") in assets:
            return assets[el.get("ref")]
        v = el.find("video")
        if v is not None and v.get("ref") in assets:
            return assets[v.get("ref")]
        a = el.find("asset-clip")
        if a is not None and a.get("ref") in assets:
            return assets[a.get("ref")]
        return el.get("name") or el.tag

    def speed_of(el) -> float:
        tm = el.find("timeMap")
        if tm is not None:
            pts = tm.findall("timept")
            if len(pts) >= 2:
                d_t = _rt(pts[-1].get("time")) - _rt(pts[0].get("time"))
                d_v = _rt(pts[-1].get("value")) - _rt(pts[0].get("value"))
                if d_t > 0:
                    return float(d_v / d_t * 100)
        return 100.0

    def walk(container, base: Fraction, cstart: Fraction, lane: int, key: str):
        """Items of a spine (lane 0) or connected to a clip: absolute time = base + offset - cstart."""
        for k, el in enumerate(container):
            if el.tag not in clip_tags + ("gap", "transition", "spine"):
                continue
            off = _rt(el.get("offset"))
            dur = _rt(el.get("duration"))
            ln = _num(el.get("lane"), 0) or 0
            a = base + off - cstart
            b = a + dur
            kk = f"{key}.{k}"
            if el.tag == "spine":
                walk(el, a, Fraction(0), lane + ln, kk)
                continue
            if el.tag == "transition":
                items.append((fr(a), fr(b), lane + ln + 1, "transition", dict(name=el.get("name") or "transition",
                                                                             key=kk)))
                continue
            st = _rt(el.get("start"))
            if el.tag != "gap" and (el.get("enabled") or "1") != "0":
                sp = speed_of(el)
                nested = el.tag in ("ref-clip", "sync-clip", "mc-clip")
                note = []
                if abs(sp - 100) > 0.01:
                    note.append(f"속도 {sp:.0f}%")
                if nested:
                    note.append("복합/멀티캠 클립")
                items.append((fr(a), fr(b), lane + ln + 1, "clip", dict(name=name_of(el), src_in=fr(st), speed=sp,
                                                                       nested=nested, note=" · ".join(note), key=kk)))
            # connected clips (lane != 0) are positioned on this clip's own time
            for ch in el:
                if ch.tag in clip_tags + ("gap", "transition", "spine") and ch.get("lane") is not None:
                    walk([ch], a, st, lane + ln, kk)

    walk(spine, Fraction(0), Fraction(0), 0, "s")
    end = Fraction(0)
    for el in spine:
        if el.get("offset") is not None:
            end = max(end, _rt(el.get("offset")) + _rt(el.get("duration")))
    dur = _rt(seq.get("duration")) or end
    tl.shots = _pad(_flatten(items), fr(dur))
    if any(it[2] > 1 and it[3] == "clip" for it in items):
        tl.warnings.append("연결된 클립(위 레인)이 있어 위쪽 클립을 기준으로 나눴습니다")
    return tl


# ------------------------------------------------------------------ CMX3600 EDL
_EV = re.compile(r"^\s*(\d{3,6})\s+(\S+)\s+(\S+)\s+(C|D|W\d*|K\s*B?|K)\s*(\d+)?\s+"
                 r"(\d\d[:;.]\d\d[:;.]\d\d[:;.]\d\d)\s+(\d\d[:;.]\d\d[:;.]\d\d[:;.]\d\d)\s+"
                 r"(\d\d[:;.]\d\d[:;.]\d\d[:;.]\d\d)\s+(\d\d[:;.]\d\d[:;.]\d\d[:;.]\d\d)")


def _read_edl(text: str, fps: float) -> Timeline:
    tl = Timeline(fmt="edl", fps=fps)
    if abs(fps - round(fps)) > 0.01:
        tl.fps_num, tl.fps_den = int(round(fps)) * 1000, 1001
    else:
        tl.fps_num, tl.fps_den = int(round(fps)), 1
    nominal = int(round(fps))
    drop = False
    items = []
    last_name = ""
    events = []
    for line in text.splitlines():
        u = line.strip()
        if u.upper().startswith("TITLE:"):
            tl.name = u[6:].strip()
            continue
        if u.upper().startswith("FCM:"):
            drop = "NON" not in u.upper()
            continue
        m = _EV.match(line)
        if m:
            reel, chan, trans, tdur = m.group(2), m.group(3), m.group(4).replace(" ", ""), m.group(5)
            if "V" not in chan.upper():
                continue
            ev = dict(reel=reel, trans=trans, tdur=_num(tdur, 0) or 0,
                      s_in=_tc_frames(m.group(6), nominal, drop), r_in=_tc_frames(m.group(8), nominal, drop),
                      r_out=_tc_frames(m.group(9), nominal, drop), name=reel, speed=100.0)
            events.append(ev)
            continue
        if events:
            lu = u.upper()
            if lu.startswith("* FROM CLIP NAME:") or lu.startswith("*FROM CLIP NAME:"):
                events[-1]["name"] = u.split(":", 1)[1].strip()
            elif lu.startswith("* SOURCE FILE:") or lu.startswith("*SOURCE FILE:"):
                events[-1]["name"] = norm_name(u.split(":", 1)[1].strip())
            elif lu.startswith("M2"):
                p = u.split()
                try:
                    sp = float(p[2])
                    events[-1]["speed"] = sp / max(1e-6, tl.fps) * 100.0
                except (IndexError, ValueError):
                    pass
    tl.drop = drop
    if not events:
        raise ValueError("EDL에서 비디오 이벤트를 찾지 못했습니다")
    t0 = min(e["r_in"] for e in events)
    tl.start_frames = t0
    tl.start_tc = _frames_tc(t0, nominal, drop)
    for k, e in enumerate(events):
        a, b = e["r_in"] - t0, e["r_out"] - t0
        if b <= a:
            continue
        if e["trans"].startswith("D") and e["tdur"] > 0:
            # the dissolve runs into this event from the previous one: its first tdur frames are mixed
            d = min(e["tdur"], b - a)
            items.append((a, a + d, 1, "transition", dict(name=f"디졸브 {d}f", key=f"t{k}")))
            if d < b - a:
                items.append((a + d, b, 1, "clip", dict(name=norm_name(e["name"]), src_in=e["s_in"] + d,
                                                        speed=e["speed"], key=f"c{k}",
                                                        note=f"속도 {e['speed']:.0f}%" if abs(e["speed"] - 100) > 0.5 else "")))
            continue
        items.append((a, b, 1, "clip", dict(name=norm_name(e["name"]), src_in=e["s_in"], speed=e["speed"], key=f"c{k}",
                                            note=f"속도 {e['speed']:.0f}%" if abs(e["speed"] - 100) > 0.5 else "")))
    # a dissolve's record range overlaps the previous event: that one ends where the mix starts
    items.sort(key=lambda it: (it[0], 0 if it[3] == "transition" else 1))
    fixed = []
    for it in items:
        if fixed and fixed[-1][3] == "clip" and fixed[-1][1] > it[0]:
            p = fixed[-1]
            fixed[-1] = (p[0], it[0], p[2], p[3], p[4])
        fixed.append(it)
    tl.shots = _flatten([it for it in fixed if it[1] > it[0]])
    return tl


# ------------------------------------------------------------------ entry
def read_timeline(path: str, fps_hint: float = 24.0) -> Timeline:
    p = Path(path)
    raw = p.read_bytes()
    head = raw[:4000].decode("utf-8", "ignore").lower()
    if p.suffix.lower() == ".edl" or ("title:" in head and "<" not in head[:200]):
        text = raw.decode("utf-8", "ignore")
        if "�" in text:
            text = raw.decode("cp949", "ignore")
        tl = _read_edl(text, fps_hint)
    else:
        root = ET.fromstring(raw)
        if root.tag == "fcpxml":
            tl = _read_fcpxml(root)
        elif root.tag == "xmeml" or root.find(".//sequence") is not None:
            tl = _read_xmeml(root)
        else:
            raise ValueError("지원하지 않는 편집 파일입니다 (FCP7/Premiere XML, FCPXML, EDL)")
    for s in tl.shots:
        if abs(s.speed - 100) > 0.5 and "속도" not in s.note:
            s.note = (s.note + " · " if s.note else "") + f"속도 {s.speed:.0f}%"
    return tl


# ------------------------------------------------------------------ checking against the baked video
def picture_change(info, progress=None, cancelled=None, size=(160, 90)) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Per frame: how much the picture changed from the previous frame (0..1), and small RGB thumbs."""
    from .media import FrameReader
    n = int(info.frames)
    diff = np.zeros(n, np.float32)
    thumbs: List[np.ndarray] = []
    prev = None
    with FrameReader(info, 0, n, "rgb24", size=size, rgb_convert=True, prefetch=8) as rd:
        for t in range(n):
            pl = rd.read()
            if pl is None:
                diff = diff[:t]
                break
            x = pl[0].astype(np.float32)
            if prev is not None:
                # colour histogram distance + mean abs difference (robust to flashes a bit)
                d1 = float(np.abs(x - prev).mean() / 255.0)
                h0 = np.histogram(prev[..., 1], 16, (0, 256))[0] / prev[..., 1].size
                h1 = np.histogram(x[..., 1], 16, (0, 256))[0] / x[..., 1].size
                diff[t] = 0.5 * d1 + 0.5 * float(np.abs(h0 - h1).sum() / 2)
            prev = x
            thumbs.append(pl[0].copy())
            if progress and t % 50 == 0:
                progress(t / max(1, n), f"편집 경계 확인 중 {t}/{n}")
            if cancelled and cancelled():
                break
    return diff, thumbs


def align(tl: Timeline, diff: np.ndarray, baked_tc_frames: Optional[int] = None) -> dict:
    """Offset of the baked video against the sequence (frames to add to sequence frames), and boundaries
    that do not show a picture change (match cuts or a wrong XML) / strong changes not in the XML."""
    n = len(diff)
    cuts = [s.rec_in for s in tl.shots[1:] if s.kind == "clip"]
    res = dict(offset=0, method="", matched=0, total=len(cuts), unmatched=[], extra=[])
    if n < 3 or not cuts:
        return res
    loc = np.zeros(n, bool)
    if n > 4:
        med = np.median(diff[1:]) + 1e-4
        mad = np.median(np.abs(diff[1:] - med)) + 1e-4
        thr = med + 6 * mad
        for t in range(1, n):
            lo, hi = max(1, t - 3), min(n, t + 4)
            if diff[t] >= thr and diff[t] >= diff[lo:hi].max() and diff[t] > 0.04:
                loc[t] = True
    peaks = np.nonzero(loc)[0]
    cand = {0}
    if baked_tc_frames is not None and tl.start_frames:
        cand.add(int(tl.start_frames - baked_tc_frames))
    # offsets that line up many boundaries with picture changes
    if len(peaks):
        votes: Dict[int, int] = {}
        for c in cuts:
            for p in peaks:
                d = int(p - c)
                if abs(d) <= 240:
                    votes[d] = votes.get(d, 0) + 1
        if votes:
            best = max(votes.items(), key=lambda kv: (kv[1], -abs(kv[0])))
            cand.add(best[0])

    def score(off):
        return sum(1 for c in cuts if 0 < c + off < n and loc[max(0, c + off - 1):c + off + 2].any())

    off = max(sorted(cand, key=abs), key=score)
    res["offset"] = int(off)
    res["matched"] = score(off)
    res["method"] = "타임코드" if baked_tc_frames is not None and off == tl.start_frames - (baked_tc_frames or 0) \
        else ("영상 비교" if off else "그대로")
    res["unmatched"] = [int(c) for c in cuts if 0 < c + off < n and not loc[max(0, c + off - 1):c + off + 2].any()]
    xs = set()
    for c in {v for sh in tl.shots for v in (sh.rec_in, sh.rec_out)}:
        xs.update(range(c + off - 2, c + off + 3))
    res["extra"] = [int(p) for p in peaks if p not in xs and diff[p] > 0.12]
    res["peaks"] = [int(p) for p in peaks]
    return res
