"""Media I/O through ffmpeg.

The render path never converts colour: frames are decoded as native YUV planes
(or float RGB planes for image sequences), warped plane by plane and handed back
to ffmpeg in the same layout. Untouched pixels therefore keep their exact code
values, including super-whites and below-black.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

SEQ_EXTS = {".exr", ".dpx", ".tif", ".tiff", ".png"}
VIDEO_EXTS = {".mov", ".mp4", ".mxf", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".m2ts"}


# --------------------------------------------------------------------------- ffmpeg discovery

_tool_paths = {"ffmpeg": None, "ffprobe": None}


def configure_ffmpeg(path_hint: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Locate ffmpeg/ffprobe: the copy that ships with DIAP ({app}\ffmpeg\bin), never one found elsewhere on the
    PC. Only a development checkout (no installed python next to it) falls back to the one on PATH.
    `path_hint` is ignored (kept for old callers)."""
    exe = ".exe" if sys.platform == "win32" else ""
    here = Path(__file__).resolve().parents[2]
    # installed build: {app}\app is the code, {app}\ffmpeg\bin the ffmpeg packed into the installer
    dirs: List[str] = [str(here.parent / "ffmpeg" / "bin"), str(here / "ffmpeg" / "bin")]
    installed = (here.parent / "python").is_dir()
    found = {}
    for tool in ("ffmpeg", "ffprobe"):
        hit = None
        for d in dirs:
            cand = Path(d) / f"{tool}{exe}"
            if cand.is_file():
                hit = str(cand)
                break
        if hit is None and not installed:
            hit = shutil.which(tool)
        found[tool] = hit
    _tool_paths.update(found)
    return found["ffmpeg"], found["ffprobe"]


def ffmpeg_bin() -> str:
    if not _tool_paths["ffmpeg"]:
        configure_ffmpeg()
    if not _tool_paths["ffmpeg"]:
        raise MediaError("ffmpeg를 찾을 수 없습니다. DIAP을 다시 설치하세요 (ffmpeg는 설치 파일에 들어 있습니다).")
    return _tool_paths["ffmpeg"]


def ffprobe_bin() -> str:
    if not _tool_paths["ffprobe"]:
        configure_ffmpeg()
    if not _tool_paths["ffprobe"]:
        raise MediaError("ffprobe를 찾을 수 없습니다. DIAP을 다시 설치하세요 (ffmpeg는 설치 파일에 들어 있습니다).")
    return _tool_paths["ffprobe"]


class MediaError(RuntimeError):
    pass


def _run(cmd: Sequence[str], timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=_CREATE_NO_WINDOW)


# --------------------------------------------------------------------------- timecode

def tc_to_frames(tc: str, fps: float) -> int:
    drop = ";" in tc or "." in tc[-3:]
    parts = [int(p) for p in re.split(r"[:;.]", tc)]
    if len(parts) != 4:
        return 0
    h, m, s, f = parts
    nominal = int(round(fps))
    frames = ((h * 60 + m) * 60 + s) * nominal + f
    if drop and nominal in (30, 60):
        d = 2 if nominal == 30 else 4
        total_minutes = h * 60 + m
        frames -= d * (total_minutes - total_minutes // 10)
    return frames


def frames_to_tc(frames: int, fps: float, drop: bool = False) -> str:
    nominal = int(round(fps)) or 1
    if drop and nominal in (30, 60):
        d = 2 if nominal == 30 else 4
        per10 = nominal * 600 - d * 9
        per1 = nominal * 60 - d
        tens, rem = divmod(frames, per10)
        if rem > d:
            frames += d * 9 * tens + d * ((rem - d) // per1)
        else:
            frames += d * 9 * tens
        sep = ";"
    else:
        sep = ":"
    f = frames % nominal
    s = (frames // nominal) % 60
    m = (frames // (nominal * 60)) % 60
    h = (frames // (nominal * 3600)) % 24
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{f:02d}"


# --------------------------------------------------------------------------- probing

@dataclass
class MediaInfo:
    path: str
    kind: str = "video"              # video | sequence
    width: int = 0
    height: int = 0
    fps_num: int = 24
    fps_den: int = 1
    frames: int = 0
    seq_pattern: str = ""            # printf pattern for sequences
    seq_start: int = 0               # first frame number of a sequence
    codec: str = ""
    pix_fmt: str = ""
    bit_depth: int = 8
    chroma: str = "420"              # 420 | 422 | 444 | rgb
    has_alpha: bool = False
    is_float: bool = False
    color_range: str = "tv"
    color_matrix: str = "bt709"
    color_trc: str = ""
    color_primaries: str = ""
    timecode: str = ""
    has_audio: bool = False
    duration: float = 0.0
    offset: int = 0                  # a stretch of a longer file (a cut of a baked timeline): its first frame
    meta_text: str = ""              # camera / encoder tags of the file (for telling its colour space)

    @property
    def fps(self) -> float:
        return self.fps_num / max(1, self.fps_den)

    @property
    def fps_fraction(self) -> str:
        return f"{self.fps_num}/{self.fps_den}"

    @property
    def drop_frame(self) -> bool:
        return ";" in self.timecode

    def start_tc_frames(self) -> int:
        return (tc_to_frames(self.timecode, self.fps) if self.timecode else 0) + int(self.offset or 0)

    def tc_at(self, frame: int) -> str:
        return frames_to_tc(self.start_tc_frames() + frame, self.fps, self.drop_frame)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MediaInfo":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


_SEQ_RE = re.compile(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$")


def detect_sequence(path: str) -> Optional[Tuple[str, int, int]]:
    """Return (printf_pattern, first_frame, count) if `path` belongs to an image sequence."""
    p = Path(path)
    if p.suffix.lower() not in SEQ_EXTS:
        return None
    m = _SEQ_RE.match(p.name)
    if not m:
        return None
    prefix, digits, ext = m.groups()
    pad = len(digits)
    numbers = []
    rx = re.compile("^" + re.escape(prefix) + r"(\d+)" + re.escape(ext) + "$")
    for f in p.parent.iterdir():
        mm = rx.match(f.name)
        if mm:
            numbers.append(int(mm.group(1)))
    if len(numbers) < 2:
        return None
    numbers.sort()
    first = numbers[0]
    # contiguous run starting at first
    count = 1
    for a, b in zip(numbers, numbers[1:]):
        if b != a + 1:
            break
        count += 1
    pattern = str(p.parent / f"{prefix}%0{pad}d{ext}")
    return pattern, first, count


_MATRIX_MAP = {
    "bt709": "bt709", "bt2020nc": "bt2020", "bt2020c": "bt2020", "bt2020": "bt2020",
    "smpte170m": "bt601", "bt470bg": "bt601", "bt601": "bt601", "fcc": "fcc", "smpte240m": "smpte240m",
}


def swscale_matrix(info: MediaInfo) -> str:
    m = _MATRIX_MAP.get(info.color_matrix or "", "")
    if m:
        return m
    return "bt709" if info.height >= 700 else "bt601"


def _pixfmt_props(pix_fmt: str) -> Tuple[int, str, bool, bool]:
    """bit depth, chroma, alpha, float"""
    pf = pix_fmt or ""
    depth = 8
    m = re.search(r"p(\d+)(le|be)?$", pf) or re.search(r"(\d+)(le|be)$", pf)
    if m:
        try:
            depth = int(m.group(1))
        except ValueError:
            depth = 8
    if "f32" in pf or "f16" in pf:
        isfloat = True
        depth = 32 if "f32" in pf else 16
    else:
        isfloat = False
    if pf.startswith(("rgb", "bgr", "gbr", "argb", "abgr", "rgba", "bgra", "gbra")):
        chroma = "rgb"
        if pf in ("rgb48le", "rgb48be", "rgba64le", "rgba64be", "bgr48le"):
            depth = 16
    elif "444" in pf:
        chroma = "444"
    elif "422" in pf:
        chroma = "422"
    elif "420" in pf or "411" in pf:
        chroma = "420"
    elif pf.startswith("gray"):
        chroma = "gray"
    else:
        chroma = "420"
    alpha = pf.startswith(("yuva", "rgba", "bgra", "gbra", "argb", "abgr", "ya"))
    return depth, chroma, alpha, isfloat


def probe(path: str) -> MediaInfo:
    path = str(path)
    if not os.path.exists(path):
        raise MediaError(f"파일이 없습니다: {path}")
    seq = detect_sequence(path)
    info = MediaInfo(path=path)
    probe_target = path
    if seq:
        pattern, first, count = seq
        info.kind = "sequence"
        info.seq_pattern = pattern
        info.seq_start = first
        info.frames = count
        probe_target = pattern % first
    cmd = [ffprobe_bin(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", probe_target]
    r = _run(cmd)
    if r.returncode != 0:
        raise MediaError(f"ffprobe 실패: {r.stderr.decode(errors='replace')[:400]}")
    data = json.loads(r.stdout.decode("utf-8", errors="replace") or "{}")
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise MediaError("영상 스트림이 없습니다.")
    info.has_audio = any(s.get("codec_type") == "audio" for s in streams)
    info.width = int(v.get("width", 0))
    info.height = int(v.get("height", 0))
    info.codec = v.get("codec_name", "")
    info.pix_fmt = v.get("pix_fmt", "")
    info.bit_depth, info.chroma, info.has_alpha, info.is_float = _pixfmt_props(info.pix_fmt)
    if v.get("bits_per_raw_sample", "").isdigit():
        info.bit_depth = max(info.bit_depth, int(v["bits_per_raw_sample"]))
    rate = v.get("r_frame_rate") or v.get("avg_frame_rate") or "24/1"
    try:
        fr = Fraction(rate)
        if fr <= 0 or fr > 1000:
            fr = Fraction(v.get("avg_frame_rate", "24/1"))
    except (ZeroDivisionError, ValueError):
        fr = Fraction(24, 1)
    if fr <= 0:
        fr = Fraction(24, 1)
    info.fps_num, info.fps_den = fr.numerator, fr.denominator
    info.color_range = v.get("color_range") or ("pc" if info.chroma == "rgb" or info.codec in ("mjpeg",) else "tv")
    if info.color_range not in ("tv", "pc"):
        info.color_range = "tv"
    info.color_matrix = v.get("color_space") or ""
    info.color_trc = v.get("color_transfer") or ""
    info.color_primaries = v.get("color_primaries") or ""
    tags = dict(data.get("format", {}).get("tags", {}) or {})
    tags.update(v.get("tags", {}) or {})
    for s in streams:
        if s.get("codec_tag_string") == "tmcd" or s.get("codec_type") == "data":
            tags.setdefault("timecode", (s.get("tags") or {}).get("timecode", ""))
    info.timecode = tags.get("timecode", "") or ""
    keep = [f"{k}={v}" for k, v in tags.items() if isinstance(v, str) and any(
        w in k.lower() for w in ("make", "model", "camera", "encoder", "com.apple", "com.sony", "com.arri", "comment",
                                 "handler", "manufacturer", "product"))]
    info.meta_text = " ".join(keep)[:400]
    try:
        info.duration = float(data.get("format", {}).get("duration") or v.get("duration") or 0)
    except ValueError:
        info.duration = 0.0
    if info.kind == "video":
        nb = v.get("nb_frames")
        if nb and str(nb).isdigit() and int(nb) > 0:
            info.frames = int(nb)
        else:
            info.frames = count_frames(path)
            if info.frames <= 0 and info.duration > 0:
                info.frames = int(round(info.duration * info.fps))
    if info.kind == "sequence":
        info.duration = info.frames / info.fps
        if info.is_float:
            info.color_range = "pc"
    return info


def count_frames(path: str) -> int:
    cmd = [ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-count_packets",
           "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path]
    r = _run(cmd, timeout=600)
    try:
        return int(r.stdout.decode().strip().split(",")[0])
    except ValueError:
        return 0


# --------------------------------------------------------------------------- plane layouts

@dataclass
class PlaneLayout:
    pix_fmt: str
    shapes: List[Tuple[int, int]]   # (h, w) per plane
    subsample: List[Tuple[int, int]]  # (sx, sy) per plane
    dtype: np.dtype
    interleaved: int = 0            # >0: one interleaved plane with N channels
    max_value: float = 1.0
    is_float: bool = False

    @property
    def frame_bytes(self) -> int:
        n = 0
        for (h, w) in self.shapes:
            n += h * w * max(1, self.interleaved)
        return n * np.dtype(self.dtype).itemsize


def plane_layout(pix_fmt: str, width: int, height: int) -> PlaneLayout:
    w, h = width, height
    cw2, ch2 = (w + 1) // 2, (h + 1) // 2
    u16 = np.dtype("<u2")
    if pix_fmt in ("yuv422p10le", "yuv422p12le", "yuv422p16le"):
        bits = int(pix_fmt[7:9])
        return PlaneLayout(pix_fmt, [(h, w), (h, cw2), (h, cw2)], [(1, 1), (2, 1), (2, 1)], u16, 0, float((1 << bits) - 1))
    if pix_fmt in ("yuv444p10le", "yuv444p12le", "yuv444p16le"):
        bits = int(pix_fmt[7:9])
        return PlaneLayout(pix_fmt, [(h, w)] * 3, [(1, 1)] * 3, u16, 0, float((1 << bits) - 1))
    if pix_fmt in ("yuva444p10le", "yuva444p12le", "yuva444p16le"):
        bits = int(pix_fmt[8:10])
        return PlaneLayout(pix_fmt, [(h, w)] * 4, [(1, 1)] * 4, u16, 0, float((1 << bits) - 1))
    if pix_fmt in ("yuv420p10le", "yuv420p12le", "yuv420p16le"):
        bits = int(pix_fmt[7:9])
        return PlaneLayout(pix_fmt, [(h, w), (ch2, cw2), (ch2, cw2)], [(1, 1), (2, 2), (2, 2)], u16, 0, float((1 << bits) - 1))
    if pix_fmt == "gbrpf32le":
        return PlaneLayout(pix_fmt, [(h, w)] * 3, [(1, 1)] * 3, np.dtype("<f4"), 0, 1.0, True)
    if pix_fmt == "rgb48le":
        return PlaneLayout(pix_fmt, [(h, w)], [(1, 1)], u16, 3, 65535.0)
    if pix_fmt == "rgb24":
        return PlaneLayout(pix_fmt, [(h, w)], [(1, 1)], np.dtype("u1"), 3, 255.0)
    if pix_fmt == "gray16le":
        return PlaneLayout(pix_fmt, [(h, w)], [(1, 1)], u16, 0, 65535.0)
    if pix_fmt == "gray":
        return PlaneLayout(pix_fmt, [(h, w)], [(1, 1)], np.dtype("u1"), 0, 255.0)
    raise MediaError(f"지원하지 않는 픽셀 포맷: {pix_fmt}")


def split_planes(buf: bytes, layout: PlaneLayout) -> List[np.ndarray]:
    arr = np.frombuffer(buf, dtype=layout.dtype)
    planes = []
    off = 0
    for (h, w) in layout.shapes:
        if layout.interleaved:
            n = h * w * layout.interleaved
            planes.append(arr[off:off + n].reshape(h, w, layout.interleaved))
        else:
            n = h * w
            planes.append(arr[off:off + n].reshape(h, w))
        off += n
    return planes


def join_planes(planes: Sequence[np.ndarray], layout: PlaneLayout) -> bytes:
    return b"".join(np.ascontiguousarray(p, dtype=layout.dtype).tobytes() for p in planes)


# --------------------------------------------------------------------------- reading

_VSTART: dict = {}


def video_start(path: str) -> float:
    """Seconds by which the first video frame comes after the file's start. H.264 in MP4 with B-frames often starts
    its picture a frame late (lngshot v3: video 0.041 s, audio and the file 0): ffmpeg's -ss counts from the file's
    start, so a seek to a frame without this landed one frame early, and every cut of the bake began on the last
    frame of the shot before."""
    key = str(path)
    if key in _VSTART:
        return _VSTART[key]
    v = 0.0
    try:
        r = _run([ffprobe_bin(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time:format=start_time",
                  "-of", "json", key])
        d = json.loads(r.stdout.decode("utf-8", errors="replace") or "{}")
        vs = float((d.get("streams") or [{}])[0].get("start_time") or 0.0)
        fs = float((d.get("format") or {}).get("start_time") or 0.0)
        v = vs - fs
        if not np.isfinite(v) or abs(v) > 5.0:
            v = 0.0
    except Exception:
        v = 0.0
    _VSTART[key] = v
    return v


def _input_args(info: MediaInfo, start: int) -> List[str]:
    start = int(start) + int(getattr(info, "offset", 0) or 0)
    if info.kind == "sequence":
        return ["-f", "image2", "-framerate", info.fps_fraction,
                "-start_number", str(info.seq_start + start), "-i", info.seq_pattern]
    args = []
    if start > 0:
        # Seek half a frame early; `-frames:v` + accurate seek keeps the first frame exact.
        t = video_start(info.path) + (start - 0.5) * info.fps_den / info.fps_num
        args += ["-ss", f"{max(0.0, t):.6f}"]
    return args + ["-i", info.path]


class FrameReader:
    """Sequential frame reader with a prefetch thread."""

    def __init__(self, info: MediaInfo, start: int, count: int, pix_fmt: str,
                 size: Optional[Tuple[int, int]] = None, rgb_convert: bool = False,
                 prefetch: int = 3, crop: Optional[Tuple[int, int, int, int]] = None):
        """crop: (x, y, w, h) in source pixels, cut out before anything else (size then is the size of the cut-out
        piece as it is handed out; None: its own size)."""
        self.info = info
        self.count = max(0, int(count))
        src_w, src_h = (int(crop[2]), int(crop[3])) if crop else (info.width, info.height)
        w, h = size if size else (src_w, src_h)
        self.layout = plane_layout(pix_fmt, w, h)
        vf = []
        if crop:
            vf.append(f"crop={int(crop[2])}:{int(crop[3])}:{int(crop[0])}:{int(crop[1])}:exact=1")
        scale_opts = []
        if size and (w, h) != (src_w, src_h):
            scale_opts += [f"w={w}", f"h={h}"]
        target_rgb = pix_fmt.startswith(("rgb", "gbr"))
        if target_rgb and info.chroma not in ("rgb", "gray"):
            scale_opts += [f"in_color_matrix={swscale_matrix(info)}", f"in_range={info.color_range}", "out_range=pc"]
        if scale_opts:
            # "area" only where the picture is made smaller: with no resize, ffmpeg 9's swscale converts 8-bit
            # 4:2:0 to rgb48 with its colours wrong under "area" (green near nothing: a 1920x1080 H.264 source read
            # at its own size, lngshot v3); "bicubic" there does nothing but pick the right converter
            resize = bool(size) and (w, h) != (src_w, src_h)
            flags = "area" if resize else "bicubic"
            vf.append("scale=" + ":".join(scale_opts) + f":flags={flags}+accurate_rnd+full_chroma_int+full_chroma_inp")
        cmd = [ffmpeg_bin(), "-hide_banner", "-v", "error", "-nostdin"]
        cmd += _input_args(info, start)
        cmd += ["-map", "0:v:0", "-frames:v", str(self.count), "-an", "-sn", "-dn"]
        if vf:
            cmd += ["-vf", ",".join(vf)]
        cmd += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
        self.cmd = cmd
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      stdin=subprocess.DEVNULL, bufsize=self.layout.frame_bytes * 2,
                                      creationflags=_CREATE_NO_WINDOW)
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, prefetch))
        self._err = b""
        self._stop = False
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()
        self._errt = threading.Thread(target=self._drain_err, daemon=True)
        self._errt.start()
        self.read_count = 0

    def _drain_err(self):
        try:
            self._err = self._proc.stderr.read() or b""
        except Exception:
            pass

    def _pump(self):
        nb = self.layout.frame_bytes
        out = self._proc.stdout
        try:
            for _ in range(self.count):
                if self._stop:
                    break
                chunks = []
                got = 0
                while got < nb:
                    c = out.read(nb - got)
                    if not c:
                        break
                    chunks.append(c)
                    got += len(c)
                if got < nb:
                    break
                self._q.put(b"".join(chunks))
        finally:
            self._q.put(None)

    def read(self) -> Optional[List[np.ndarray]]:
        buf = self._q.get()
        if buf is None:
            self._q.put(None)
            return None
        self.read_count += 1
        return split_planes(buf, self.layout)

    def __iter__(self) -> Iterator[List[np.ndarray]]:
        while True:
            p = self.read()
            if p is None:
                return
            yield p

    @property
    def error_text(self) -> str:
        return self._err.decode(errors="replace")

    def close(self):
        self._stop = True
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass
        try:
            self._proc.kill()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def read_frame_rgb(info: MediaInfo, frame: int, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Random access: one frame as float32 RGB (0..1 code values, full range)."""
    pf = "gbrpf32le" if info.is_float else "rgb48le"
    with FrameReader(info, frame, 1, pf, size=size, prefetch=1) as r:
        planes = r.read()
        if planes is None:
            raise MediaError(f"프레임 {frame}을(를) 읽지 못했습니다. {r.error_text[:300]}")
    return planes_to_rgb_float(planes, r.layout)


def planes_to_rgb_float(planes: List[np.ndarray], layout: PlaneLayout) -> np.ndarray:
    if layout.pix_fmt == "gbrpf32le":
        g, b, r = planes
        return np.dstack([r, g, b]).astype(np.float32)
    if layout.interleaved == 3:
        return planes[0].astype(np.float32) / layout.max_value
    raise MediaError("RGB 변환 불가 레이아웃")


# --------------------------------------------------------------------------- YUV <-> RGB (numpy, exact matrices)

_KRKB = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114), "bt2020": (0.2627, 0.0593),
         "smpte240m": (0.212, 0.087), "fcc": (0.30, 0.11)}


def yuv_to_rgb(planes: List[np.ndarray], layout: PlaneLayout, matrix: str, rng: str,
               out_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert native YUV planes to float32 RGB (0..1). Chroma is upsampled bilinearly."""
    import cv2
    y, u, v = (p.astype(np.float32) for p in planes[:3])
    h, w = y.shape
    if u.shape != y.shape:
        u = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
        v = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)
    maxv = layout.max_value
    bits_scale = (maxv + 1) / 256.0
    if rng == "tv":
        yn = (y - 16 * bits_scale) / (219 * bits_scale)
        un = (u - 128 * bits_scale) / (224 * bits_scale)
        vn = (v - 128 * bits_scale) / (224 * bits_scale)
    else:
        yn = y / maxv
        un = (u - 128 * bits_scale) / maxv
        vn = (v - 128 * bits_scale) / maxv
    kr, kb = _KRKB.get(matrix, _KRKB["bt709"])
    kg = 1 - kr - kb
    r = yn + 2 * (1 - kr) * vn
    b = yn + 2 * (1 - kb) * un
    g = (yn - kr * r - kb * b) / kg
    rgb = np.dstack([r, g, b])
    if out_size and (w, h) != tuple(out_size):
        rgb = cv2.resize(rgb, tuple(out_size), interpolation=cv2.INTER_AREA)
    return rgb


# --------------------------------------------------------------------------- writing

PRORES_PROFILES = {
    "prores_422_proxy": (0, "yuv422p10le", "ProRes 422 Proxy"),
    "prores_422_lt": (1, "yuv422p10le", "ProRes 422 LT"),
    "prores_422": (2, "yuv422p10le", "ProRes 422"),
    "prores_422_hq": (3, "yuv422p10le", "ProRes 422 HQ"),
    "prores_4444": (4, "yuv444p10le", "ProRes 4444"),
    "prores_4444_xq": (5, "yuv444p10le", "ProRes 4444 XQ"),
}
VIDEO_PROFILES = dict(PRORES_PROFILES, hevc_mp4=(-1, "yuv420p10le", "H.265 MP4 10bit"))
_NVENC: dict = {}


def hevc_encoder(prefer: str = "auto") -> str:
    """hevc_nvenc when the GPU encoder works here (checked once), else libx265."""
    if prefer == "x265":
        return "libx265"
    if "ok" not in _NVENC:
        try:
            r = subprocess.run([ffmpeg_bin(), "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                                "color=black:s=256x256:r=24:d=0.1", "-c:v", "hevc_nvenc", "-pix_fmt", "p010le",
                                "-f", "null", "-"], capture_output=True, timeout=30, creationflags=_CREATE_NO_WINDOW)
            _NVENC["ok"] = r.returncode == 0
        except Exception:
            _NVENC["ok"] = False
    if prefer == "nvenc" and not _NVENC["ok"]:
        raise MediaError("이 PC에서 NVIDIA 하드웨어 인코더(NVENC)를 쓸 수 없습니다")
    return "hevc_nvenc" if _NVENC["ok"] else "libx265"


def hevc_args(encoder: str, quality: str = "high") -> List[str]:
    """10-bit 4:2:0 H.265 at a quality meant for review and delivery copies (not a master)."""
    if encoder == "hevc_nvenc":
        cq = {"max": "14", "high": "17", "normal": "21"}.get(quality, "17")
        return ["-c:v", "hevc_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", cq, "-b:v", "0",
                "-multipass", "fullres", "-spatial-aq", "1", "-temporal-aq", "1", "-profile:v", "main10",
                "-pix_fmt", "p010le", "-tag:v", "hvc1"]
    crf = {"max": "12", "high": "15", "normal": "19"}.get(quality, "15")
    return ["-c:v", "libx265", "-preset", "slow" if quality != "normal" else "medium", "-crf", crf,
            "-pix_fmt", "yuv420p10le", "-profile:v", "main10", "-x265-params", "aq-mode=3:no-sao=1:log-level=error",
            "-tag:v", "hvc1"]

_TRC_OK = {"bt709", "gamma22", "gamma28", "smpte170m", "smpte240m", "linear", "log", "log_sqrt",
           "iec61966-2-4", "bt1361e", "iec61966-2-1", "bt2020-10", "bt2020-12", "smpte2084",
           "smpte428", "arib-std-b67"}
_PRIM_OK = {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "bt2020", "smpte428",
            "smpte431", "smpte432", "jedec-p22"}
_MAT_OK = {"bt709", "fcc", "bt470bg", "smpte170m", "smpte240m", "ycgco", "bt2020nc", "bt2020c"}


class FrameWriter:
    """Pipes raw planes into ffmpeg and encodes ProRes (or any ffmpeg codec)."""

    def __init__(self, out_path: str, width: int, height: int, fps_num: int, fps_den: int,
                 in_pix_fmt: str, profile: str = "prores_422_hq", src_info: Optional[MediaInfo] = None,
                 timecode: str = "", audio_from: Optional[Tuple[str, float, float]] = None,
                 gray: bool = False, encoder: str = "auto", quality: str = "high"):
        if profile not in VIDEO_PROFILES:
            raise MediaError(f"알 수 없는 출력 포맷: {profile}")
        prof, enc_fmt, _ = VIDEO_PROFILES[profile]
        hevc = profile == "hevc_mp4"
        alpha = in_pix_fmt.startswith("yuva")
        if alpha:
            enc_fmt = "yuva444p10le"
        self.layout = plane_layout(in_pix_fmt, width, height)
        self.out_path = out_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # what the raw planes are (YUV: the source's own matrix and range). Left unsaid, newer ffmpeg (8, 9) takes
        # untagged YUV as BT.601 and converts it to the BT.709 the output is tagged as: a plain render changed colour.
        in_tags: List[str] = []
        if in_pix_fmt.startswith("yuv") and not gray:
            m_ = src_info.color_matrix if src_info is not None else "bt709"
            m_ = m_ if m_ in _MAT_OK else ("bt709" if height >= 700 else "smpte170m")
            r_ = src_info.color_range if src_info is not None and src_info.color_range in ("tv", "pc") else "tv"
            in_tags = ["-colorspace", m_, "-color_range", r_]
        cmd = [ffmpeg_bin(), "-hide_banner", "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", in_pix_fmt, "-s", f"{width}x{height}",
               "-framerate", f"{fps_num}/{fps_den}"] + in_tags + ["-i", "-"]
        if audio_from:
            apath, astart, adur = audio_from
            cmd += ["-ss", f"{max(0.0, astart):.6f}", "-t", f"{adur:.6f}", "-i", apath]
        cmd += ["-map", "0:v:0"]
        if audio_from:
            cmd += ["-map", "1:a?"] + (["-c:a", "aac", "-b:a", "320k"] if hevc else ["-c:a", "pcm_s24le"])
        matrix = "bt709"
        if src_info is not None:
            m = src_info.color_matrix
            matrix = m if m in _MAT_OK else ("bt709" if height >= 700 else "smpte170m")
        vf = None
        if in_pix_fmt.startswith(("rgb", "gbr")) or in_pix_fmt.startswith("gray"):
            sws_m = _MATRIX_MAP.get(matrix, "bt709")
            if not gray:
                vf = f"scale=out_color_matrix={sws_m}:out_range=tv:flags=accurate_rnd+full_chroma_int"
            else:
                vf = "scale=out_range=tv:flags=accurate_rnd"
        elif in_pix_fmt.startswith("yuv") and src_info is not None and src_info.color_range == "pc":
            # full-range planes from a full-range source go out as limited range (what the file is tagged as and
            # what ProRes / H.265 deliverables are): converted, not just relabelled (a relabel darkened the picture)
            sws_m = _MATRIX_MAP.get(matrix, "bt709")
            vf = (f"scale=in_color_matrix={sws_m}:out_color_matrix={sws_m}:in_range=pc:out_range=tv"
                  ":flags=accurate_rnd+full_chroma_int")
        if vf:
            cmd += ["-vf", vf]
        if hevc:
            cmd += hevc_args(hevc_encoder(encoder), quality)
        else:
            cmd += ["-c:v", "prores_ks", "-profile:v", str(prof), "-vendor", "apl0", "-pix_fmt", enc_fmt]
            if alpha:
                cmd += ["-alpha_bits", "16"]
        if src_info is not None and not gray:
            if src_info.color_primaries in _PRIM_OK:
                cmd += ["-color_primaries", src_info.color_primaries]
            if src_info.color_trc in _TRC_OK:
                cmd += ["-color_trc", src_info.color_trc]
            cmd += ["-colorspace", matrix, "-color_range", "tv"]
        elif not gray:
            cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
        if timecode:
            cmd += ["-timecode", timecode]
        cmd += ["-movflags", ("+faststart+write_colr" if hevc else "+write_colr"), out_path]
        self.cmd = cmd
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      creationflags=_CREATE_NO_WINDOW)
        self._err = b""
        self._errt = threading.Thread(target=self._drain_err, daemon=True)
        self._errt.start()
        self.frames = 0

    def _drain_err(self):
        try:
            self._err = self._proc.stderr.read() or b""
        except Exception:
            pass

    def write(self, planes: Sequence[np.ndarray]):
        try:
            self._proc.stdin.write(join_planes(planes, self.layout))
        except (BrokenPipeError, OSError):
            self._proc.wait(timeout=10)
            self._errt.join(timeout=2)
            raise MediaError("인코더가 중단되었습니다: " + self._err.decode(errors="replace")[-600:])
        self.frames += 1

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        rc = self._proc.wait()
        self._errt.join(timeout=5)
        if rc != 0:
            raise MediaError("인코딩 실패: " + self._err.decode(errors="replace")[-600:])

    def abort(self) -> None:
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:
            pass
        try:
            os.remove(self.out_path)
        except OSError:
            pass


# --------------------------------------------------------------------------- EXR / image sequence writing

class SequenceWriter:
    """EXR (half float, ZIP) image sequence through ffmpeg's native EXR encoder.

    Accepts gbrpf32le planes (G, B, R) or a single grayf32le plane."""

    def __init__(self, folder: str, name: str, width: int, height: int, start_number: int, gray: bool = False):
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self.pix_fmt = "grayf32le" if gray else "gbrpf32le"
        self.layout = PlaneLayout(self.pix_fmt, [(height, width)] * (1 if gray else 3), [(1, 1)] * (1 if gray else 3),
                                  np.dtype("<f4"), 0, 1.0, True)
        pattern = str(Path(folder) / f"{name}.%07d.exr")
        cmd = [ffmpeg_bin(), "-hide_banner", "-v", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", self.pix_fmt, "-s", f"{width}x{height}", "-framerate", "24", "-i", "-",
               "-c:v", "exr", "-format", "half", "-compression", "zip16",
               "-start_number", str(int(start_number)), "-f", "image2", pattern]
        self.cmd = cmd
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                      creationflags=_CREATE_NO_WINDOW)
        self._err = b""
        self._errt = threading.Thread(target=self._drain, daemon=True)
        self._errt.start()
        self.frames = 0

    def _drain(self):
        try:
            self._err = self._proc.stderr.read() or b""
        except Exception:
            pass

    def write(self, planes: Sequence[np.ndarray]):
        try:
            self._proc.stdin.write(join_planes(planes, self.layout))
        except (BrokenPipeError, OSError):
            self._proc.wait(timeout=10)
            self._errt.join(timeout=2)
            raise MediaError("EXR 인코더가 중단되었습니다: " + self._err.decode(errors="replace")[-600:])
        self.frames += 1

    def close(self):
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        rc = self._proc.wait()
        self._errt.join(timeout=5)
        if rc != 0:
            raise MediaError("EXR 저장 실패: " + self._err.decode(errors="replace")[-600:])

    def abort(self):
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:
            pass


def unique_path(path: str) -> str:
    """Never overwrite: append _v02, _v03 ... when the path exists."""
    p = Path(path)
    if not p.exists():
        return str(p)
    stem, suf = p.stem, p.suffix
    m = re.match(r"^(.*)_v(\d+)$", stem)
    base, n = (m.group(1), int(m.group(2))) if m else (stem, 1)
    while True:
        n += 1
        cand = p.with_name(f"{base}_v{n:02d}{suf}")
        if not cand.exists():
            return str(cand)
