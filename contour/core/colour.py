"""OCIO colour management for the viewer, proxies and analysis frames.

Rendering never passes through OCIO unless an output conversion is requested:
the render path is code-value passthrough. OCIO is used to show footage
correctly (log, wide gamut, show LUT) and to feed tracking models a
display-referred image.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import PyOpenColorIO as OCIO
except Exception:  # pragma: no cover
    OCIO = None

BUILTIN_STUDIO = "ocio://studio-config-latest"
PASSTHROUGH = "__passthrough__"   # show code values as-is (no transform)


@dataclass
class ViewSettings:
    input_cs: str = PASSTHROUGH
    display: str = ""
    view: str = ""
    lut_path: str = ""          # optional show LUT (.cube/.3dl/.csp...) applied after input conversion
    exposure: float = 0.0        # stops, viewer only

    def key(self) -> str:
        return f"{self.input_cs}|{self.display}|{self.view}|{self.lut_path}|{self.exposure:.3f}"

    def to_dict(self) -> dict:
        return dict(input_cs=self.input_cs, display=self.display, view=self.view,
                    lut_path=self.lut_path, exposure=self.exposure)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ViewSettings":
        d = d or {}
        return cls(**{k: d[k] for k in ("input_cs", "display", "view", "lut_path", "exposure") if k in d})


# ----------------------------------------------------------------------------- one picture in strips (1.27)
# OCIO's CPU processor converts every pixel on its own and gives up Python's lock while it works, so the rows of a picture can be
# converted by several threads at once and the result is the same, bit for bit, as one call over the whole picture. With the ACES 2.0
# view a 4K frame takes about 4 s in one call; it was most of the time a render and an analysis spent on a frame.
STRIP_MIN_PIXELS = 1 << 20          # a picture smaller than this (1 MP) is converted in one call
_strip_pool: Optional[ThreadPoolExecutor] = None
_strip_pool_n = 0
_strip_lock = threading.Lock()


def strip_threads() -> int:
    """How many threads convert one picture: CONTOUR_COLOUR_THREADS (1: no strips), else half of the logical CPUs, 2 to 8."""
    try:
        n = int(os.environ.get("CONTOUR_COLOUR_THREADS", "") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        n = max(2, min(8, (os.cpu_count() or 2) // 2))
    return n


def _strips(n: int) -> ThreadPoolExecutor:
    global _strip_pool, _strip_pool_n
    with _strip_lock:
        if _strip_pool is None or n > _strip_pool_n:
            old = _strip_pool
            _strip_pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="colour")
            _strip_pool_n = n
            if old is not None:
                old.shutdown(wait=False)
        return _strip_pool


def _can_strip(a: np.ndarray, n: int) -> bool:
    return (n > 1 and a.ndim == 3 and a.shape[2] == 3 and a.shape[0] >= n * 8 and a.shape[0] * a.shape[1] >= STRIP_MIN_PIXELS)


def _run_strips(n: int, h: int, job) -> None:
    """job(y0, y1) for n blocks of rows, side by side; what one of them raises is raised here."""
    pool = _strips(n)
    futs = [pool.submit(job, h * i // n, h * (i + 1) // n) for i in range(n)]
    err = None
    for f in futs:
        try:
            f.result()
        except BaseException as e:      # (every strip is waited for before this goes on: none is left writing into the picture)
            err = err or e
    if err is not None:
        raise err


def apply_rgb(proc, img: np.ndarray) -> None:
    """proc.applyRGB(img) in place (float32, C-contiguous), in strips of rows when the picture is big."""
    n = strip_threads()
    if _can_strip(img, n) and img.flags.c_contiguous:
        _run_strips(n, img.shape[0], lambda y0, y1: proc.applyRGB(img[y0:y1]))
    else:
        proc.applyRGB(img)


# ---------------------------------------------------------------------------------------------- the display transform on the GPU
# CONTOUR_COLOUR_GPU=1: a view's transform (code values -> display) is read once from OpenColorIO into a 3D table of GPU_LUT_N points a side and applied
# on the GPU (torch, trilinear). The render converts every 4K frame several times (the analysis view, the log wrap, its Jacobian); on the CPU that was a
# third of the render's work while the GPU idled. The table's error on LogC4 footage is under 0.001 of the display range (measured: mean 0.00005).
# Anything that goes wrong (no torch, no CUDA, out of GPU memory) falls back to OpenColorIO on the CPU, for that call.
GPU_LUT_N = 193
_gpu_luts: Dict[int, object] = {}
_gpu_lock = threading.Lock()
_gpu_off = [False]


def gpu_wanted() -> bool:
    return (not _gpu_off[0]) and os.environ.get("CONTOUR_COLOUR_GPU", "0") not in ("", "0")


class _GpuLut:
    def __init__(self, proc, n: int):
        import torch
        self.torch = torch
        self.dev = torch.device("cuda")
        g = np.linspace(0.0, 1.0, n, dtype=np.float32)
        grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 1, 3).copy()      # (r, g, b) in C order
        for a in range(0, grid.shape[0], 1 << 20):
            proc.applyRGB(grid[a:a + (1 << 20)])
        lut = grid.reshape(n, n, n, 3)
        # grid_sample: input [1, C, D, H, W]; a point's (x, y, z) picks (W, H, D) = (b, g, r)
        self.lut = torch.from_numpy(np.ascontiguousarray(lut.transpose(3, 0, 1, 2))[None]).to(self.dev)
        self.n = n

    def apply(self, img: np.ndarray, u8: bool = False) -> np.ndarray:
        torch = self.torch
        F = torch.nn.functional
        shp = img.shape
        flat = np.ascontiguousarray(img, dtype=np.float32).reshape(-1, 3)
        out = np.empty(flat.shape, np.uint8 if u8 else np.float32)
        step = 1 << 22
        with torch.no_grad():
            for a in range(0, flat.shape[0], step):
                x = torch.from_numpy(flat[a:a + step]).to(self.dev)
                gr = (x[:, [2, 1, 0]] * 2.0 - 1.0).view(1, 1, 1, -1, 3)
                y = F.grid_sample(self.lut, gr, mode="bilinear", padding_mode="border", align_corners=True)      # [1, 3, 1, 1, P]
                y = y.view(3, -1).t()
                if u8:
                    y = (y.clamp_(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
                out[a:a + step] = y.cpu().numpy()
                del x, gr, y
        return out.reshape(shp)


def _gpu_lut(proc):
    """The GPU table of an OpenColorIO processor, or None (not wanted / not possible)."""
    if proc is None or not gpu_wanted():
        return None
    k = id(proc)
    lut = _gpu_luts.get(k)
    if lut is None:
        with _gpu_lock:
            lut = _gpu_luts.get(k)
            if lut is None:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError("no CUDA")
                    try:
                        n = int(os.environ.get("CONTOUR_COLOUR_GPU_N", "") or GPU_LUT_N)
                    except ValueError:
                        n = GPU_LUT_N
                    lut = _GpuLut(proc, n)
                    _gpu_luts[k] = lut
                except Exception as e:
                    _gpu_off[0] = True
                    print("colour: the GPU table is not used:", repr(e)[:200], flush=True)
                    return None
    return lut


class ColourManager:
    def __init__(self, config_path: str = ""):
        self._lock = threading.Lock()
        self._cache: Dict[str, object] = {}
        self.config_path = config_path or BUILTIN_STUDIO
        self.config = None
        self.error = ""
        self.load(self.config_path)

    # ----------------------------------------------------------------- config
    def load(self, path: str) -> bool:
        self._cache.clear()
        if OCIO is None:
            self.error = "OpenColorIO가 설치되지 않았습니다."
            self.config = None
            return False
        try:
            self.config = OCIO.Config.CreateFromFile(path)
            self.config_path = path
            self.error = ""
            return True
        except Exception as e:  # fall back to builtin
            self.error = f"OCIO 설정을 불러오지 못했습니다: {e}"
            try:
                self.config = OCIO.Config.CreateFromFile(BUILTIN_STUDIO)
                self.config_path = BUILTIN_STUDIO
            except Exception:
                self.config = None
            return False

    @property
    def available(self) -> bool:
        return self.config is not None

    def config_name(self) -> str:
        if not self.config:
            return "없음"
        try:
            return self.config.getName() or Path(self.config_path).name
        except Exception:
            return self.config_path

    def colorspaces(self) -> List[Tuple[str, str]]:
        """[(name, family)]"""
        if not self.config:
            return []
        out = []
        for cs in self.config.getColorSpaces():
            out.append((cs.getName(), cs.getFamily() or ""))
        return out

    def displays(self) -> List[str]:
        return list(self.config.getDisplays()) if self.config else []

    def views(self, display: str) -> List[str]:
        if not self.config or not display:
            return []
        return list(self.config.getViews(display))

    def default_display(self) -> str:
        return self.config.getDefaultDisplay() if self.config else ""

    def default_view(self, display: str = "") -> str:
        if not self.config:
            return ""
        d = display or self.default_display()
        return self.config.getDefaultView(d)

    def detect_input(self, info, name: str = "", look: bool = True) -> Tuple[str, str]:
        """The input colour space of a file, worked out by itself: (OCIO colour space or '' for Rec.709 as it is,
        why). The file name and the camera tags first (S-Log3, LogC3, LogC4, the camera maker), then the picture: log
        footage is flat (no deep black, no bright white, little colour). Float sequences are scene linear."""
        import re
        if getattr(info, "is_float", False):
            g = self.guess_input(info)
            return ("" if g == PASSTHROUGH else g), "부동소수점 시퀀스"
        txt = " ".join([str(getattr(info, "path", "")), name, str(getattr(info, "meta_text", ""))]).lower()
        names = {n for n, _ in self.colorspaces()} if self.config else set()

        def pick(*cands):
            return next((c for c in cands if c in names), "")
        slog = pick("S-Log3 S-Gamut3.Cine", "S-Log3 S-Gamut3")
        logc3, logc4 = pick("ARRI LogC3 (EI800)"), pick("ARRI LogC4")
        if re.search(r"s[-_ .]?log[-_ .]?3", txt) and slog:
            return slog, "파일 이름·태그에 S-Log3"
        if re.search(r"log[-_ .]?c[-_ .]?4", txt) and logc4:
            return logc4, "파일 이름·태그에 LogC4"
        if re.search(r"log[-_ .]?c", txt) and logc3:
            return logc3, "파일 이름·태그에 LogC"
        # camera originals (the maker in the file's tags): Sony cine cameras shoot S-Log3, ARRI LogC3 (ALEXA 35 LogC4)
        if any(w in txt for w in ("alexa 35", "alexa35", "alexa_35")) and logc4:
            return logc4, "카메라 태그 ARRI ALEXA 35"
        if re.search(r"\b(arri|alexa)\b", txt) and logc3:
            return logc3, "카메라 태그 ARRI"
        if re.search(r"\b(venice|burano|ilme-fx\d|fx[369]|pxw-fx9)\b", txt) and slog:
            return slog, "카메라 태그 Sony 시네마 카메라"
        # the picture alone cannot tell log from a flat, dark or desaturated grade (tried on the test plates: the two
        # overlap), so without a name or a tag it stays Rec.709; the user changes it in the 출력 tab
        trc = str(getattr(info, "color_trc", "") or "")
        return "", ("태그 " + trc if trc else "로그 표시 없음") + " → Rec.709"

    @staticmethod
    def short_name(cs: str) -> str:
        """A few letters for lists: Rec.709, S-Log3, LogC3, LogC4, ACEScg ..."""
        if not cs or cs == PASSTHROUGH:
            return "Rec.709"
        for k in ("S-Log3", "LogC3", "LogC4", "Apple Log", "V-Log", "D-Log", "CanonLog3", "CanonLog2", "Log3G10",
                  "ACEScct", "ACEScc", "ACEScg", "ACES2065-1"):
            if k in cs:
                return k
        return cs.split(" - ")[0][:14]

    def guess_input(self, info) -> str:
        """Heuristic default: passthrough for display-referred video, linear for float sequences."""
        if getattr(info, "is_float", False) and self.config:
            for name in ("ACEScg", "Linear Rec.709 (sRGB)"):
                if self.config.getColorSpace(name) is not None:
                    return name
        return PASSTHROUGH

    # ----------------------------------------------------------------- processing
    def _processor(self, vs: ViewSettings):
        key = vs.key()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            proc = None
            if self.config and vs.input_cs and vs.input_cs != PASSTHROUGH:
                group = OCIO.GroupTransform()
                if vs.exposure:
                    m = 2.0 ** vs.exposure
                    group.appendTransform(OCIO.MatrixTransform(OCIO.MatrixTransform.Scale([m, m, m, 1.0])))
                display = vs.display or self.default_display()
                view = vs.view or self.default_view(display)
                if vs.lut_path:
                    # input -> (show LUT expects its own input; we apply LUT directly to the input encoding)
                    group.appendTransform(OCIO.FileTransform(src=vs.lut_path,
                                                             interpolation=OCIO.INTERP_BEST))
                else:
                    dvt = OCIO.DisplayViewTransform(src=vs.input_cs, display=display, view=view)
                    group.appendTransform(dvt)
                try:
                    p = self.config.getProcessor(group)
                    proc = p.getDefaultCPUProcessor()
                except Exception as e:
                    self.error = f"색 변환 오류: {e}"
                    proc = None
            elif vs.lut_path and OCIO is not None:
                try:
                    cfg = self.config or OCIO.Config.CreateRaw()
                    p = cfg.getProcessor(OCIO.FileTransform(src=vs.lut_path, interpolation=OCIO.INTERP_BEST))
                    proc = p.getDefaultCPUProcessor()
                except Exception as e:
                    self.error = f"LUT 오류: {e}"
                    proc = None
            self._cache[key] = proc
            return proc

    def to_display(self, rgb: np.ndarray, vs: ViewSettings) -> np.ndarray:
        """float32 HxWx3 (code values) -> float32 display-referred 0..1 (copy)."""
        proc = self._processor(vs)
        lut = _gpu_lut(proc)
        if lut is not None and getattr(rgb, "ndim", 0) >= 2 and rgb.shape[-1] == 3:
            try:
                return lut.apply(rgb)
            except Exception as e:
                print("colour: GPU table failed for a call, CPU used:", repr(e)[:160], flush=True)
        img = np.ascontiguousarray(rgb, dtype=np.float32).copy()
        if proc is not None:
            apply_rgb(proc, img)
        elif vs.exposure:
            img *= 2.0 ** vs.exposure
        return img

    def to_display_u8(self, rgb: np.ndarray, vs: ViewSettings) -> np.ndarray:
        proc = self._processor(vs)
        lut = _gpu_lut(proc)
        if lut is not None and getattr(rgb, "ndim", 0) >= 2 and rgb.shape[-1] == 3:
            try:
                return lut.apply(rgb, u8=True)
            except Exception as e:
                print("colour: GPU table failed for a call, CPU used:", repr(e)[:160], flush=True)
        n = strip_threads()
        src = None
        if proc is not None:
            src = np.ascontiguousarray(rgb, dtype=np.float32)
            if not _can_strip(src, n):
                src = None
        if src is not None:
            # a strip at a time: convert, clip, scale and cut to 8 bits (the same operations in the same order as below, on a piece of the rows),
            # so the whole picture is not copied as float again
            out = np.empty(src.shape, np.uint8)

            def one(y0: int, y1: int) -> None:
                t = src[y0:y1].copy()
                proc.applyRGB(t)
                np.clip(t, 0.0, 1.0, out=t)
                t *= 255.0
                t += 0.5
                out[y0:y1] = t.astype(np.uint8)
            _run_strips(n, src.shape[0], one)
            return out
        out = self.to_display(rgb, vs)
        np.clip(out, 0.0, 1.0, out=out)
        return (out * 255.0 + 0.5).astype(np.uint8)

    def convert(self, rgb: np.ndarray, src: str, dst: str) -> np.ndarray:
        """Colourspace conversion for output (render option)."""
        if not self.config or not src or not dst or src == dst or PASSTHROUGH in (src, dst):
            return rgb
        key = f"conv|{src}|{dst}"
        with self._lock:
            proc = self._cache.get(key)
            if proc is None:
                proc = self.config.getProcessor(src, dst).getDefaultCPUProcessor()
                self._cache[key] = proc
        img = np.ascontiguousarray(rgb, dtype=np.float32).copy()
        apply_rgb(proc, img)
        return img


_manager: Optional[ColourManager] = None


def manager(config_path: str = "") -> ColourManager:
    global _manager
    if _manager is None:
        _manager = ColourManager(config_path)
    elif config_path and config_path != _manager.config_path:
        _manager.load(config_path)
    return _manager
