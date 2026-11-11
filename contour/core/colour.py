"""OCIO colour management for the viewer, proxies and analysis frames.

Rendering never passes through OCIO unless an output conversion is requested:
the render path is code-value passthrough. OCIO is used to show footage
correctly (log, wide gamut, show LUT) and to feed tracking models a
display-referred image.
"""
from __future__ import annotations

import threading
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
        img = np.ascontiguousarray(rgb, dtype=np.float32).copy()
        proc = self._processor(vs)
        if proc is not None:
            proc.applyRGB(img)
        elif vs.exposure:
            img *= 2.0 ** vs.exposure
        return img

    def to_display_u8(self, rgb: np.ndarray, vs: ViewSettings) -> np.ndarray:
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
        proc.applyRGB(img)
        return img


_manager: Optional[ColourManager] = None


def manager(config_path: str = "") -> ColourManager:
    global _manager
    if _manager is None:
        _manager = ColourManager(config_path)
    elif config_path and config_path != _manager.config_path:
        _manager.load(config_path)
    return _manager
