"""Display-referred 8-bit proxy cache (memory-mapped) used by the viewer and all analysis."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .colour import ColourManager, ViewSettings
from .media import FrameReader, MediaInfo, planes_to_rgb_float


def proxy_size(info: MediaInfo, long_side: int) -> Tuple[int, int]:
    w, h = info.width, info.height
    s = min(1.0, float(long_side) / max(w, h))
    pw = max(2, int(round(w * s / 2)) * 2)
    ph = max(2, int(round(h * s / 2)) * 2)
    return pw, ph


class Proxy:
    def __init__(self, cache_dir: Path, info: MediaInfo, long_side: int = 1920):
        self.dir = Path(cache_dir)
        self.info = info
        self.w, self.h = proxy_size(info, long_side)
        self.n = info.frames
        self.sx = self.w / info.width
        self.sy = self.h / info.height
        self.path = self.dir / f"proxy_{self.w}x{self.h}.u8"
        self.meta_path = self.dir / "proxy_meta.json"
        self._mm: Optional[np.memmap] = None
        self._gray_cache: dict = {}

    # ------------------------------------------------------------------ state
    def meta(self) -> dict:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _source_sig(self) -> str:
        try:
            st = os.stat(self.info.path if self.info.kind == "video" else self.info.seq_pattern % self.info.seq_start)
            return f"{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            return ""

    def is_valid(self, view_key: str) -> bool:
        m = self.meta()
        return (self.path.exists() and m.get("complete") and m.get("view") == view_key
                and m.get("w") == self.w and m.get("h") == self.h and m.get("n") == self.n
                and m.get("sig") == self._source_sig())

    def has_frames(self) -> bool:
        m = self.meta()
        return self.path.exists() and m.get("w") == self.w and m.get("h") == self.h and m.get("n") == self.n

    def open(self) -> None:
        if self._mm is None and self.path.exists():
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.n, self.h, self.w, 3))

    def close(self) -> None:
        self._mm = None
        self._gray_cache.clear()

    # ------------------------------------------------------------------ access
    def frame(self, i: int) -> np.ndarray:
        self.open()
        if self._mm is None:
            return np.zeros((self.h, self.w, 3), np.uint8)
        i = int(np.clip(i, 0, self.n - 1))
        return np.array(self._mm[i])      # copy: never hand out views of the mapping (Windows file locks)

    def gray(self, i: int) -> np.ndarray:
        g = self._gray_cache.get(i)
        if g is None:
            g = cv2.cvtColor(np.asarray(self.frame(i)), cv2.COLOR_RGB2GRAY)
            if len(self._gray_cache) > 64:
                self._gray_cache.pop(next(iter(self._gray_cache)))
            self._gray_cache[i] = g
        return g

    def to_proxy(self, xy: np.ndarray) -> np.ndarray:
        out = np.array(xy, dtype=np.float32, copy=True)
        out[..., 0] *= self.sx
        out[..., 1] *= self.sy
        return out

    def to_source(self, xy: np.ndarray) -> np.ndarray:
        out = np.array(xy, dtype=np.float32, copy=True)
        out[..., 0] /= self.sx
        out[..., 1] /= self.sy
        return out

    # ------------------------------------------------------------------ build
    def build(self, cm: ColourManager, vs: ViewSettings, progress: Optional[Callable[[float, str], None]] = None,
              cancel: Optional[Callable[[], bool]] = None) -> bool:
        self.close()
        self.dir.mkdir(parents=True, exist_ok=True)
        mm = np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(self.n, self.h, self.w, 3))
        pf = "gbrpf32le" if self.info.is_float else "rgb48le"
        meta = {"w": self.w, "h": self.h, "n": self.n, "view": vs.key(), "complete": False, "sig": self._source_sig(),
                "built": time.time()}
        self.meta_path.write_text(json.dumps(meta), encoding="utf-8")
        t0 = time.time()
        done = 0
        with FrameReader(self.info, 0, self.n, pf, size=(self.w, self.h)) as reader:
            for i, planes in enumerate(reader):
                if cancel and cancel():
                    del mm
                    return False
                rgb = planes_to_rgb_float(planes, reader.layout)
                mm[i] = cm.to_display_u8(rgb, vs)
                done = i + 1
                if progress and (i % 8 == 0 or i == self.n - 1):
                    el = time.time() - t0
                    fps = done / max(el, 1e-3)
                    progress(done / self.n, f"프록시 생성 {done}/{self.n} ({fps:.0f} fps)")
            err = reader.error_text
        mm.flush()
        del mm
        if done < self.n:
            # Some containers report more frames than they decode; keep what we have.
            meta["n_decoded"] = done
            if done == 0:
                raise RuntimeError("프레임을 디코딩하지 못했습니다. " + err[:300])
        meta["complete"] = True
        self.meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return True

    def thumbnail(self, i: int, width: int = 160) -> np.ndarray:
        f = np.asarray(self.frame(i))
        h = max(1, int(round(f.shape[0] * width / f.shape[1])))
        return cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)


class MatteStore:
    """Per-frame combined person matte (uint8) at half proxy resolution.

    The same layout keeps the SAM 2 person labels (prefix 'seglab': 0 = nobody, else person id + 1)."""

    def __init__(self, cache_dir: Path, n: int, w: int, h: int, prefix: str = "matte"):
        self.n, self.w, self.h = n, w, h
        self.path = Path(cache_dir) / f"{prefix}_{w}x{h}.u8"
        self._mm = None

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size == self.n * self.w * self.h

    def create(self) -> np.memmap:
        self._mm = None
        return np.memmap(self.path, dtype=np.uint8, mode="w+", shape=(self.n, self.h, self.w))

    def get(self, i: int) -> Optional[np.ndarray]:
        if not self.exists():
            return None
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.n, self.h, self.w))
        return np.array(self._mm[int(np.clip(i, 0, self.n - 1))])

    def close(self):
        self._mm = None

    def open_rw(self) -> np.memmap:
        """Write access to an existing store (creates it when missing)."""
        self._mm = None
        if not self.exists():
            return self.create()
        return np.memmap(self.path, dtype=np.uint8, mode="r+", shape=(self.n, self.h, self.w))

    def remove(self) -> None:
        self._mm = None
        try:
            self.path.unlink()
        except OSError:
            pass
