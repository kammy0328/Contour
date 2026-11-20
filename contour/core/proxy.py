"""Display-referred 8-bit proxy cache (memory-mapped) used by the viewer and all analysis."""
from __future__ import annotations

import json
import os
import shutil
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


SHARED = "proxy_shared.json"       # in a timeline cut's cache: its frames are a stretch of the bake's proxy
SPACE_MARGIN = 1 << 30             # free space left on the drive after a proxy or matte is written


class DiskSpaceError(RuntimeError):
    pass


def ensure_space(path: Path, total: int, what: str) -> None:
    """Before a memory-mapped file of `total` bytes is made at `path`: enough room on its drive (a full drive halfway
    through writing a mapped file stops the app with an error deep in numpy)."""
    path = Path(path)
    have = 0
    try:
        if path.exists():
            have = path.stat().st_size
    except OSError:
        pass
    need = max(0, int(total) - have)
    if need <= 0:
        return
    try:
        free = shutil.disk_usage(path.parent if path.parent.exists() else path.anchor).free
    except OSError:
        return
    if free < need + SPACE_MARGIN:
        drive = path.anchor or str(path.parent)
        raise DiskSpaceError(
            f"{what}을(를) 만들 공간이 부족합니다 ({drive}): 필요 {need / 1e9:.1f}GB, 남은 공간 {free / 1e9:.1f}GB.\n"
            "프로젝트 관리에서 쓰지 않는 프로젝트의 '프록시 캐시 정리'로 공간을 비우거나, 설정에서 캐시 폴더를 "
            "공간이 넉넉한 드라이브로 바꾸세요.")


class Proxy:
    """A whole cut decoded once into a file. Optional since 1.18: frames come from the original (frames.SourceFrames)
    when there is none; this file only makes repeated access faster."""
    kind = "proxy"

    def __init__(self, cache_dir: Path, info: MediaInfo, long_side: int = 1920, size: Optional[Tuple[int, int]] = None):
        self.dir = Path(cache_dir)
        self.info = info
        self.w, self.h = (int(size[0]), int(size[1])) if size else proxy_size(info, long_side)
        self.n = info.frames
        self.offset = 0                 # first frame of this cut inside the (shared) proxy file
        self.n_file = self.n            # frames in the proxy file
        self.meta_path = self.dir / "proxy_meta.json"
        shared = self._shared()
        if shared is not None:
            # a stretch of the edited timeline's bake: the bake's one proxy serves every shot
            self.path, self.meta_path = Path(shared["path"]), Path(shared["meta"])
            self.offset, self.n_file = int(shared["offset"]), int(shared["n"])
            self.w, self.h = int(shared["w"]), int(shared["h"])
        else:
            m = self.meta()
            if size is None and m.get("complete") and m.get("n") == self.n and m.get("w") and m.get("h"):
                # keep the size this cut's proxy was made at (the default may have changed since)
                self.w, self.h = int(m["w"]), int(m["h"])
            self.path = self.dir / f"proxy_{self.w}x{self.h}.u8"
        self.sx = self.w / info.width
        self.sy = self.h / info.height
        self._mm: Optional[np.memmap] = None
        self._gray_cache: dict = {}

    def _shared(self) -> Optional[dict]:
        f = self.dir / SHARED
        if not f.exists():
            return None
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            m = json.loads(Path(d["meta"]).read_text(encoding="utf-8"))
        except Exception:
            return None
        if not (Path(d["path"]).exists() and m.get("complete")):
            return None
        if int(d["offset"]) + self.n > int(m.get("n", 0)):
            return None
        return dict(d, w=m["w"], h=m["h"], n=m["n"])

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
        return (self.has_frames() and m.get("view") == view_key
                and m.get("w") == self.w and m.get("h") == self.h and m.get("n") == self.n_file
                and m.get("sig") == self._source_sig())

    def has_frames(self) -> bool:
        """Every frame is in the file: a build that was cancelled, failed or decoded too few frames is not (its frames
        past that point are black). Used before analysis, render and export."""
        m = self.meta()
        if not (self.path.exists() and m.get("complete") and m.get("w") == self.w and m.get("h") == self.h
                and m.get("n") == self.n_file):
            return False
        if int(m.get("n_decoded", self.n_file) or 0) < self.n_file:
            return False                    # made by 1.15 or older from a file that ended early: a black tail
        try:
            return self.path.stat().st_size >= self.n_file * self.w * self.h * 3
        except OSError:
            return False

    @property
    def shared(self) -> bool:
        return self.n_file != self.n or self.offset != 0

    def open(self) -> None:
        if self._mm is None and self.path.exists():
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.n_file, self.h, self.w, 3))

    def close(self) -> None:
        self._mm = None
        self._gray_cache.clear()
        sc = self.__dict__.get("_src_crops")        # pieces of the original read for this cut (frames.SourceCrops)
        if sc is not None:
            sc.close()

    # ------------------------------------------------------------------ access
    def frame(self, i: int) -> np.ndarray:
        self.open()
        if self._mm is None:
            return np.zeros((self.h, self.w, 3), np.uint8)
        i = int(np.clip(i, 0, self.n - 1)) + self.offset
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
        if self.shared:
            # never rewrite the bake's shared proxy from one shot: build this shot its own
            try:
                (self.dir / SHARED).unlink()
            except OSError:
                pass
            self.offset, self.n_file = 0, self.n
            self.meta_path = self.dir / "proxy_meta.json"
            self.path = self.dir / f"proxy_{self.w}x{self.h}.u8"
        self.close()
        self.dir.mkdir(parents=True, exist_ok=True)
        ensure_space(self.path, self.n * self.h * self.w * 3, "프록시")
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
            # fewer frames than the file says: the rest of the proxy would be black (analysed as an empty picture,
            # shown as a black tail). Never marked complete; the cut cannot be used until the file is fixed.
            meta["n_decoded"] = done
            self.meta_path.write_text(json.dumps(meta), encoding="utf-8")
            if done == 0:
                raise RuntimeError("프레임을 디코딩하지 못했습니다. " + err[:300])
            raise RuntimeError(f"영상에서 {done}/{self.n} 프레임만 읽혔습니다. 파일이 잘렸거나 메타데이터의 프레임 수가 "
                               f"실제와 다릅니다. 원본을 다시 내보낸 뒤 불러오세요. {err[:200]}")
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
        self.live_path = self.path          # the file in use (path is the stage's while a new version is written)
        self.stage_removed = False          # the new version has no such file (remove() while staged)
        self._mm = None

    # ---- a whole new version written beside the one in use (1.22.3: a new analysis or a matte made again writes
    # here; the file in use stays as it is until track.matte_rev activates the new version, and for good when the
    # work is cancelled or fails)
    @property
    def stage_path(self) -> Path:
        return self.live_path.with_name(self.live_path.name + ".stage")

    @property
    def staged(self) -> bool:
        return self.path != self.live_path

    def begin_stage(self, copy: bool = True) -> None:
        """From here on this store reads and writes the stage: a copy of the file in use (copy=False, or no file
        yet: nothing until create() makes it)."""
        import shutil
        self._mm = None
        self.stage_removed = False
        try:
            self.stage_path.unlink()
        except OSError:
            pass
        live = self.live_path
        if copy and live.exists() and live.stat().st_size == self.n * self.w * self.h:
            ensure_space(self.stage_path, self.n * self.h * self.w, "인물 매트")
            shutil.copyfile(live, self.stage_path)
        self.path = self.stage_path

    def end_stage(self, drop: bool = True) -> None:
        """Back on the file in use. drop: the stage file is deleted (cancelled, failed, or already activated)."""
        self._mm = None
        if drop:
            try:
                self.stage_path.unlink()
            except OSError:
                pass
        self.path = self.live_path
        self.stage_removed = False

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size == self.n * self.w * self.h

    def create(self) -> np.memmap:
        self._mm = None
        ensure_space(self.path, self.n * self.h * self.w, "인물 매트")
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
        if self.staged:
            self.stage_removed = True       # the file in use goes only when the new version is activated
        try:
            self.path.unlink()
        except OSError:
            pass

    # ---- rebuilt beside the one in use, swapped in only when complete (clothing masks, 1.22.1)
    @property
    def tmp_path(self) -> Path:
        return self.path.with_name(self.path.name + ".tmp")

    def open_tmp(self) -> np.memmap:
        """A writable copy of the store (empty when there is none) next to it; the store in use is not touched."""
        import shutil
        ensure_space(self.tmp_path, self.n * self.h * self.w, "옷 영역")
        if self.exists():
            shutil.copyfile(self.path, self.tmp_path)
            return np.memmap(self.tmp_path, dtype=np.uint8, mode="r+", shape=(self.n, self.h, self.w))
        return np.memmap(self.tmp_path, dtype=np.uint8, mode="w+", shape=(self.n, self.h, self.w))

    def commit_tmp(self) -> None:
        """The finished copy replaces the store in use (retried while a reader on Windows still has it mapped)."""
        import os
        import time as _t
        for i in range(60):
            self._mm = None
            try:
                os.replace(self.tmp_path, self.path)
                return
            except PermissionError:
                if i == 59:
                    raise
                _t.sleep(0.05)

    def drop_tmp(self) -> None:
        try:
            self.tmp_path.unlink()
        except OSError:
            pass


def link_shared(cut_cache_dir: Path, bake_proxy: "Proxy", offset: int) -> None:
    """Points a timeline shot at the bake's proxy (frames offset .. offset + shot length)."""
    d = Path(cut_cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    if d.resolve() != Path(bake_proxy.dir).resolve():
        # a proxy of its own made while the shot was in another colour space: no longer used
        for f in list(d.glob("proxy_*.u8")) + [d / "proxy_meta.json"]:
            try:
                f.unlink()
            except OSError:
                pass
    (d / SHARED).write_text(json.dumps(dict(path=str(bake_proxy.path), meta=str(bake_proxy.meta_path),
                                            offset=int(offset))), encoding="utf-8")
