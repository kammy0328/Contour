"""Per-cut tracking arrays (raw model output, fused tracks, flags, widths, camera motion)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import skeleton as S
from ..warp.body import CONTOUR_K, N_SEG, NC


NF = 68          # face landmarks (iBUG 68, from RTMW + the dedicated face model)
NH = 42          # hand points: 21 per hand (person's left hand first, as in COCO-WholeBody)

VIS_HIDDEN = 0
VIS_VISIBLE = 1
VIS_OUTSIDE = 2  # beyond the frame edge: kept for limb direction only

PER_PERSON = ("boxes", "present", "blur", "meas", "disagree", "auto", "final", "flags", "conf", "auto_conf", "scale",
              "widths", "adapt_bias", "vis", "vis_meas", "face_raw", "face", "face_vis",
              "face106_raw", "hand_raw", "hand21_raw", "hand", "contour_raw", "contour", "facing")


class TrackStore:
    VERSION = 5

    def __init__(self, n: int, pids: Optional[List[int]] = None):
        self.n = int(n)
        self.pids: List[int] = list(pids or [])
        P = len(self.pids)
        self.raw: Dict[str, np.ndarray] = {}         # key -> (P, N, K, 3)
        self.boxes = np.full((P, n, 4), np.nan, np.float32)
        self.present = np.zeros((P, n), bool)
        self.blur = np.ones((P, n), np.float32)
        self.meas = np.full((P, n, S.K, 3), np.nan, np.float32)      # fused per-frame measurement
        self.disagree = np.zeros((P, n, S.K), np.float32)
        self.auto = np.full((P, n, S.K, 3), np.nan, np.float32)      # automatic result (no user input)
        self.final = np.full((P, n, S.K, 3), np.nan, np.float32)     # with corrections
        self.flags = np.zeros((P, n, S.K), np.uint8)
        self.conf = np.zeros((P, n), np.float32)                    # per-frame confidence
        self.auto_conf = np.zeros((P, n), np.float32)
        self.scale = np.full((P, n), np.nan, np.float32)
        self.widths = np.full((P, n, N_SEG, 2), np.nan, np.float32)
        self.homog = np.tile(np.eye(3, dtype=np.float64), (n, 1, 1))  # maps frame t -> t+1 (source px)
        self.homog_ok = np.zeros(n, bool)
        self.jump = np.zeros(n, bool)                               # picture jumps between t and t+1
        self.adapt_bias = np.zeros((P, S.K, 2), np.float32)         # learned per-joint bias (person-local units)
        self.adapt_info: Dict[str, float] = {}
        self.vis = np.ones((P, n, S.K), np.uint8)                   # final visibility (VIS_*)
        self.vis_meas = np.ones((P, n, S.K), np.uint8)              # visibility from the models alone
        self.face_raw = np.full((P, n, NF, 3), np.nan, np.float32)  # RTMW face landmarks per frame
        self.face = np.full((P, n, NF, 3), np.nan, np.float32)      # smoothed face landmarks
        self.face_vis = np.zeros((P, n), np.float32)                # 0..1 how frontal/visible the face is
        self.face106_raw = np.full((P, n, NF, 3), np.nan, np.float32)  # dedicated face model, mapped to 68
        self.hand_raw = np.full((P, n, NH, 3), np.nan, np.float32)     # RTMW hand points
        self.hand21_raw = np.full((P, n, NH, 3), np.nan, np.float32)   # dedicated hand model
        self.hand = np.full((P, n, NH, 3), np.nan, np.float32)         # tracked hand points
        self.contour_raw = np.full((P, n, NC, CONTOUR_K, 2), np.nan, np.float32)  # body edge distances (measured)
        self.contour = np.full((P, n, NC, CONTOUR_K, 2), np.nan, np.float32)      # tracked body edge distances
        self.facing = np.full((P, n), np.nan, np.float32)                # +1 facing the camera .. -1 back to it
        self.has_vis = False
        self.has_face = False
        self.has_hands = False
        self.has_contour = False
        self.removed: Dict[int, np.ndarray] = {}                       # deleted person id -> boxes (n,4)

    # ------------------------------------------------------------------ persons
    def index(self, pid: int) -> int:
        return self.pids.index(pid)

    def has(self, pid: int) -> bool:
        return pid in self.pids

    def keep_persons(self, pids: List[int]) -> None:
        idx = [self.pids.index(p) for p in pids if p in self.pids]
        for name in PER_PERSON:
            setattr(self, name, getattr(self, name)[idx])
        for k in list(self.raw):
            self.raw[k] = self.raw[k][idx]
        self.pids = [self.pids[i] for i in idx]

    def merge_persons(self, keep_pid: int, drop_pid: int) -> None:
        """Merge `drop` into `keep` (frames where keep is absent take drop's data)."""
        a, b = self.index(keep_pid), self.index(drop_pid)
        take = ~self.present[a] & self.present[b]
        for name in PER_PERSON:
            if name in ("present", "adapt_bias"):
                continue
            arr = getattr(self, name)
            arr[a][take] = arr[b][take]
        for k in self.raw:
            self.raw[k][a][take] = self.raw[k][b][take]
        self.present[a] |= self.present[b]
        self.keep_persons([p for p in self.pids if p != drop_pid])

    def swap_persons(self, pid_a: int, pid_b: int, t0: int, t1: int) -> None:
        """Exchange everything tracked for two people over frames t0..t1-1."""
        a, b = self.index(pid_a), self.index(pid_b)
        sl = slice(int(t0), int(t1))
        for name in PER_PERSON:
            if name == "adapt_bias":
                continue
            arr = getattr(self, name)
            tmp = arr[a, sl].copy()
            arr[a, sl] = arr[b, sl]
            arr[b, sl] = tmp
        for k, arr in self.raw.items():
            tmp = arr[a, sl].copy()
            arr[a, sl] = arr[b, sl]
            arr[b, sl] = tmp
        self.invalidate_cache()

    def add_person(self, pid: int) -> None:
        """A new, empty person row."""
        blank = TrackStore(self.n, [pid])
        for name in PER_PERSON:
            mine, new = getattr(self, name), getattr(blank, name)
            if new.shape[1:] != mine.shape[1:]:
                new = np.zeros((1,) + mine.shape[1:], mine.dtype) if mine.dtype == bool else \
                    np.full((1,) + mine.shape[1:], np.nan if mine.dtype.kind == "f" else 0, mine.dtype)
            setattr(self, name, np.concatenate([mine, new.astype(mine.dtype)], 0))
        for k in list(self.raw):
            r = self.raw[k]
            self.raw[k] = np.concatenate([r, np.full((1,) + r.shape[1:], np.nan, r.dtype)], 0)
        self.pids = list(self.pids) + [int(pid)]
        self.invalidate_cache()

    # ------------------------------------------------------------------ io
    def save(self, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp.npz")
        data = dict(version=np.array(self.VERSION), n=np.array(self.n), pids=np.array(self.pids, np.int32),
                    homog=self.homog, homog_ok=self.homog_ok, jump=self.jump, has_vis=np.array(self.has_vis),
                    has_face=np.array(self.has_face), has_hands=np.array(self.has_hands),
                    has_contour=np.array(self.has_contour))
        for name in PER_PERSON:
            data[name] = getattr(self, name)
        for k, v in self.raw.items():
            data["raw__" + k] = v
        for pid, v in self.removed.items():
            data[f"removed__{pid}"] = v
        with open(tmp, "wb") as f:
            np.savez(f, **data)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Optional["TrackStore"]:
        path = Path(path)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as d:
                st = cls(int(d["n"]), [int(x) for x in d["pids"]])
                for name in PER_PERSON + ("homog", "homog_ok", "jump"):
                    if name in d.files:
                        arr = d[name]
                        if name in ("contour", "contour_raw") and arr.shape != getattr(st, name).shape:
                            # outline segments were added in a newer version: keep the ones that still match
                            cur = getattr(st, name)
                            if arr.ndim == cur.ndim and arr.shape[:2] == cur.shape[:2] and arr.shape[3:] == cur.shape[3:]:
                                k = min(arr.shape[2], cur.shape[2])
                                cur[:, :, :k] = arr[:, :, :k]
                            continue
                        setattr(st, name, arr.copy())
                st.has_vis = bool(d["has_vis"]) if "has_vis" in d.files else False
                st.has_face = bool(d["has_face"]) if "has_face" in d.files else False
                st.has_hands = bool(d["has_hands"]) if "has_hands" in d.files else False
                st.has_contour = bool(d["has_contour"]) if "has_contour" in d.files else False
                if "contour" in d.files and d["contour"].shape != st.contour.shape:
                    st.has_contour = False
                for k in d.files:
                    if k.startswith("raw__"):
                        st.raw[k[5:]] = d[k].copy()
                    elif k.startswith("removed__"):
                        st.removed[int(k[9:])] = d[k].copy()
        except Exception:
            return None
        return st

    # ------------------------------------------------------------------ queries
    def kp(self, pid: int, frame: int, which: str = "final") -> Optional[np.ndarray]:
        if pid not in self.pids or not (0 <= frame < self.n):
            return None
        p = self.index(pid)
        if not self.present[p, frame]:
            return None
        arr = getattr(self, which)[p, frame]
        if not np.isfinite(arr[:, :2]).any():
            return None
        return arr

    def homography(self, f0: int, f1: int) -> Optional[np.ndarray]:
        """Source-px homography mapping points in frame f0 to frame f1 (None across a picture jump)."""
        if f0 == f1:
            return np.eye(3)
        if 0 <= min(f0, f1) and max(f0, f1) <= self.n and not self.connected(f0, f1):
            return None
        if abs(f1 - f0) <= 12:
            H = np.eye(3)
            if f1 > f0:
                for t in range(f0, f1):
                    H = self.homog[t] @ H
            else:
                for t in range(f0 - 1, f1 - 1, -1):
                    H = np.linalg.inv(self.homog[t]) @ H
            return H
        cum = getattr(self, "_cum", None)
        if cum is None or len(cum) != self.n:
            cum = np.empty((self.n, 3, 3))
            cum[0] = np.eye(3)
            for t in range(1, self.n):
                cum[t] = self.homog[t - 1] @ cum[t - 1]
                cum[t] /= cum[t][2, 2]
            self._cum = cum
        H = cum[f1] @ np.linalg.inv(cum[f0])
        return H / H[2, 2]

    def breaks(self) -> set:
        """Frames t with a jump between t and t+1 (punch-in zoom, flash frame): no smoothing across."""
        return set(int(x) for x in np.nonzero(self.jump)[0])

    def connected(self, f0: int, f1: int) -> bool:
        lo, hi = min(f0, f1), max(f0, f1)
        return not self.jump[lo:hi].any()

    def invalidate_cache(self) -> None:
        self._cum = None
